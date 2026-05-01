"""
train_hf.py — RunPod training script backed by the jpena-173/sm-eyes-v1 HuggingFace Parquet dataset.

Parquet columns used:
  look_vec (4,)          gaze direction in world space (same as JSON eye_details.look_vec)
  pupil_size float        normalised pupil diameter
  iris_2d    (32, 3)     iris outline points in image-pixel space (x, y, z)
  upper_eyelid_2d (8, 3) upper margin points  (mirrors JSON upper_interior_margin_2d)
  lower_eyelid_2d (8, 3) lower margin points
  image.bytes            JPEG-encoded 640×480 image

Everything else (transforms, model, losses, checkpointing) matches train.py exactly.
"""

import os, glob, io, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import decode_image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from huggingface_hub import snapshot_download

from model import ShinraCNN, HEATMAP_BORDER
from synthetic import _to_hm, render_gaussian, derive_pupil_stats, INPUT_H, INPUT_W
from dataset import SyntheticTransform
from early_stopping import EarlyStopping

# ── Config ────────────────────────────────────────────────────────────────────
HF_REPO      = 'jpena-173/sm-eyes-v1'
DATA_DIR     = './sm-eyes-v1'
BATCH_SIZE   = 128
NUM_WORKERS  = 8
NUM_EPOCHS   = 80
ES_PATIENCE  = 5
LOSS_WEIGHTS = {'diameter': .85, 'heatmaps': 700, 'contour': 2, 'gaze': .85}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset download ──────────────────────────────────────────────────────────
def get_parquet_files():
    existing = sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.parquet'), recursive=True))
    if existing:
        print(f'Found {len(existing)} local shard(s) in {DATA_DIR}, skipping download.')
        return existing
    print(f'Downloading {HF_REPO} to {DATA_DIR} …')
    snapshot_download(repo_id=HF_REPO, repo_type='dataset', local_dir=DATA_DIR)
    shards = sorted(glob.glob(os.path.join(DATA_DIR, '**', '*.parquet'), recursive=True))
    print(f'Downloaded {len(shards)} shard(s).')
    return shards


# ── Label conversion ──────────────────────────────────────────────────────────
def convert_parquet(row, crop_xy=(0, 0), flipped=False, resize=(224, 224)):
    """Build GT tensors from a Parquet row. Mirrors synthetic.convert() for JSON."""
    gt = {}
    gx, gy, gz, _ = row['look_vec']
    gt['gaze_vector'] = torch.tensor([-gx if flipped else gx, gy, gz], dtype=torch.float32)

    pupil_x, pupil_y, gt['pupil_diameter'] = derive_pupil_stats(row['iris_2d'], row['pupil_size'])

    eye_heatmaps = []
    hm_x, hm_y = _to_hm(pupil_x, pupil_y, crop_xy, flipped, resize)
    eye_heatmaps.append(render_gaussian(INPUT_H, INPUT_W, hm_x, hm_y, 1.5))

    upper = list(row['upper_eyelid_2d'])
    lower = list(row['lower_eyelid_2d'])
    if flipped:
        upper = upper[::-1]
        lower = lower[::-1]

    for pts in (upper, lower):
        for pt in pts:
            hm_x, hm_y = _to_hm(pt[0], pt[1], crop_xy, flipped, resize)
            eye_heatmaps.append(render_gaussian(INPUT_H, INPUT_W, hm_x, hm_y, 2))

    gt['eye_heatmaps'] = torch.stack(eye_heatmaps)  # (17, 112, 112)
    return gt


# ── Dataset ───────────────────────────────────────────────────────────────────
class ParquetEyeDS(Dataset):
    def __init__(self, parquet_files, transforms=None):
        self.transforms = transforms
        print(f'Loading {len(parquet_files)} shard(s) into memory …')
        self.df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files],
            ignore_index=True,
        )
        print(f'Dataset ready: {len(self.df):,} samples.')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_np = np.frombuffer(row['image']['bytes'], dtype=np.uint8).copy()
        img    = decode_image(torch.from_numpy(img_np))  # (C, H, W) uint8

        crop_xy = (0, 0)
        flipped = False
        resize  = (224, 224)

        if self.transforms is not None:
            img, crop_xy, flipped = self.transforms(img)
            resize = (256, 256)

        return img, convert_parquet(row, crop_xy=crop_xy, flipped=flipped, resize=resize)


# ── Losses ────────────────────────────────────────────────────────────────────
def focal_loss(logits, targets, gamma=2, alpha=0.9):
    bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t    = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss   = alpha_t * (1 - p_t) ** gamma * bce

    m = HEATMAP_BORDER
    mask = torch.ones_like(loss)
    mask[..., :m, :] = 0;  mask[..., -m:, :] = 0
    mask[..., :, :m] = 0;  mask[..., :, -m:] = 0
    focal = (loss * mask).sum() / mask.sum()

    pred_peak   = logits.amax(dim=(-2, -1), keepdim=True)
    gt_peak     = targets.amax(dim=(-2, -1), keepdim=True)
    pred_region = F.relu(logits   - pred_peak * 0.75)
    gt_region   = F.relu(targets  - gt_peak   * 0.75)
    active      = pred_peak > 0.1
    contour_loss = F.mse_loss(pred_region * active, gt_region * active)

    return focal + LOSS_WEIGHTS['contour'] * contour_loss


def heteroscedastic_loss(pred, truth, log_var, weight=None):
    precision = torch.exp(-log_var)
    sq_err = (pred - truth) ** 2
    if weight is not None:
        sq_err = sq_err * weight
    return (precision * sq_err + log_var).mean(), sq_err.mean().item(), log_var.mean().item()


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]),
        reverse=True,
    )
    for phase_dir in phase_dirs:
        phase_idx = int(phase_dir.split('_')[1])
        ckpts = sorted(
            [f for f in os.listdir(phase_dir) if f.startswith('shinra_checkpoint_') and f.endswith('.pth')],
            key=lambda f: int(f.split('_')[-1].split('.')[0]),
        )
        if ckpts:
            ckpt_path = os.path.join(phase_dir, ckpts[-1])
            epoch     = int(ckpts[-1].split('_')[-1].split('.')[0])
            return ckpt_path, phase_idx, epoch
    return None, 0, 0


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    shards  = get_parquet_files()
    full_ds = ParquetEyeDS(shards, transforms=SyntheticTransform())
    n_train = int(len(full_ds) * 0.8)
    train_ds, val_ds = random_split(full_ds, [n_train, len(full_ds) - n_train])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )

    torch.backends.cudnn.benchmark = True

    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    shinra   = ShinraCNN(backbone, out_channels=17).to(device)

    ckpt_path, start_phase, _ = find_latest_checkpoint()
    thaw_gen = shinra.thaw()
    _, optim_groups = next(thaw_gen)
    if start_phase > 1:
        for _ in range(start_phase - 1):
            _, optim_groups = next(thaw_gen)

    optimizer = optim.Adam(optim_groups, lr=1e-3)
    if ckpt_path:
        print(f'Resuming from {ckpt_path} (phase {start_phase})')
        ckpt = torch.load(ckpt_path, map_location=device)
        shinra.load_state_dict(ckpt['shinra'])
        start_epoch = ckpt['epoch'] + 1
    else:
        print('Starting from scratch.')
        start_epoch = 0

    scaler  = torch.amp.GradScaler('cuda')
    stopper = EarlyStopping(ES_PATIENCE)

    for phase_idx, (phase, optim_groups) in enumerate(thaw_gen, start=start_phase):
        print(f'\n── PHASE {phase_idx} ──────────────────────────────────────')
        phase_dir = f'phase_{phase_idx}'
        os.makedirs(phase_dir, exist_ok=True)

        optimizer = optim.Adam(optim_groups, lr=1e-3)
        stopper.reset()

        for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
            epoch_losses = {k: [] for k in LOSS_WEIGHTS}
            pred_times   = []

            shinra.train()
            for batch_idx, (imgs, lbls) in enumerate(train_loader):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    t0 = time.monotonic_ns()
                    (diam_val, diam_lv), hm_val, (gaze_val, gaze_lv) = shinra(imgs.to(device))
                    if batch_idx > 0:
                        pred_times.append(time.monotonic_ns() - t0)

                    diam_gt = lbls['pupil_diameter'].to(device, dtype=torch.float32).unsqueeze(-1)
                    hm_gt   = lbls['eye_heatmaps'].to(device,   dtype=torch.float32)
                    gaze_gt = lbls['gaze_vector'].to(device,    dtype=torch.float32)

                    diameter_loss, _, _ = heteroscedastic_loss(diam_val, diam_gt, diam_lv)
                    heatmap_loss        = focal_loss(hm_val, hm_gt)
                    gaze_loss,     _, _ = heteroscedastic_loss(gaze_val, gaze_gt, gaze_lv)
                    total_loss          = (LOSS_WEIGHTS['diameter'] * diameter_loss
                                        + LOSS_WEIGHTS['heatmaps'] * heatmap_loss
                                        + LOSS_WEIGHTS['gaze']     * gaze_loss)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses['diameter'].append(diameter_loss.item() * LOSS_WEIGHTS['diameter'])
                epoch_losses['heatmaps'].append(heatmap_loss.item() * LOSS_WEIGHTS['heatmaps'])
                epoch_losses['gaze'].append(gaze_loss.item()        * LOSS_WEIGHTS['gaze'])

                if batch_idx % 25 == 0:
                    avg_inf = np.mean(pred_times) * 1e-6 if pred_times else 0.0
                    print(f'EPOCH {epoch} batch {batch_idx}: '
                          f'diam {np.mean(epoch_losses["diameter"]):.4f}  '
                          f'hmap {np.mean(epoch_losses["heatmaps"]):.4f}  '
                          f'gaze {np.mean(epoch_losses["gaze"]):.4f}  '
                          f'| inf {avg_inf:.2f}ms')

            shinra.eval()
            val_losses = []
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    (diam_val, diam_lv), hm_val, (gaze_val, gaze_lv) = shinra(imgs.to(device))
                    diam_gt = lbls['pupil_diameter'].to(device, dtype=torch.float32).unsqueeze(-1)
                    hm_gt   = lbls['eye_heatmaps'].to(device,   dtype=torch.float32)
                    gaze_gt = lbls['gaze_vector'].to(device,    dtype=torch.float32)

                    diameter_loss, _, _ = heteroscedastic_loss(diam_val, diam_gt, diam_lv)
                    heatmap_loss        = focal_loss(hm_val, hm_gt)
                    gaze_loss,     _, _ = heteroscedastic_loss(gaze_val, gaze_gt, gaze_lv)
                    val_losses.append((LOSS_WEIGHTS['diameter'] * diameter_loss
                                     + LOSS_WEIGHTS['heatmaps'] * heatmap_loss
                                     + LOSS_WEIGHTS['gaze']     * gaze_loss).item())

            avg_val = float(np.mean(val_losses))
            print(f'Epoch {epoch} — val loss: {avg_val:.4f}')

            torch.save({
                'shinra':    shinra.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch':     epoch,
            }, os.path.join(phase_dir, f'shinra_checkpoint_{epoch}.pth'))

            stopper.feed(avg_val)
            if stopper.stop:
                print(f'Early stopping at epoch {epoch} (phase {phase_idx})')
                break

        start_epoch = 0
