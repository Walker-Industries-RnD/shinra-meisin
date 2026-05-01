import time

from model import ShinraCNN, HEATMAP_BORDER
from visualize import hard_argmax_2d
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from dataset import SyntheticDS, synth_transforms
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from numpy import mean
import torch, os
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

early_stopping_patience = 5
num_epochs = 80
batch_size = 64
loss_weights = {
    'diameter': .85,
    'heatmaps': 700,
    'contour': 2,
    'gaze': .85
}

def focal_loss(logits, targets, gamma=2, alpha=0.9):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * bce

    # zero out the outer HEATMAP_BORDER pixels so residual edge activations don't tamper with the loss
    m = HEATMAP_BORDER
    mask = torch.ones_like(loss)
    mask[..., :m, :] = 0
    mask[..., -m:, :] = 0
    mask[..., :, :m] = 0
    mask[..., :, -m:] = 0

    focal = (loss * mask).sum() / mask.sum()

    # half-max contour MSE
    pred_peak   = logits.amax(dim=(-2, -1), keepdim=True)   # (B, C, 1, 1)
    gt_peak     = targets.amax(dim=(-2, -1), keepdim=True)

    pred_region = F.relu(logits - pred_peak * 0.75)
    gt_region   = F.relu(targets  - gt_peak   * 0.75)

    active      = (pred_peak > 0.1)          # (B, C, 1, 1) bool, broadcasts over H×W
    pred_region = pred_region * active
    gt_region   = gt_region   * active

    contour_loss = F.mse_loss(pred_region, gt_region)

    return focal + loss_weights['contour'] * contour_loss

def heteroscedastic_loss(pred, truth, log_var, weight=None):
    precision = torch.exp(-log_var)
    sq_err = (pred - truth) ** 2
    if weight is not None:
        sq_err = sq_err * weight
    loss = (precision * sq_err + log_var).mean()
    return loss, sq_err.mean().item(), log_var.mean().item()

def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]),
        reverse=True
    )
    for phase_dir in phase_dirs:
        phase_idx = int(phase_dir.split('_')[1])
        checkpoints = sorted(
            [f for f in os.listdir(phase_dir) if f.startswith('shinra_checkpoint_') and f.endswith('.pth')],
            key=lambda f: int(f.split('_')[-1].split('.')[0])
        )
        if checkpoints:
            ckpt_path = os.path.join(phase_dir, checkpoints[-1])
            epoch = int(checkpoints[-1].split('_')[-1].split('.')[0])
            return ckpt_path, phase_idx, epoch
    return None, 0, 0

synth_ds = SyntheticDS(transforms=synth_transforms)
synth_sets = random_split(synth_ds, [int(len(synth_ds) * 0.8), int(len(synth_ds) * 0.2)])

train_set, val_set = synth_sets[0], synth_sets[1]

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=6, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=6, pin_memory=True, drop_last=True)

torch.backends.cudnn.benchmark = True

backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
shinra = ShinraCNN(backbone, out_channels=17).to(device)

# Advance thaw generator to start_phase, applying unfreezing along the way
ckpt_path, start_phase, _ = find_latest_checkpoint()
thaw_gen = shinra.thaw()
_, optim_groups = next(thaw_gen)
if start_phase > 1:
    for _ in range(start_phase - 1):
        _, optim_groups = next(thaw_gen)

# Load weights and optimizer state if resuming
optimizer = optim.Adam(optim_groups, lr=1e-3)
if ckpt_path:
    print(f'Resuming from {ckpt_path} (phase {start_phase})')
    ckpt = torch.load(ckpt_path, map_location=device)
    shinra.load_state_dict(ckpt['shinra'])
    #optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1
else:
    print('Beginning from fresh start.')
    start_epoch = 0

scaler = torch.amp.GradScaler('cuda')
stopper = EarlyStopping(early_stopping_patience)

for phase_idx, (phase, optim_groups) in enumerate(thaw_gen, start=start_phase):
    print(f'PHASE {phase_idx}')
    phase_dir = f'phase_{phase_idx}'
    os.makedirs(phase_dir, exist_ok=True)

    #if phase_idx != start_phase:
    optimizer = optim.Adam(optim_groups, lr=1e-3)

    stopper.reset()
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_losses = {head: [] for head in loss_weights.keys()}
        pred_time = []

        shinra.train()
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                time.sleep(0.2)
                before = time.monotonic_ns()
                (diam_val, diam_lv), hm_val, (gaze_val, gaze_lv) = shinra(imgs.to(device, dtype=torch.float32))
                if batch_idx > 0:
                    pred_time.append(time.monotonic_ns() - before)

                diam_gt  = lbls['pupil_diameter'].to(device, dtype=torch.float32).unsqueeze(-1)
                hm_gt    = lbls['eye_heatmaps'].to(device, dtype=torch.float32)
                gaze_gt  = lbls['gaze_vector'].to(device, dtype=torch.float32)

                diameter_loss, _, _ = heteroscedastic_loss(diam_val, diam_gt, diam_lv)
                heatmap_loss        = focal_loss(hm_val, hm_gt)
                gaze_loss,     _, _ = heteroscedastic_loss(gaze_val, gaze_gt, gaze_lv)

                total_loss = (loss_weights['diameter'] * diameter_loss
                            + loss_weights['heatmaps'] * heatmap_loss
                            + loss_weights['gaze']     * gaze_loss)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses['diameter'].append(diameter_loss.item() * loss_weights['diameter'])
            epoch_losses['heatmaps'].append(heatmap_loss.item() * loss_weights['heatmaps'])
            epoch_losses['gaze'].append(gaze_loss.item() * loss_weights['gaze'])

            if batch_idx % 25 == 0:
                print(f'EPOCH {epoch}: batch {batch_idx}: '
                    f'diameter {mean(epoch_losses["diameter"]):.4f} '
                    f'heatmaps {mean(epoch_losses["heatmaps"]):.4f} '
                    f'gaze {mean(epoch_losses["gaze"]):.4f} '
                    f'| inference time {mean(pred_time)*1.e-6:.3f}')

        print(f'Epoch {epoch} finished. Validating...')

        val_losses = []
        shinra.eval()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                time.sleep(0.2)
                (diam_val, diam_lv), hm_val, (gaze_val, gaze_lv) = shinra(imgs.to(device, dtype=torch.float32))

                diam_gt  = lbls['pupil_diameter'].to(device, dtype=torch.float32).unsqueeze(-1)
                hm_gt    = lbls['eye_heatmaps'].to(device, dtype=torch.float32)
                gaze_gt  = lbls['gaze_vector'].to(device, dtype=torch.float32)

                diameter_loss, _, _ = heteroscedastic_loss(diam_val, diam_gt, diam_lv)
                heatmap_loss        = focal_loss(hm_val, hm_gt)
                gaze_loss,     _, _ = heteroscedastic_loss(gaze_val, gaze_gt, gaze_lv)

                val_loss = (loss_weights['diameter'] * diameter_loss
                          + loss_weights['heatmaps'] * heatmap_loss
                          + loss_weights['gaze']     * gaze_loss)
                val_losses.append(val_loss.item())

        avg_val_loss = mean(val_losses)
        print(f'VAL LOSS: {avg_val_loss:.4f}')

        torch.save({
            'shinra': shinra.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(phase_dir, f'shinra_checkpoint_{epoch}.pth'))

        stopper.feed(avg_val_loss)
        if stopper.stop:
            print(f'Early stopping at epoch {epoch} (phase {phase_idx})')
            break

    start_epoch = 0
