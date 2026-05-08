from input import GIWDataset, giw_transforms
from model import ShinraCNN
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch
import time, os
from torch.optim import Adam
from early_stopping import EarlyStopping
from numpy import mean

BATCH_SIZE = 128
NUM_EPOCHS = 10 # per phase
PATIENCE = 5 # for early stopping, this is how many epochs of no val loss improvement to end phase

batch_size = 64
loss_weights = {
    'diameter_se': 1,
    'diameter_lvar': 1,
    'landmarks': 1,
    'gaze_se': 1,
    'gaze_lvar': 1
    # 'state': 1,
}

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shinra = ShinraCNN().to(device)

real_ds = GIWDataset(transforms=giw_transforms)
_n_train  = int(len(real_ds) * 0.7)
real_sets = random_split(real_ds, [_n_train, len(real_ds) - _n_train])

train_set, val_set = real_sets[0], real_sets[1]

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=6, pin_memory=True, drop_last=True)

torch.backends.cudnn.benchmark = True

ckpt_path, start_phase, _ = find_latest_checkpoint()
if ckpt_path:
    print(f'Resuming from {ckpt_path} (phase {start_phase})')
    ckpt = torch.load(ckpt_path, map_location=device)
    shinra.load_state_dict(ckpt['shinra'])
    start_epoch = ckpt['epoch'] + 1
else:
    print('Beginning from fresh start.')
    start_epoch = 0
    
scaler = torch.amp.GradScaler('cuda')
stopper = EarlyStopping(PATIENCE)

def process_batch(model, optim, imgs, lbls, epoch_losses=None, eval=False):
    if not eval:
        optim.zero_grad()

    before = time.monotonic_ns()
    landmark_pred, (diam_pred, diam_lvar), (gaze_pred, gaze_lvar) = model(imgs.to(device))
    pred_time = time.monotonic_ns() - before

    landmark_gt = lbls['landmarks'].to(device)
    diam_gt = lbls['pupil_diameter'].to(device)
    gaze_gt = lbls['gaze_vector'].to(device)

    diameter_loss, diam_sq_err, diam_mean_lvar = heteroscedastic_loss(diam_pred, diam_gt, diam_lvar)
    gaze_loss, gaze_sq_err, gaze_mean_lvar = heteroscedastic_loss(gaze_pred, gaze_gt, gaze_lvar)
    landmark_loss = F.smooth_l1_loss(landmark_pred, landmark_gt, beta=0.025)

    total_loss = (loss_weights['diameter_se'] * diameter_loss) + (loss_weights['gaze_se'] * gaze_loss) + (loss_weights['landmarks'] * landmark_loss)

    if epoch_losses:
        epoch_losses['diameter_se'].append(diam_sq_err)
        epoch_losses['diameter_lvar'].append(diam_mean_lvar)
        epoch_losses['gaze_se'].append(gaze_sq_err)
        epoch_losses['gaze_lvar'].append(gaze_mean_lvar)
        epoch_losses['landmarks'].append(landmark_loss.item())
    
    return total_loss, pred_time

for idx, groups in shinra.thaw():
    if idx < start_phase:
        continue
    print(f'SHINRA-MEISIN | PHASE {idx}')

    phase_dir = f'phase_{idx}'
    os.makedirs(phase_dir, exist_ok=True)
    
    optim = Adam(params=groups, lr=1e-4)

    stopper.reset()
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        epoch_losses = {head: [] for head in loss_weights.keys()}
        pred_time = []

        shinra.train()
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            with torch.amp.autocast('cuda'):
                total_loss, t = process_batch(shinra, optim, imgs, lbls, epoch_losses)
            pred_time.append(t)

            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()

            if batch_idx % 25 == 0:
                print(f'EPOCH {epoch}: batch {batch_idx}: '
                    f'diameter {mean(epoch_losses["diameter_se"]):.4f}|{mean(epoch_losses["diameter_lvar"]):.4f} '
                    f'heatmaps {mean(epoch_losses["landmarks"]):.4f} '
                    f'gaze {mean(epoch_losses["gaze_se"]):.4f}|{mean(epoch_losses["gaze_lvar"]):.4f} '
                    f'| inference time {mean(pred_time)*1.e-6:.3f}')
                    
        print(f'Epoch {epoch} finished. Validating...')

        val_losses = []
        shinra.eval()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                with torch.amp.autocast('cuda'):
                    total_loss, _ = process_batch(shinra, optim, imgs, lbls, epoch_losses=None, eval=True)
                val_losses.append(total_loss.item())

        print(f'VAL LOSS: {mean(val_losses):.4f}')

        torch.save({
            'shinra': shinra.state_dict(),
            'optimizer': optim.state_dict(),
            'epoch': epoch
        }, os.path.join(phase_dir, f'shinra_checkpoint_{epoch}.pth'))

        stopper.feed(mean(val_losses))
        if stopper.stop:
            print(f'Early stopping at epoch {epoch} (phase {idx})')
            break

    start_epoch = 0