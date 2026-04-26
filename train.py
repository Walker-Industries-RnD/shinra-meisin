from model import ShinraCNN
from dataset import SyntheticDS, synth_transforms
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from numpy import mean
import torch, os
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

early_stopping_patience = 5
num_epochs = 90
batch_size = 64
loss_weights = {
    'pupil': 1,
    'eyelid': 1.5,
    'gaze': 0.75
}

def heteroscedastic_loss(pred, truth, log_var):
    precision = torch.exp(-log_var)
    mse = F.mse_loss(pred, truth, reduction='none')
    loss = precision * mse + log_var
    return loss.mean(), mse.mean().item(), log_var.mean().item()

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

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16, pin_memory=True, drop_last=True)

torch.backends.cudnn.benchmark = True

shinra = ShinraCNN().to(device)

# Advance thaw generator to start_phase, applying unfreezing along the way
ckpt_path, start_phase, _ = find_latest_checkpoint()
thaw_gen = shinra.thaw()
_, optim_groups = next(thaw_gen)
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
    for epoch in range(start_epoch, num_epochs):
        epoch_losses = {head: [] for head in loss_weights.keys()}

        shinra.train()
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pupil_pred, eyelid_pred, gaze_pred = shinra(imgs.to(device, dtype=torch.float32))

                pupil_loss, pupil_mse, pupil_lv = heteroscedastic_loss(pupil_pred[0], lbls['pupil'].to(device, dtype=torch.float32), pupil_pred[1])
                eyelid_loss, eyelid_mse, eyelid_lv = heteroscedastic_loss(eyelid_pred[0], lbls['eyelid_shape'].to(device, dtype=torch.float32), eyelid_pred[1])
                gaze_loss, gaze_mse, gaze_lv = heteroscedastic_loss(gaze_pred[0], lbls['gaze_vector'].to(device, dtype=torch.float32), gaze_pred[1])

                total_loss = loss_weights['pupil'] * pupil_loss + loss_weights['eyelid'] * eyelid_loss + loss_weights['gaze'] * gaze_loss
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses['pupil'].append(pupil_loss.item())
            epoch_losses['eyelid'].append(eyelid_loss.item())
            epoch_losses['gaze'].append(gaze_loss.item())

            if batch_idx % 25 == 0:
                print(f'EPOCH {epoch}: batch {batch_idx}: '
                    f'pupil {mean(epoch_losses["pupil"]):.4f} '
                    f'eyelid {mean(epoch_losses["eyelid"]):.4f} '
                    f'gaze {mean(epoch_losses["gaze"]):.4f} '
                    f'| gaze mse {gaze_mse:.4f} lv {gaze_lv:.4f} '
                    f'[{100 * (batch_idx / (len(train_set) / batch_size)):.2f}%]')

        print(f'Epoch {epoch} finished. Validating...')

        val_losses = []
        shinra.eval()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                pupil_pred, eyelid_pred, gaze_pred = shinra(imgs.to(device, dtype=torch.float32))

                pupil_loss, _, _ = heteroscedastic_loss(pupil_pred[0], lbls['pupil'].to(device, dtype=torch.float32), pupil_pred[1])
                eyelid_loss, _, _ = heteroscedastic_loss(eyelid_pred[0], lbls['eyelid_shape'].to(device, dtype=torch.float32), eyelid_pred[1])
                gaze_loss, _, _ = heteroscedastic_loss(gaze_pred[0], lbls['gaze_vector'].to(device, dtype=torch.float32), gaze_pred[1])

                val_loss = loss_weights['pupil'] * pupil_loss + loss_weights['eyelid'] * eyelid_loss + loss_weights['gaze'] * gaze_loss
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
