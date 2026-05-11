from input import SharedDataset, WebcamDataset, giw_transforms, cam_transforms, GIWDataset, session_split, SyntheticDataset
from model import ShinraCNN
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch
import time, os, math
from torch.optim import Adam
from early_stopping import EarlyStopping
from numpy import mean

def _infinite_loader(loader):
    """Yields batches from loader indefinitely without caching them in RAM."""
    while True:
        yield from loader

BATCH_SIZE = 64
NUM_EPOCHS = 10 # per phase
PATIENCE = 5 # for early stopping, this is how many epochs of no val loss improvement to end phase

loss_weights = {
    'diameter_se': 1,
    'diameter_lvar': 1,
    'landmarks': 1,
    'gaze_se': 1,
    'gaze_lvar': 1,
    'clf_orig': 1,
    'clf_cam': 1
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
            first_epoch = int(checkpoints[0].split('_')[-1].split('.')[0])
            latest_epoch = int(checkpoints[-1].split('_')[-1].split('.')[0])
            return ckpt_path, phase_idx, first_epoch, latest_epoch
    return None, 0, 0, 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

shinra = ShinraCNN().to(device)

torch.backends.cudnn.benchmark = True

ckpt_path, start_phase, start_epoch, latest_epoch = find_latest_checkpoint()
if ckpt_path:
    print(f'Resuming from {ckpt_path} (phase {start_phase})')
    ckpt = torch.load(ckpt_path, map_location=device)
    shinra.load_state_dict(ckpt['shinra'])
    resume_epoch = latest_epoch + 1
    start_epoch = resume_epoch
else:
    print('Beginning from fresh start.')
    start_epoch = 0
    resume_epoch = 0


real_ds  = GIWDataset(transforms=giw_transforms)
synth_ds = SyntheticDataset(transforms=giw_transforms)
cam_ds = WebcamDataset(transforms=cam_transforms)
giw_train, giw_val = session_split(real_ds)

#train_len = int(len(synth_ds) * .7)
#synth_train, synth_val = random_split(synth_ds, [train_len, len(synth_ds) - train_len])

train_ds = SharedDataset(giw_subset=giw_train, synthetic=synth_ds, phase=start_phase, epoch_size=len(synth_ds))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True, shuffle=False)
val_loader   = DataLoader(giw_val,  batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)

cam_loader   = DataLoader(cam_ds,   batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
cam_cycle = _infinite_loader(cam_loader)

scaler = torch.amp.GradScaler('cuda')
stopper = EarlyStopping(PATIENCE)

def process_batch(phase, model, optim, imgs, lbls, epoch_losses=None, eval=False):
    if not eval:
        optim.zero_grad()

    train_imgs = imgs[0]
    cam_imgs = imgs[1]

    before = time.monotonic_ns()
    
    cam_arg = cam_imgs.to(device) if (phase > 1 and cam_imgs is not None) else None
    landmark_pred, (diam_pred, diam_lvar), (gaze_pred, gaze_lvar), clf_preds = model(train_imgs.to(device), cam_imgs=cam_arg)

    pred_time = time.monotonic_ns() - before

    landmark_gt = lbls['landmarks'].to(device)
    diam_gt = lbls['pupil_diameter'].to(device)
    gaze_gt = lbls['gaze_vector'].to(device)

    diameter_loss, diam_sq_err, diam_mean_lvar = heteroscedastic_loss(diam_pred, diam_gt, diam_lvar)
    gaze_loss, gaze_sq_err, gaze_mean_lvar = heteroscedastic_loss(gaze_pred, gaze_gt, gaze_lvar)
    landmark_loss = F.smooth_l1_loss(landmark_pred, landmark_gt, beta=0.025)

    total_loss = (loss_weights['diameter_se'] * diameter_loss) + (loss_weights['gaze_se'] * gaze_loss) + (loss_weights['landmarks'] * landmark_loss)

    if phase > 1 and clf_preds:
        b = clf_preds[0].size(0)
        domain_loss_orig = F.binary_cross_entropy_with_logits(
            clf_preds[0].squeeze(1),
            torch.zeros(b, device=device)
        )
        domain_loss_cam = F.binary_cross_entropy_with_logits(
            clf_preds[1].squeeze(1),
            torch.ones(b, device=device)
        )
        total_loss += loss_weights['clf_orig'] * domain_loss_orig + loss_weights['clf_cam'] * domain_loss_cam
        if epoch_losses:
            epoch_losses['clf_orig'].append(domain_loss_orig.item())
            epoch_losses['clf_cam'].append(domain_loss_cam.item())



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
    dann_active = idx > 1
    for epoch in range(resume_epoch, start_epoch + NUM_EPOCHS):
        epoch_losses = {head: [] for head in loss_weights.keys()}
        pred_time = []

        epoch_progress = (epoch - start_epoch) / max(NUM_EPOCHS - 1, 1)

        if dann_active:
            lam = max(0.1, 2.0 / (1.0 + math.exp(-10.0 * epoch_progress)) - 1.0)
            shinra.dann_clf.grl.lambda_.fill_(lam)

        train_ds.reshuffle(epoch_progress=epoch_progress)
        shinra.train()

        batch_iter = ( # only phase 2+ pulls from the webcam dataset, so change up the generator accordingly
            ((imgs, lbls), next(cam_cycle)) for imgs, lbls in train_loader
        ) if dann_active else (
            ((imgs, lbls), None) for imgs, lbls in train_loader
        )
        for batch_idx, ((imgs, lbls), cam_imgs) in enumerate(batch_iter):
            with torch.amp.autocast('cuda'):
                total_loss, t = process_batch(idx, shinra, optim, (imgs, cam_imgs), lbls, epoch_losses)
            pred_time.append(t)

            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()

            if batch_idx % 25 == 0:
                print(f'EPOCH {epoch}: batch {batch_idx}: '
                    f'diameter {mean(epoch_losses["diameter_se"]):.4f}|{mean(epoch_losses["diameter_lvar"]):.4f} '
                    f'heatmaps {mean(epoch_losses["landmarks"]):.4f} '
                    f'gaze {mean(epoch_losses["gaze_se"]):.4f}|{mean(epoch_losses["gaze_lvar"]):.4f} '
                    f'| DANN orig: {mean(epoch_losses["clf_orig"]):.4f} | DANN cam: {mean(epoch_losses["clf_cam"]):.4f} '
                    f'| inference time {mean(pred_time)*1.e-6:.3f}')
                    
        print(f'Epoch {epoch} finished. Validating...')

        val_losses = []
        shinra.eval()
        with torch.no_grad():
            for imgs, lbls in val_loader:
                with torch.amp.autocast('cuda'):
                    total_loss, _ = process_batch(idx, shinra, optim, (imgs, None), lbls, epoch_losses=None, eval=True)
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