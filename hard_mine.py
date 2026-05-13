"""Hard mining on the captures dataset based on landmark (heatmap) MSE.

Runs inference over ./captures/ with the latest checkpoint, computes per-sample
landmark MSE, and copies all samples exceeding mean+1*std into --out.
"""

import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ShinraCNN
from input import RealDataset, REAL_DIR


def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]),
        reverse=True,
    )
    for phase_dir in phase_dirs:
        checkpoints = sorted(
            [f for f in os.listdir(phase_dir) if f.startswith('shinra_checkpoint_') and f.endswith('.pth')],
            key=lambda f: int(f.split('_')[-1].split('.')[0]),
        )
        if checkpoints:
            return os.path.join(phase_dir, checkpoints[-1])
    return None


def main():
    parser = argparse.ArgumentParser(description='Hard-mine captures by landmark MSE')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path (default: latest)')
    parser.add_argument('--out', default='hard_mines', help='Destination folder for hard examples')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if not ckpt_path:
        raise RuntimeError('No checkpoint found. Pass --checkpoint explicitly.')
    print(f'Checkpoint : {ckpt_path}')

    model = ShinraCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['shinra'])
    model.eval()

    # No augmentation — deterministic eval pass
    dataset = RealDataset(transforms=None)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        shuffle=False,
    )

    all_mse = []
    all_paths = []
    cursor = 0

    use_amp = device.type == 'cuda'
    print(f'Running inference on {len(dataset)} frames  (device={device}, amp={use_amp})...')

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            landmark_gt = lbls['landmarks'].to(device, non_blocking=True)  # (B, 17, 2)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    (landmark_pred, _), _, _, _ = model(imgs)
            else:
                (landmark_pred, _), _, _, _ = model(imgs)

            # Per-sample mean MSE across all 17 landmarks × 2 coords
            mse = F.mse_loss(landmark_pred, landmark_gt, reduction='none')  # (B, 17, 2)
            per_sample = mse.mean(dim=[1, 2]).cpu().numpy()                 # (B,)

            for j, m in enumerate(per_sample):
                si, fi = dataset._index[cursor + j]
                all_paths.append(dataset._sessions[si]['pngs'][fi])
                all_mse.append(float(m))

            cursor += len(per_sample)
            if cursor % 500 < args.batch_size or cursor >= len(dataset):
                print(f'  {cursor}/{len(dataset)}')

    all_mse = np.array(all_mse, dtype=np.float32)
    mean_mse = float(all_mse.mean())
    std_mse  = float(all_mse.std())
    threshold = mean_mse + std_mse

    print(f'\nLandmark MSE  mean={mean_mse:.6f}  std={std_mse:.6f}  threshold={threshold:.6f}')

    hard_idx = np.where(all_mse > threshold)[0]
    print(f'Hard examples : {len(hard_idx)} / {len(dataset)} ({100*len(hard_idx)/len(dataset):.1f}%)')

    os.makedirs(args.out, exist_ok=True)
    for i in hard_idx:
        src_png  = all_paths[i]
        src_json = os.path.splitext(src_png)[0] + '.json'

        rel      = os.path.relpath(src_png, REAL_DIR)
        dst_png  = os.path.join(args.out, rel)
        dst_json = os.path.splitext(dst_png)[0] + '.json'

        os.makedirs(os.path.dirname(dst_png), exist_ok=True)
        shutil.copy2(src_png, dst_png)
        if os.path.exists(src_json):
            shutil.copy2(src_json, dst_json)

    np.savez(
        os.path.join(args.out, 'mse_summary.npz'),
        paths=np.array(all_paths),
        mse=all_mse,
        threshold=np.float32(threshold),
    )
    print(f'Done. Hard examples + mse_summary.npz written to {args.out}/')


if __name__ == '__main__':
    main()
