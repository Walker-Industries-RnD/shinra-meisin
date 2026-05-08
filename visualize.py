#!/usr/bin/env python3
"""Inference visualization for Shinra-Meisin.

Usage:
    python visualize.py [n_samples] [checkpoint_path]

Picks n_samples random GIWDataset entries, runs inference, and shows GT vs
predicted landmarks (eyelid curves + pupil), diameter, and gaze arrows,
all rendered on a 640×480 display frame.
"""

import glob
import os
import random
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from input import GIWDataset, MAX_PUPIL_DIAMETER, RAW_INPUT_H, RAW_INPUT_W
from model import ShinraCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DISPLAY_W, DISPLAY_H = RAW_INPUT_W, RAW_INPUT_H  # 640×480

EYE_STATES = {1: 'Error', 2: 'Blink', 3: 'Fixation', 4: 'Saccade', 5: 'SmoothPursuit'}

COLOR_GT   = '#00e676'
COLOR_PRED = '#ff5252'


# ── Checkpoint ────────────────────────────────────────────────────────────────

def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]),
        reverse=True,
    )
    for phase_dir in phase_dirs:
        checkpoints = sorted(
            glob.glob(os.path.join(phase_dir, 'shinra_checkpoint_*.pth')),
            key=lambda f: int(f.split('_')[-1].split('.')[0]),
        )
        if checkpoints:
            return checkpoints[-1]
    return None


def load_model(ckpt_path):
    model = ShinraCNN().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['shinra'])
    model.eval()
    return model


# ── Drawing helpers ───────────────────────────────────────────────────────────

def denorm_image(img_tensor):
    """(1, H, W) [-1,1] tensor → (DISPLAY_H, DISPLAY_W) uint8 array."""
    img = (img_tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(img, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)


def scale_lm(lm):
    """(17, 2) float [0,1] → (17, 2) float in 640×480 pixel space."""
    pts = lm if isinstance(lm, np.ndarray) else lm.cpu().numpy()
    out = pts.copy()
    out[:, 0] *= DISPLAY_W
    out[:, 1] *= DISPLAY_H
    return out


def draw_landmarks(ax, lm_px, color, label=None, alpha=0.9):
    """lm_px: (17, 2) in pixel coords. Indices 0-7 upper, 8-15 lower, 16 pupil."""
    upper = lm_px[:8]
    lower = lm_px[8:16]
    pupil = lm_px[16]
    ax.plot(upper[:, 0], upper[:, 1], '-o', color=color, ms=3, lw=1.4,
            alpha=alpha, label=label)
    ax.plot(lower[:, 0], lower[:, 1], '-o', color=color, ms=3, lw=1.4, alpha=alpha)
    ax.plot(pupil[0], pupil[1], '*', color=color, ms=9, alpha=alpha)


def draw_gaze(ax, gaze_xyz, origin_px, color, scale=90):
    """Arrow from origin in gaze X/Y direction (Z ignored for 2-D projection)."""
    ox, oy = origin_px
    dx, dy = float(gaze_xyz[0]) * scale, float(gaze_xyz[1]) * scale
    ax.annotate(
        '', xy=(ox + dx, oy + dy), xytext=(ox, oy),
        arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(n_samples=6, ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        print('No checkpoint found — train the model first.')
        sys.exit(1)
    print(f'Checkpoint: {ckpt_path}')

    model = load_model(ckpt_path)
    ds = GIWDataset()  # no augmentation — deterministic

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    ncols = min(3, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    axes = np.array(axes).flatten()

    for ax_idx, ds_idx in enumerate(indices):
        img_tensor, gt = ds[ds_idx]

        with torch.no_grad():
            lm_pred, (diam_pred, _), (gaze_pred, _) = model(
                img_tensor.unsqueeze(0).to(DEVICE)
            )

        # Unpack predictions
        lm_pred_px   = scale_lm(lm_pred[0].cpu().numpy())
        lm_gt_px     = scale_lm(gt['landmarks'].numpy())
        diam_pred_px = diam_pred[0, 0].item() * MAX_PUPIL_DIAMETER
        diam_gt_px   = gt['pupil_diameter'].item() * MAX_PUPIL_DIAMETER
        gaze_pred_v  = gaze_pred[0].cpu().numpy()
        gaze_gt_v    = gt['gaze_vector'].numpy()

        img_display = denorm_image(img_tensor)

        ax = axes[ax_idx]
        ax.imshow(img_display, cmap='gray', origin='upper',
                  extent=[0, DISPLAY_W, DISPLAY_H, 0])

        draw_landmarks(ax, lm_gt_px,   COLOR_GT,   label='GT')
        draw_landmarks(ax, lm_pred_px, COLOR_PRED, label='Pred')
        draw_gaze(ax, gaze_gt_v,   lm_gt_px[16],   COLOR_GT)
        draw_gaze(ax, gaze_pred_v, lm_pred_px[16], COLOR_PRED)

        state  = EYE_STATES.get(int(gt['eye_state'].item()), '?')
        pvalid = 'pup✓' if gt['pupil_valid'].item() else 'pup✗'
        lvalid = 'lid✓' if gt['lid_valid'].item()   else 'lid✗'
        ax.set_title(
            f'{state}  {pvalid} {lvalid}\n'
            f'diam  GT={diam_gt_px:.1f}  pred={diam_pred_px:.1f}  (px)',
            fontsize=8,
        )
        ax.set_xlim(0, DISPLAY_W)
        ax.set_ylim(DISPLAY_H, 0)
        ax.axis('off')

        if ax_idx == 0:
            ax.legend(
                handles=[
                    mpatches.Patch(color=COLOR_GT,   label='GT'),
                    mpatches.Patch(color=COLOR_PRED, label='Pred'),
                ],
                loc='lower right', fontsize=8,
            )

    for ax in axes[len(indices):]:
        ax.axis('off')

    plt.suptitle('Shinra-Meisin — inference visualization', fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    n    = int(sys.argv[1])    if len(sys.argv) > 1 else 6
    ckpt = sys.argv[2]         if len(sys.argv) > 2 else None
    visualize(n_samples=n, ckpt_path=ckpt)
