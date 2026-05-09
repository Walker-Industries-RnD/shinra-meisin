#!/usr/bin/env python3
"""Inference visualization for Shinra-Meisin.

Dataset mode (default):
    python visualize.py [--n N] [--ckpt PATH]

Webcam mode:
    python visualize.py --webcam [--cam INDEX] [--ckpt PATH]
    Assumes 1280×720 input; center-crops to 640×480 then resizes to 256×192.
    Press 'q' to quit.
"""

import argparse
import glob
import os
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF

from input import (
    GIWDataset, INPUT_H, INPUT_W,
    MAX_PUPIL_DIAMETER, RAW_INPUT_H, RAW_INPUT_W,
    to_gray_tensor,
)
from model import ShinraCNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DISPLAY_W, DISPLAY_H = RAW_INPUT_W, RAW_INPUT_H  # 640×480

CAPTURE_DIR = 'captures'

EYE_STATES = {1: 'Error', 2: 'Blink', 3: 'Fixation', 4: 'Saccade', 5: 'SmoothPursuit'}

# Matplotlib colors (dataset mode)
COLOR_GT   = '#00e676'
COLOR_PRED = '#ff5252'

# OpenCV colors BGR (webcam mode)
CV_PRED  = (82, 82, 255)   # #ff5252
CV_GAZE  = (255, 200, 0)   # cyan-ish


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


# ── Shared inference ──────────────────────────────────────────────────────────

def run_inference(model, img_tensor):
    """(1, 192, 256) tensor → (lm_np, diam_px, gaze_xyz)."""
    with torch.no_grad():
        lm_pred, (diam_pred, _), (gaze_pred, _) = model(
            img_tensor.unsqueeze(0).to(DEVICE)
        )
    lm_np    = lm_pred[0].cpu().numpy()              # (17, 2) in [0,1]
    diam_px  = diam_pred[0, 0].item() * MAX_PUPIL_DIAMETER
    gaze_xyz = gaze_pred[0].cpu().numpy()             # (3,)
    return lm_np, diam_px, gaze_xyz


def scale_lm(lm):
    """(17, 2) [0,1] → (17, 2) float in 640×480 pixel space."""
    out = lm.copy()
    out[:, 0] *= DISPLAY_W
    out[:, 1] *= DISPLAY_H
    return out


# ── Matplotlib drawing (dataset mode) ────────────────────────────────────────

def denorm_image(img_tensor):
    """(1, H, W) [-1,1] tensor → (DISPLAY_H, DISPLAY_W) uint8."""
    img = (img_tensor.squeeze(0).cpu().numpy() * 0.5 + 0.5) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(img, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)


def mpl_draw_landmarks(ax, lm_px, color, label=None):
    upper, lower, pupil = lm_px[:8], lm_px[8:16], lm_px[16]
    ax.plot(upper[:, 0], upper[:, 1], '-o', color=color, ms=3, lw=1.4,
            alpha=0.9, label=label)
    ax.plot(lower[:, 0], lower[:, 1], '-o', color=color, ms=3, lw=1.4, alpha=0.9)
    ax.plot(pupil[0], pupil[1], '*', color=color, ms=9, alpha=0.9)


def mpl_draw_gaze(ax, gaze_xyz, origin_px, color, scale=90):
    ox, oy = origin_px
    dx, dy = float(gaze_xyz[0]) * scale, float(gaze_xyz[1]) * scale
    ax.annotate('', xy=(ox + dx, oy + dy), xytext=(ox, oy),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0))


# ── OpenCV drawing (webcam mode) ──────────────────────────────────────────────

def cv_draw_landmarks(frame, lm_px, color):
    pts = lm_px.astype(np.int32)
    cv2.polylines(frame, [pts[:8].reshape(-1, 1, 2)],  False, color, 1, cv2.LINE_AA)
    cv2.polylines(frame, [pts[8:16].reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
    for pt in pts[:16]:
        cv2.circle(frame, tuple(pt), 2, color, -1, cv2.LINE_AA)
    cv2.drawMarker(frame, tuple(pts[16]), color, cv2.MARKER_STAR, 10, 1, cv2.LINE_AA)


def cv_draw_gaze(frame, gaze_xyz, origin_px, color, scale=90):
    ox, oy = int(origin_px[0]), int(origin_px[1])
    dx = int(float(gaze_xyz[0]) * scale)
    dy = int(float(gaze_xyz[1]) * scale)
    cv2.arrowedLine(frame, (ox, oy), (ox + dx, oy + dy),
                    color, 2, cv2.LINE_AA, tipLength=0.25)


# ── Webcam preprocessing ──────────────────────────────────────────────────────

CAM_W, CAM_H = 1280, 720
_CROP_X = (CAM_W - DISPLAY_W) // 2   # 320
_CROP_Y = (CAM_H - DISPLAY_H) // 2   # 120


def preprocess_frame(bgr_frame):
    """1280×720 BGR → (1, 192, 256) float32 tensor, plus the 640×480 BGR crop."""
    crop = bgr_frame[_CROP_Y:_CROP_Y + DISPLAY_H, _CROP_X:_CROP_X + DISPLAY_W]
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t    = torch.from_numpy(rgb).permute(2, 0, 1)          # (3, 480, 640)
    t    = TF.resize(t, [INPUT_H, INPUT_W], antialias=True) # (3, 192, 256)
    t    = to_gray_tensor(t)                                 # (1, 192, 256)
    return t, crop


# ── Modes ─────────────────────────────────────────────────────────────────────

def dataset_mode(n_samples, ckpt_path):
    import random
    model = load_model(ckpt_path)
    ds    = GIWDataset()

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    ncols = min(3, len(indices))
    nrows = (len(indices) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    axes = np.array(axes).flatten()

    for ax_idx, ds_idx in enumerate(indices):
        img_tensor, gt = ds[ds_idx]
        lm_pred, diam_pred_px, gaze_pred = run_inference(model, img_tensor)

        lm_pred_px = scale_lm(lm_pred)
        lm_gt_px   = scale_lm(gt['landmarks'].numpy())
        diam_gt_px = gt['pupil_diameter'].item() * MAX_PUPIL_DIAMETER
        gaze_gt    = gt['gaze_vector'].numpy()

        img_display = denorm_image(img_tensor)

        ax = axes[ax_idx]
        ax.imshow(img_display, cmap='gray', origin='upper',
                  extent=[0, DISPLAY_W, DISPLAY_H, 0])

        mpl_draw_landmarks(ax, lm_gt_px,   COLOR_GT,   label='GT')
        mpl_draw_landmarks(ax, lm_pred_px, COLOR_PRED, label='Pred')
        mpl_draw_gaze(ax, gaze_gt,   lm_gt_px[16],   COLOR_GT)
        mpl_draw_gaze(ax, gaze_pred, lm_pred_px[16], COLOR_PRED)

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


def webcam_mode(cam_index, ckpt_path):
    model = load_model(ckpt_path)

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    existing = [
        int(os.path.splitext(f)[0]) for f in os.listdir(CAPTURE_DIR)
        if f.endswith('.jpg') and os.path.splitext(f)[0].isdigit()
    ]
    capture_idx = max(existing) + 1 if existing else 0

    print("Webcam live — hold '1' to capture frames, press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame.')
            break

        img_tensor, crop = preprocess_frame(frame)
        lm_pred, diam_px, gaze_xyz = run_inference(model, img_tensor)

        # Full frame as canvas; swap crop region to grayscale so it reads as
        # distinct from the surrounding color input.
        display = frame.copy()
        crop_gray = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                 cv2.COLOR_GRAY2BGR)
        display[_CROP_Y:_CROP_Y + DISPLAY_H, _CROP_X:_CROP_X + DISPLAY_W] = crop_gray

        # Shift landmarks from crop-space into full-frame-space before drawing.
        lm_px = scale_lm(lm_pred) + np.array([_CROP_X, _CROP_Y], dtype=np.float32)

        cv_draw_landmarks(display, lm_px, CV_PRED)
        cv_draw_gaze(display, gaze_xyz, lm_px[16], CV_GAZE)

        cv2.putText(display, f'diam={diam_px:.1f}px',
                    (_CROP_X + 8, _CROP_Y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CV_PRED, 1, cv2.LINE_AA)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            path = os.path.join(CAPTURE_DIR, f'{capture_idx:06d}.jpg')
            cv2.imwrite(path, gray_crop)
            capture_idx += 1
            cv2.putText(display, 'REC', (_CROP_X + 8, _CROP_Y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)

        cv2.imshow('Shinra-Meisin', display)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Shinra-Meisin inference visualizer')
    parser.add_argument('--webcam', action='store_true',
                        help='Live webcam mode (1280×720 → 640×480 center crop)')
    parser.add_argument('--cam',  type=int, default=0,
                        help='Camera device index (default: 0)')
    parser.add_argument('--n',    type=int, default=6,
                        help='Number of random dataset samples (default: 6)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to checkpoint .pth (default: latest found)')
    args = parser.parse_args()

    ckpt_path = args.ckpt or find_latest_checkpoint()
    if ckpt_path is None:
        print('No checkpoint found — train the model first.')
        sys.exit(1)
    print(f'Checkpoint: {ckpt_path}')

    if args.webcam:
        webcam_mode(args.cam, ckpt_path)
    else:
        dataset_mode(args.n, ckpt_path)


if __name__ == '__main__':
    main()
