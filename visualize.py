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
    GIWDataset, SyntheticDataset, WebcamDataset,
    INPUT_H, INPUT_W, MAX_PUPIL_DIAMETER, RAW_INPUT_H, RAW_INPUT_W,
    cam_transforms, to_gray_tensor,
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
    """(1, 192, 256) tensor → (lm_np, diam_px, gaze_xyz, heatmap).

    heatmap: (H, W) float32 — max activation across all 17 softmax landmark maps.
    """
    with torch.no_grad():
        lm_pred, (diam_pred, _), (gaze_pred, _), _ = model(
            img_tensor.unsqueeze(0).to(DEVICE)
        )
    lm_np    = lm_pred[0].cpu().numpy()              # (17, 2) in [0,1]
    diam_px  = diam_pred[0, 0].item() * MAX_PUPIL_DIAMETER
    gaze_xyz = gaze_pred[0].cpu().numpy()             # (3,)
    heatmap  = model.landmark_head.last_soft_map[0].max(0).values.cpu().numpy()  # (H, W)
    return lm_np, diam_px, gaze_xyz, heatmap


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

    n_per  = max(n_samples // 3, 1)
    syn_ds = SyntheticDataset()
    giw_ds = GIWDataset()
    cam_ds = WebcamDataset(transforms=cam_transforms)

    def sample(ds, k):
        return random.sample(range(len(ds)), min(k, len(ds)))

    groups = [
        ('Synthetic', sample(syn_ds, n_per), syn_ds, True),
        ('GIW',       sample(giw_ds, n_per), giw_ds, True),
        ('Webcam',    sample(cam_ds, n_per), cam_ds, False),
    ]

    # Pre-collect inference results so toggling doesn't re-run the model.
    collected = []
    for domain, idxs, ds, has_gt in groups:
        row_samples = []
        for idx in idxs:
            if has_gt:
                img_tensor, gt = ds[idx]
            else:
                img_tensor = ds[idx]
                gt = None
            lm_pred, diam_pred_px, gaze_pred, heatmap = run_inference(model, img_tensor)
            s = {
                'img_display': denorm_image(img_tensor),
                'lm_pred_px':  scale_lm(lm_pred),
                'gaze_pred':   gaze_pred,
                'diam_pred_px': diam_pred_px,
                'heatmap':     heatmap,
                'domain':      domain,
                'has_gt':      has_gt,
            }
            if gt is not None:
                eye_state = int(gt['eye_state'].item())
                s.update({
                    'lm_gt_px':   scale_lm(gt['landmarks'].numpy()),
                    'diam_gt_px': gt['pupil_diameter'].item() * MAX_PUPIL_DIAMETER,
                    'gaze_gt':    gt['gaze_vector'].numpy(),
                    'state':  'Synthetic' if eye_state == 0 else EYE_STATES.get(eye_state, '?'),
                    'pvalid': 'pup✓' if gt['pupil_valid'].item() else 'pup✗',
                    'lvalid': 'lid✓' if gt['lid_valid'].item()   else 'lid✗',
                })
            row_samples.append(s)
        collected.append((domain, row_samples))

    ncols = n_per
    fig, axes = plt.subplots(3, ncols, figsize=(ncols * 5, 3 * 4.2), squeeze=False)
    show_hm = [False]

    def draw(hm_mode):
        for row_idx, (domain, samples) in enumerate(collected):
            for col, s in enumerate(samples):
                ax = axes[row_idx][col]
                ax.cla()
                if col >= ncols:
                    ax.axis('off')
                    continue

                if hm_mode:
                    hm = cv2.resize(s['heatmap'], (DISPLAY_W, DISPLAY_H),
                                    interpolation=cv2.INTER_NEAREST)
                    ax.imshow(hm, cmap='hot', interpolation='nearest', origin='upper',
                              extent=[0, DISPLAY_W, DISPLAY_H, 0])
                else:
                    ax.imshow(s['img_display'], cmap='gray', origin='upper',
                              extent=[0, DISPLAY_W, DISPLAY_H, 0])

                ax.set_xlim(0, DISPLAY_W)
                ax.set_ylim(DISPLAY_H, 0)
                ax.axis('off')

                if s['has_gt'] and not hm_mode:
                    mpl_draw_landmarks(ax, s['lm_gt_px'],  COLOR_GT,   label='GT')
                    mpl_draw_gaze(ax, s['gaze_gt'], s['lm_gt_px'][16], COLOR_GT)

                mpl_draw_landmarks(ax, s['lm_pred_px'], COLOR_PRED, label='Pred')
                mpl_draw_gaze(ax, s['gaze_pred'], s['lm_pred_px'][16], COLOR_PRED)

                if s['has_gt']:
                    ax.set_title(
                        f'{s["state"]}  {s["pvalid"]} {s["lvalid"]}\n'
                        f'diam  GT={s["diam_gt_px"]:.1f}  pred={s["diam_pred_px"]:.1f}  (px)',
                        fontsize=8,
                    )
                else:
                    ax.set_title(f'diam pred={s["diam_pred_px"]:.1f} px', fontsize=8)

                if col == 0:
                    ax.text(-0.08, 0.5, domain, transform=ax.transAxes,
                            fontsize=11, fontweight='bold', va='center', ha='right',
                            rotation=90)

        handles = [mpatches.Patch(color=COLOR_PRED, label='Pred')]
        if not hm_mode:
            handles.insert(0, mpatches.Patch(color=COLOR_GT, label='GT'))
        axes[0][0].legend(handles=handles, loc='lower right', fontsize=8)

        mode_label = 'Heatmap' if hm_mode else 'Normal'
        plt.suptitle(
            f'Shinra-Meisin — multi-domain inference  [{mode_label} | ←/→ to toggle]',
            fontsize=12,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ('left', 'right'):
            show_hm[0] = not show_hm[0]
            draw(show_hm[0])

    # Remove matplotlib's default arrow-key navigation bindings so our handler
    # receives them instead of the toolbar consuming them first.
    plt.rcParams['keymap.back']    = [k for k in plt.rcParams['keymap.back']    if k != 'left']
    plt.rcParams['keymap.forward'] = [k for k in plt.rcParams['keymap.forward'] if k != 'right']

    fig.canvas.mpl_connect('key_press_event', on_key)
    draw(False)
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

    show_heatmap = False
    print("Webcam live — ←/→ to toggle heatmap, hold '1' to capture, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame.')
            break

        img_tensor, crop = preprocess_frame(frame)
        lm_pred, diam_px, gaze_xyz, heatmap = run_inference(model, img_tensor)

        display = frame.copy()

        if show_heatmap:
            hm_u8 = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
            hm_bgr = cv2.applyColorMap(
                cv2.resize(hm_u8, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST),
                cv2.COLORMAP_HOT,
            )
            display[_CROP_Y:_CROP_Y + DISPLAY_H, _CROP_X:_CROP_X + DISPLAY_W] = hm_bgr
        else:
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

        # waitKeyEx returns full extended keycodes for special keys (arrows etc.)
        key  = cv2.waitKeyEx(1)
        key8 = key & 0xFF

        if key in (65361, 65363):  # left / right arrow (Linux X11 / GTK)
            show_heatmap = not show_heatmap

        if key8 == ord('1'):
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            path = os.path.join(CAPTURE_DIR, f'{capture_idx:06d}.jpg')
            cv2.imwrite(path, gray_crop)
            capture_idx += 1
            cv2.putText(display, 'REC', (_CROP_X + 8, _CROP_Y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)

        cv2.imshow('Shinra-Meisin', display)
        if key8 == ord('q'):
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
    parser.add_argument('--n',    type=int, default=9,
                        help='Number of random dataset samples (default: 9, 3 per domain)')
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
