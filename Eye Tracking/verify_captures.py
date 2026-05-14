"""Slideshow viewer for verifying capture_dataset.py ground truths.

Per frame, overlays:
  • 16 eyelid landmarks connected as upper / lower polylines, pupil as a dot
  • pupil diameter as a horizontal segment centred on the pupil
  • gaze vector as a red arrow from the pupil along (gx, gy); a filled/hollow
    tip disc indicates gz sign and magnitude (filled = +z toward camera,
    hollow = −z away from camera)

Loads a single capture run (the directory containing eye0/ and eye1/) and
walks both eyes in sync.

By default the viewer normalizes randomly-cropped frames so the pupil sits at
a consistent position: it finds the maximally-open frame for each eye (largest
vertical gap between upper and lower lid landmarks) and translates every other
frame to match that pupil position.  Pass --no-normalize to disable.

Controls: SPACE pause/resume, ← / → step, , / . step (no arrow keys),
          + / - speed, q or ESC quit.
"""

import argparse
import glob
import json
import math
from pathlib import Path

import cv2
import numpy as np


WHITE  = (255, 255, 255)
GREEN  = (80, 220, 80)
ORANGE = (40, 160, 255)
YELLOW = (0, 255, 255)
CYAN   = (255, 200, 0)
RED    = (40, 40, 255)


def overlay(img_bgr, gt):
    h, w = img_bgr.shape[:2]
    lm = np.array(gt['landmarks'], dtype=np.float32)
    lm[:, 0] *= w
    lm[:, 1] *= h
    upper, lower, pupil = lm[:8], lm[8:16], lm[16]

    # Lid polylines.
    cv2.polylines(img_bgr, [upper.astype(np.int32)], False, GREEN, 1, cv2.LINE_AA)
    cv2.polylines(img_bgr, [lower.astype(np.int32)], False, GREEN, 1, cv2.LINE_AA)
    for p in np.vstack([upper, lower]):
        cv2.circle(img_bgr, tuple(p.astype(int)), 2, ORANGE, -1, cv2.LINE_AA)
    pcx, pcy = float(pupil[0]), float(pupil[1])
    cv2.circle(img_bgr, (int(pcx), int(pcy)), 3, YELLOW, -1, cv2.LINE_AA)

    # Diameter segment (raw pixels, since MAX_PUPIL_DIAMETER=1 in capture).
    d = float(gt['pupil_diameter'])
    cv2.line(img_bgr,
             (int(pcx - d / 2), int(pcy)),
             (int(pcx + d / 2), int(pcy)),
             CYAN, 2, cv2.LINE_AA)

    # Gaze arrow from pupil along (gx, gy); tip disc encodes gz.
    outer, inner = upper[0], upper[-1]
    eye_w = math.hypot(outer[0] - inner[0], outer[1] - inner[1])
    L = eye_w * 1.2
    gx, gy, gz = (float(v) for v in gt['gaze_vector'])
    tip = (int(pcx + L * gx), int(pcy + L * gy))
    cv2.arrowedLine(img_bgr, (int(pcx), int(pcy)), tip,
                    RED, 2, cv2.LINE_AA, tipLength=0.2)
    r_z = max(2, int(abs(gz) * eye_w * 0.18))
    cv2.circle(img_bgr, tip, r_z, RED, -1 if gz >= 0 else 2, cv2.LINE_AA)


def load_gt(png_path):
    with open(Path(png_path).with_suffix('.json')) as f:
        return json.load(f)


def find_reference_pupil(png_paths):
    """Return (px, py) in pixels from the frame whose eye is most open.

    Eye openness is the mean vertical gap between the upper (lm[0:8]) and
    lower (lm[8:16]) lid rows.  Only JSONs are read after the first image
    is decoded to learn the frame dimensions.
    """
    h = w = None
    for p in png_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
            break
    if h is None:
        return None

    best_open = -1.0
    best_pupil = None
    for png_path in png_paths:
        json_path = Path(png_path).with_suffix('.json')
        if not json_path.exists():
            continue
        try:
            with open(json_path) as f:
                gt = json.load(f)
        except Exception:
            continue
        lm = np.array(gt['landmarks'], dtype=np.float32)
        openness = float(lm[8:16, 1].mean() - lm[:8, 1].mean()) * h
        if openness > best_open:
            best_open = openness
            best_pupil = (float(lm[16, 0] * w), float(lm[16, 1] * h))
    return best_pupil


def render_panel(png_path, label, target_pupil=None):
    gray = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    gt = load_gt(png_path)
    h, w = gray.shape[:2]

    if target_pupil is not None:
        lm = np.array(gt['landmarks'], dtype=np.float32)
        dx = int(round(target_pupil[0] - lm[16, 0] * w))
        dy = int(round(target_pupil[1] - lm[16, 1] * h))
        if dx != 0 or dy != 0:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            margin = 12
            skin_val = int(np.median([
                gray[:margin, :margin],
                gray[:margin, -margin:],
                gray[-margin:, :margin],
                gray[-margin:, -margin:],
            ]))
            gray = cv2.warpAffine(gray, M, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=skin_val)
            lm[:, 0] = (lm[:, 0] * w + dx) / w
            lm[:, 1] = (lm[:, 1] * h + dy) / h
            gt = dict(gt)
            gt['landmarks'] = lm.tolist()

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay(img, gt)
    es = {3: 'fixation', 4: 'saccade', 5: 'pursuit'}.get(int(gt['eye_state']), '?')
    g = gt['gaze_vector']
    txt = (f'{label}  {Path(png_path).stem}  {es}  '
           f'd={gt["pupil_diameter"]:.1f}px  '
           f'g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f})')
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, WHITE, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir',
                    help='captures/{run_ts} directory with eye0/ and eye1/')
    ap.add_argument('--fps',          type=float, default=4.0)
    ap.add_argument('--scale',        type=float, default=1.5)
    ap.add_argument('--start',        type=int,   default=0)
    ap.add_argument('--no-normalize', action='store_true',
                    help='disable pupil-position normalization')
    args = ap.parse_args()

    run = Path(args.run_dir).expanduser()
    sides = [s for s in ('eye0', 'eye1') if (run / s).is_dir()]
    if not sides:
        raise SystemExit(f'no eye0/ or eye1/ under {run}')
    listings = {s: sorted(glob.glob(str(run / s / '*.png'))) for s in sides}
    n = min(len(v) for v in listings.values())
    if n == 0:
        raise SystemExit(f'no PNGs found under {run}')

    ref_pupils = {}
    if not args.no_normalize:
        print('Scanning for max-openness reference frames...', end=' ', flush=True)
        for s in sides:
            ref_pupils[s] = find_reference_pupil(listings[s])
        print('done')

    i = max(0, min(args.start, n - 1))
    paused = False
    fps = args.fps

    win = 'verify_captures'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        panels = [render_panel(listings[s][i], s, ref_pupils.get(s))
                  for s in sides]
        combo = np.concatenate(panels, axis=1)
        if args.scale != 1.0:
            combo = cv2.resize(combo, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_LINEAR)
        norm_tag = '' if args.no_normalize else '  [normalized]'
        footer = (f'{i + 1}/{n}   '
                  f'{"PAUSED" if paused else f"{fps:.1f} fps"}'
                  f'{norm_tag}   '
                  f'[SPACE pause] [, .  step] [+ -  speed] [q quit]')
        cv2.putText(combo, footer, (10, combo.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.imshow(win, combo)

        delay = 0 if paused else max(1, int(1000 / fps))
        key = cv2.waitKey(delay) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord(','), 81):
            i = (i - 1) % n
            paused = True
        elif key in (ord('.'), 83):
            i = (i + 1) % n
            paused = True
        elif key in (ord('+'), ord('=')):
            fps = min(60.0, fps * 1.5)
        elif key in (ord('-'), ord('_')):
            fps = max(0.25, fps / 1.5)
        elif not paused:
            i = (i + 1) % n

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
