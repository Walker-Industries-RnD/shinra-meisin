"""Random-crop existing 640x480 captures to 320x240 in-place.

Crop centre is offset from the source centre by a random vector of L2 magnitude
≤ 50 px (uniform within the disc).  Updates landmarks in each .json; everything
else (gaze_vector, pupil_diameter in raw px, eye_state, validity flags) is
intrinsic and stays untouched.
"""

import argparse
import glob
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np


SRC_W, SRC_H = 640, 480
OUT_W, OUT_H = 320, 240
MAX_OFFSET   = 50


def random_offset():
    r = MAX_OFFSET * math.sqrt(random.random())
    th = random.uniform(0.0, 2 * math.pi)
    return int(round(r * math.cos(th))), int(round(r * math.sin(th)))


def process(png_path, json_path):
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[0] != SRC_H or img.shape[1] != SRC_W:
        return False
    with open(json_path) as f:
        gt = json.load(f)

    dx, dy = random_offset()
    x0 = (SRC_W - OUT_W) // 2 + dx
    y0 = (SRC_H - OUT_H) // 2 + dy
    x0 = max(0, min(SRC_W - OUT_W, x0))
    y0 = max(0, min(SRC_H - OUT_H, y0))

    cv2.imwrite(str(png_path), img[y0:y0 + OUT_H, x0:x0 + OUT_W])

    lm = np.array(gt['landmarks'], dtype=np.float32)
    lm[:, 0] = (lm[:, 0] * SRC_W - x0) / OUT_W
    lm[:, 1] = (lm[:, 1] * SRC_H - y0) / OUT_H
    gt['landmarks'] = lm.tolist()

    with open(json_path, 'w') as f:
        json.dump(gt, f)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?',
                    default=str(Path(__file__).parent / 'captures'),
                    help='captures/ root or any descendant directory')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    root = Path(args.root).expanduser()
    pngs = sorted(Path(p) for p in glob.glob(str(root / '**/*.png'), recursive=True))
    if not pngs:
        raise SystemExit(f'no PNGs found under {root}')

    n_ok = n_skip = 0
    for png in pngs:
        js = png.with_suffix('.json')
        if not js.exists():
            n_skip += 1
            continue
        if process(png, js):
            n_ok += 1
        else:
            n_skip += 1
    print(f'cropped {n_ok} / {len(pngs)}  (skipped {n_skip})')


if __name__ == '__main__':
    main()
