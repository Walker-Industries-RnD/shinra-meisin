import cv2
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF

GIW_DIR = '/home/john/Downloads/GIW'

# RAW_INPUT_(W|H) - expected camera input
RAW_INPUT_W = 640
RAW_INPUT_H = 480

# INIT_SCALE - maximum possible resolution handled by shinra-meisin 
INIT_SCALE = 0.4 # 640x480 * 0.4 -> 256x192
INPUT_W = int(RAW_INPUT_W * INIT_SCALE)
INPUT_H = int(RAW_INPUT_H * INIT_SCALE)

# lid_lm_2D.txt has 34 landmarks: 17 upper then 17 lower.
# Subsample each half to 8 evenly-spaced points → 16 eyelid + 1 pupil = 17 heatmaps.
_LID_HALF = 17
_N_SAMPLE  = 8
_LID_IDX   = np.round(np.linspace(0, _LID_HALF - 1, _N_SAMPLE)).astype(int)

MAX_PUPIL_DIAMETER = 148.82 # empirically verified across the entire GIW dataset, at least what i've restructured


def _to_hm_giw(px, py, flipped, y_inverted=False):
    """Map a raw 640×480 GIW landmark to INPUT_WxINPUT_H heatmap coordinates.
    y_inverted: eye0 cameras store y from the bottom of the frame; correct before mapping.
    """
    if y_inverted:
        py = RAW_INPUT_H - py
    hm_x = (px * INIT_SCALE)
    hm_y = (py * INIT_SCALE)
    if flipped:
        hm_x = INPUT_W - hm_x
    return hm_x, hm_y


to_gray_tensor = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

_photometric = v2.Compose([
    v2.ColorJitter(brightness=0.225, contrast=0.225),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])


# ── Annotation parsing ────────────────────────────────────────────────────────

def _parse_sv(path, skip_cols, skip_lines=1):
    """Semicolon-delimited GIW file → float32 array (n_frames, remaining_cols).

    skip_lines: header lines to discard before data (default 1).
                eye_movements.txt needs 3 (description + blank + column header).
    """
    rows = []
    with open(path) as f:
        for _ in range(skip_lines):
            next(f)
        for line in f:
            parts = line.strip().rstrip(';').split(';')
            rows.append([float(v) for v in parts[skip_cols:]])
    return np.array(rows, dtype=np.float32)


def load_session(session_dir, eye):
    """Parse GT files for one eye video; returns per-frame arrays for all N frames.

    Keys:
      video_path   — str, path to eye{N}.mp4
      n_frames     — int, total frame count
      gaze_xyz     — (N, 3) float32, unit gaze vector X Y Z
      pupil_cx/cy  — (N,) float32, pupil ellipse centre in 640×480 px
      pupil_w      — (N,) float32, width of pupil ellipse (argument 3 of pupil_eli)
      lid_pts      — (N, 16, 2) float32, 8 upper + 8 lower subsampled landmarks
      pupil_valid  — (N,) bool, validity_pupil.txt flag
      lid_valid    — (N,) bool, validity_lid.txt flag
      eye_state    — (N,) uint8, 1=Error 2=Blink 3=Fixation 4=Saccade 5=SmoothPursuit
    """
    base = os.path.join(session_dir, f'{eye}.mp4')

    # gaze_vec:        FRAME | X Y Z                 → skip_lines=1, skip_cols=1
    # pupil_eli:       FRAME | ANGLE CX CY W H        → skip_lines=1, skip_cols=1
    # lid_lm_2D:       FRAME | INACCURACY | 34*(X Y)  → skip_lines=1, skip_cols=2
    # validity:        FRAME | VALIDITY                → skip_lines=1, skip_cols=1
    # eye_movements:   description + blank + header    → skip_lines=3, skip_cols=1
    gaze  = _parse_sv(base + 'gaze_vec.txt',        skip_cols=1)              # (N, 3)
    pupil = _parse_sv(base + 'pupil_eli.txt',       skip_cols=1)              # (N, 5)
    lids  = _parse_sv(base + 'lid_lm_2D.txt',       skip_cols=2)              # (N, 68)
    v_pu  = _parse_sv(base + 'validity_pupil.txt',  skip_cols=1).ravel().astype(bool)
    v_lid = _parse_sv(base + 'validity_lid.txt',    skip_cols=1).ravel().astype(bool)
    emov  = _parse_sv(base + 'eye_movements.txt',   skip_cols=1, skip_lines=3).ravel().astype(np.uint8)

    n = len(gaze)
    lids  = lids.reshape(n, _LID_HALF * 2, 2)
    upper = lids[:, :_LID_HALF][:, _LID_IDX]   # (N, 8, 2)
    lower = lids[:, _LID_HALF:][:, _LID_IDX]   # (N, 8, 2)

    return {
        'video_path':  base,
        'n_frames':    n,
        'gaze_xyz':    gaze,                                       # (N, 3)
        'pupil_cx':    pupil[:, 1],                                # (N,)
        'pupil_cy':    pupil[:, 2],                                # (N,)
        'pupil_w':     pupil[:, 3],                                # (N,)  WIDTH — "argument 3"
        'lid_pts':     np.concatenate([upper, lower], axis=1),    # (N, 16, 2)
        'pupil_valid': v_pu,                                       # (N,) bool
        'lid_valid':   v_lid,                                      # (N,) bool
        'eye_state':   emov,                                       # (N,) uint8
        # eye0 cameras record with y-from-bottom convention (verified by pixel intensity test).
        'y_inverted':  eye == 'eye0',
    }


# ── GT construction ───────────────────────────────────────────────────────────

def convert_giw(row, flipped=False, y_inverted=False):
    gx, gy, gz    = row['gaze_xyz']
    lid_pts       = row['lid_pts']         # (16, 2)
    pupil_valid   = bool(row['pupil_valid'])
    lid_valid     = bool(row['lid_valid'])

    gt = {}
    gt['gaze_vector']    = torch.tensor(
        [-float(gx) if flipped else float(gx), float(gy), float(gz)],
        dtype=torch.float32,
    )
    gt['pupil_diameter'] = torch.tensor(float(row['pupil_w']) / MAX_PUPIL_DIAMETER, dtype=torch.float32)
    gt['pupil_valid']    = torch.tensor(pupil_valid)
    gt['lid_valid']      = torch.tensor(lid_valid)
    gt['eye_state']      = torch.tensor(int(row['eye_state']), dtype=torch.int8)

    upper = lid_pts[:8].copy()   # (8, 2) raw 640×480
    lower = lid_pts[8:].copy()   # (8, 2) raw 640×480
    if flipped:
        # Reverse within each half so channel N stays semantically consistent.
        upper = upper[::-1]
        lower = lower[::-1]

    def _norm(pts):
        """Raw 640×480 landmark array (N,2) → [0,1] normalized, with geometric corrections."""
        x = pts[:, 0] / RAW_INPUT_W
        y = pts[:, 1] / RAW_INPUT_H
        if y_inverted:
            y = 1.0 - y
        if flipped:
            x = 1.0 - x
        return np.stack([x, y], axis=-1).astype(np.float32)

    px = row['pupil_cx'] / RAW_INPUT_W
    py = row['pupil_cy'] / RAW_INPUT_H
    if y_inverted:
        py = 1.0 - py
    if flipped:
        px = 1.0 - px

    upper = torch.from_numpy(_norm(upper))                   # (8, 2) in [0,1]
    lower = torch.from_numpy(_norm(lower))                   # (8, 2) in [0,1]
    pupil = torch.tensor([px, py], dtype=torch.float32)      # (2,)  in [0,1]

    gt['landmarks'] = torch.cat([upper, lower, pupil.unsqueeze(0)], dim=0)  # (17, 2)

    return gt


# ── Transform ─────────────────────────────────────────────────────────────────

class GIWTransform:
    def __call__(self, img):
        img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
        flipped = random.random() < 0.5
        if flipped:
            img = TF.horizontal_flip(img)
        img = _photometric(img)
        return img, flipped


giw_transforms = GIWTransform()


# ── Dataset ───────────────────────────────────────────────────────────────────

class GIWDataset(Dataset):
    """Streams frames from GIW IR eye-tracking videos with GW annotation GT.

    Directory layout expected:
      {root}/{outer}/{session}/eye{N}.mp4
      {root}/{outer}/{session}/eye{N}.mp4gaze_vec.txt
      {root}/{outer}/{session}/eye{N}.mp4pupil_eli.txt
      {root}/{outer}/{session}/eye{N}.mp4lid_lm_2D.txt
      {root}/{outer}/{session}/eye{N}.mp4validity_{pupil,lid}.txt

    All frames are included. Invalid frames have zeroed heatmap channels and
    gt['pupil_valid'] / gt['lid_valid'] == False so the training loop can mask loss.
    """

    def __init__(self, root=GIW_DIR, transforms=None):
        self.transforms = transforms
        self._sessions  = []
        self._index     = []   # (session_idx, frame_idx)

        for eye_path in sorted(glob.glob(os.path.join(root, '*', '*', 'eye?.mp4'))):
            session_dir = os.path.dirname(eye_path)
            eye = os.path.splitext(os.path.basename(eye_path))[0]
            try:
                data = load_session(session_dir, eye)
            except FileNotFoundError:
                continue
            si = len(self._sessions)
            self._sessions.append(data)
            for fi in range(0, data['n_frames'], 2):
                self._index.append((si, fi))
        print(f'GIW | {len(self._index)} frames loaded.')

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        si, fi = self._index[idx]
        sess   = self._sessions[si]

        cap = cv2.VideoCapture(sess['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"failed to read frame {fi} from {sess['video_path']}")

        # BGR → RGB, HWC → CHW tensor
        img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        row = {k: sess[k][fi] for k in
               ('gaze_xyz', 'pupil_cx', 'pupil_cy', 'pupil_w', 'lid_pts',
                'pupil_valid', 'lid_valid', 'eye_state')}

        if self.transforms is not None:
            img, flipped = self.transforms(img)
        else:
            img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
            img = to_gray_tensor(img)
            flipped = False

        return img, convert_giw(row, flipped=flipped,
                                y_inverted=sess['y_inverted'])