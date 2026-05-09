from pathlib import Path

import cv2
import glob
import json
import os
import random
import re
from PIL import Image
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as TF

GIW_DIR       = os.path.expanduser('~/Downloads/datasets/GIW/')
SYNTHETIC_DIR = os.path.expanduser('~/Downloads/datasets/synthetic_v2/')
WEBCAM_DIR    = os.path.expanduser('~/Downloads/datasets/HybridGaze/')

RAW_INPUT_W = 640
RAW_INPUT_H = 480

INIT_SCALE = 0.4  # 640×480 → 256×192
INPUT_W = int(RAW_INPUT_W * INIT_SCALE)
INPUT_H = int(RAW_INPUT_H * INIT_SCALE)

# lid_lm_2D.txt has 34 landmarks: 17 upper then 17 lower.
# Subsample each half to 8 evenly-spaced points → 16 eyelid + 1 pupil = 17 landmarks.
_LID_HALF = 17
_N_SAMPLE  = 8
_LID_IDX   = np.round(np.linspace(0, _LID_HALF - 1, _N_SAMPLE)).astype(int)

MAX_PUPIL_DIAMETER = 148.82  # empirically verified across the GIW dataset
MAX_REFLECT_PAD = 96  # pixels in raw 640×480 space; ~15% each side → eye at ~77% scale


# ── Annotation parsing ────────────────────────────────────────────────────────

def _parse_sv(path, skip_cols, skip_lines=1):
    """Semicolon-delimited GIW annotation file → float32 array (n_frames, remaining_cols).

    skip_lines: header lines to discard (eye_movements.txt needs 3).
    """
    rows = []
    with open(path) as f:
        for _ in range(skip_lines):
            next(f)
        for line in f:
            parts = line.strip().rstrip(';').split(';')
            rows.append([float(v) for v in parts[skip_cols:]])
    return np.array(rows, dtype=np.float32)


def _load_session(session_dir, eye):
    """Parse GT annotation files for one eye camera; returns per-frame arrays for all N frames.

    Keys:
      n_frames    — int
      gaze_xyz    — (N, 3) float32, unit gaze vector X Y Z
      pupil_cx/cy — (N,) float32, pupil ellipse centre in 640×480 px
      pupil_w     — (N,) float32, pupil ellipse width
      lid_pts     — (N, 16, 2) float32, 8 upper + 8 lower subsampled lid landmarks
      pupil_valid — (N,) bool
      lid_valid   — (N,) bool
      eye_state   — (N,) uint8, 1=Error 2=Blink 3=Fixation 4=Saccade 5=SmoothPursuit
      y_inverted  — bool, eye0 cameras store y from the bottom of the frame
    """
    prefix = os.path.join(session_dir, f'{eye}.mp4')

    # Column layout (after skipping FRAME col):
    #   gaze_vec.txt:       X Y Z
    #   pupil_eli.txt:      ANGLE CX CY W H
    #   lid_lm_2D.txt:      (skip INACCURACY too) 34*(X Y)
    #   validity_*.txt:     VALIDITY
    #   eye_movements.txt:  Eye Movement Type  (3-line header)
    gaze  = _parse_sv(prefix + 'gaze_vec.txt',       skip_cols=1)
    pupil = _parse_sv(prefix + 'pupil_eli.txt',      skip_cols=1)
    lids  = _parse_sv(prefix + 'lid_lm_2D.txt',      skip_cols=2)
    v_pu  = _parse_sv(prefix + 'validity_pupil.txt', skip_cols=1).ravel().astype(bool)
    v_lid = _parse_sv(prefix + 'validity_lid.txt',   skip_cols=1).ravel().astype(bool)
    emov  = _parse_sv(prefix + 'eye_movements.txt',  skip_cols=1, skip_lines=3).ravel().astype(np.uint8)

    n = len(gaze)
    lids  = lids.reshape(n, _LID_HALF * 2, 2)
    upper = lids[:, :_LID_HALF][:, _LID_IDX]   # (N, 8, 2)
    lower = lids[:, _LID_HALF:][:, _LID_IDX]   # (N, 8, 2)

    return {
        'n_frames':    n,
        'gaze_xyz':    gaze,
        'pupil_cx':    pupil[:, 1],
        'pupil_cy':    pupil[:, 2],
        'pupil_w':     pupil[:, 3],
        'lid_pts':     np.concatenate([upper, lower], axis=1),   # (N, 16, 2)
        'pupil_valid': v_pu,
        'lid_valid':   v_lid,
        'eye_state':   emov,
        'y_inverted':  eye == 'eye0',
    }


# ── GT construction ───────────────────────────────────────────────────────────

def _build_gt(row, flipped, y_inverted, pad=0):
    """Convert a single-frame annotation row to model-ready tensors.

    Returns:
      gaze_vector    — (3,) float32, unit gaze vector (x sign flipped when flipped=True)
      pupil_diameter — scalar float32, width / MAX_PUPIL_DIAMETER
      pupil_valid    — bool tensor
      lid_valid      — bool tensor
      eye_state      — int8 tensor
      landmarks      — (17, 2) float32 in [0, 1]: 8 upper + 8 lower lid + 1 pupil centre
    """
    gx, gy, gz = row['gaze_xyz']

    gt = {
        'gaze_vector':    torch.tensor(
            [-float(gx) if flipped else float(gx), float(gy), float(gz)],
            dtype=torch.float32,
        ),
        'pupil_diameter': torch.tensor(float(row['pupil_w']) / MAX_PUPIL_DIAMETER, dtype=torch.float32),
        'pupil_valid':    torch.tensor(bool(row['pupil_valid'])),
        'lid_valid':      torch.tensor(bool(row['lid_valid'])),
        'eye_state':      torch.tensor(int(row['eye_state']), dtype=torch.int8),
    }

    upper = row['lid_pts'][:8].copy()   # (8, 2) raw 640×480
    lower = row['lid_pts'][8:].copy()   # (8, 2) raw 640×480
    if flipped:
        # Reverse within each half so channel N stays semantically consistent.
        upper = upper[::-1]
        lower = lower[::-1]

    # Padded canvas dimensions (pad=0 → original 640×480 behaviour unchanged).
    pw = RAW_INPUT_W + 2 * pad
    ph = RAW_INPUT_H + 2 * pad

    def _norm(pts):
        x = (pts[:, 0] + pad) / pw
        # Convert to "pixels from top" before normalizing so padding offset is uniform.
        y_top = RAW_INPUT_H - pts[:, 1] if y_inverted else pts[:, 1]
        y = (y_top + pad) / ph
        if flipped:
            x = 1.0 - x
        return np.stack([x, y], axis=-1).astype(np.float32)

    px = (row['pupil_cx'] + pad) / pw
    py_top = RAW_INPUT_H - row['pupil_cy'] if y_inverted else row['pupil_cy']
    py = (py_top + pad) / ph
    if flipped:
        px = 1.0 - px

    pupil = torch.tensor([px, py], dtype=torch.float32)
    gt['landmarks'] = torch.cat([
        torch.from_numpy(_norm(upper)),
        torch.from_numpy(_norm(lower)),
        pupil.unsqueeze(0),
    ], dim=0)   # (17, 2)

    return gt


# ── Transforms ────────────────────────────────────────────────────────────────

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


class GIWTransform:
    def __call__(self, img):
        # Reflect-pad before resize so the eye appears zoomed out; landmarks
        # are adjusted in _build_gt using the returned pad value.
        pad = random.randint(0, MAX_REFLECT_PAD)
        if pad > 0:
            img = TF.pad(img, padding=pad, padding_mode='reflect')
        img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
        flipped = random.random() < 0.5
        if flipped:
            img = TF.horizontal_flip(img)
        img = _photometric(img)
        return img, flipped, pad


giw_transforms = GIWTransform()


# ── Dataset ───────────────────────────────────────────────────────────────────

class GIWDataset(Dataset):
    """Pre-extracted PNG frames from GIW with annotation GT.

    Layout:
      {root}/{outer}/{session}/frames/eye{N}/{fi:06d}.png
      {root}/{outer}/{session}/eye{N}.mp4gaze_vec.txt   (and other annotation files)

    JPEG filenames are 0-indexed; GT files are 1-indexed (FRAME col skipped), so
    fi=0 maps to GT array row 0 = FRAME 1.
    """

    def __init__(self, root=GIW_DIR, transforms=None):
        self.transforms = transforms
        self._sessions  = []
        self._index     = []   # (session_idx, frame_idx)

        for frames_path in sorted(glob.glob(os.path.join(root, '*', '*', 'frames', 'eye*'))):
            eye         = os.path.basename(frames_path)
            session_dir = os.path.dirname(os.path.dirname(frames_path))
            try:
                sess = _load_session(session_dir, eye)
            except FileNotFoundError:
                continue

            jpegs = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
            if not jpegs:
                continue

            sess['frames_path'] = frames_path
            si   = len(self._sessions)
            n_gt = sess['n_frames']
            self._sessions.append(sess)
            for p in jpegs:
                fi = int(os.path.splitext(os.path.basename(p))[0])
                if fi < n_gt:
                    self._index.append((si, fi))

        print(f'GIW | {len(self._index)} frames loaded.')

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        si, fi = self._index[idx]
        sess   = self._sessions[si]

        img = cv2.imread(os.path.join(sess['frames_path'], f'{fi:06d}.jpg'))
        if img is None:
            raise RuntimeError(f"missing frame {fi:06d}.png in {sess['frames_path']}")

        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        row = {k: sess[k][fi] for k in
               ('gaze_xyz', 'pupil_cx', 'pupil_cy', 'pupil_w', 'lid_pts',
                'pupil_valid', 'lid_valid', 'eye_state')}

        if self.transforms is not None:
            img, flipped, pad = self.transforms(img)
        else:
            img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
            img = to_gray_tensor(img)
            flipped = False
            pad = 0

        return img, _build_gt(row, flipped=flipped, y_inverted=sess['y_inverted'], pad=pad)


# ── Batch sampler ─────────────────────────────────────────────────────────────

def _build_giw_runs(dataset) -> list[list[int]]:
    """Same-state contiguous runs as lists of local indices into dataset/subset."""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        local_to_global = dataset.indices
    else:
        base = dataset
        local_to_global = range(len(dataset))

    by_session: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for local_i, global_i in enumerate(local_to_global):
        si, fi = base._index[global_i]
        state = int(base._sessions[si]['eye_state'][fi])
        by_session[si].append((fi, state, local_i))
    for entries in by_session.values():
        entries.sort(key=lambda x: x[0])

    runs: list[list[int]] = []
    for entries in by_session.values():
        run = [entries[0][2]]
        prev_fi, prev_state = entries[0][0], entries[0][1]
        for fi, state, local_i in entries[1:]:
            if state == prev_state and fi > prev_fi:
                run.append(local_i)
            else:
                runs.append(run)
                run = [local_i]
                prev_state = state
            prev_fi = fi
        runs.append(run)

    return runs


class EyeStateBatchSampler(Sampler):
    """Streams contiguous same-state runs in a shuffled order, slicing into
    fixed-size batches.  A batch may straddle two runs; the next batch picks
    up where the previous one left off.  Run order is reshuffled each epoch.

    Works with both GIWDataset directly and torch.utils.data.Subset wrapping one.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        self._runs = _build_giw_runs(dataset)
        self._batch_size = batch_size
        self._drop_last = drop_last

    def __iter__(self):
        runs = list(self._runs)
        random.shuffle(runs)
        stream = [idx for run in runs for idx in run]
        for start in range(0, len(stream), self._batch_size):
            chunk = stream[start:start + self._batch_size]
            if self._drop_last and len(chunk) < self._batch_size:
                break
            yield chunk

    def __len__(self) -> int:
        total = sum(len(r) for r in self._runs)
        return total // self._batch_size if self._drop_last else -(-total // self._batch_size)


def session_split(dataset: 'GIWDataset', train_frac: float = 0.7, seed: int = 0):
    """Split GIWDataset into train/val Subsets at the session level.

    All frames from a session land entirely in one split, keeping temporal
    runs intact for EyeStateBatchSampler.  Returns (train_subset, val_subset).
    """
    n_sessions = len(dataset._sessions)
    session_ids = list(range(n_sessions))
    rng = random.Random(seed)
    rng.shuffle(session_ids)

    train_sessions = set(session_ids[:max(1, int(n_sessions * train_frac))])

    train_idx = [i for i, (si, _) in enumerate(dataset._index) if si in train_sessions]
    val_idx   = [i for i, (si, _) in enumerate(dataset._index) if si not in train_sessions]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ── Synthetic dataset ─────────────────────────────────────────────────────────

_VEC_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][+-]?\d+)?')


def _parse_tuple(s: str) -> list[float]:
    return [float(x) for x in _VEC_RE.findall(s)]


def _parse_synthetic_label(data: dict) -> dict:
    """Extract GT fields from a synthetic_v2 JSON dict."""
    ed  = data['eye_details']
    cam = data['cameras']['cam0']

    # look_vec is a unit quaternion-padded tuple; first 3 coords are the unit
    # gaze direction in scene space.  NOTE: verify axis convention matches GIW
    # before trusting fine-grained gaze error metrics.
    gaze_xyz = np.array(_parse_tuple(ed['look_vec'])[:3], dtype=np.float32)

    # Iris pixel diameter from the 32 boundary points; pupil fraction scales it.
    iris_pts  = np.array([_parse_tuple(p)[:2] for p in cam['iris_2d']], dtype=np.float32)
    iris_diam = (
        (iris_pts[:, 0].max() - iris_pts[:, 0].min()) +
        (iris_pts[:, 1].max() - iris_pts[:, 1].min())
    ) / 2.0
    pupil_diam_px = iris_diam * float(ed['pupil_size']) / float(ed['iris_size'])

    upper_pts = np.array(
        [_parse_tuple(pt['pos'])[:2] for pt in cam['upper_interior_margin_2d']],
        dtype=np.float32,
    )  # (8, 2)
    lower_pts = np.array(
        [_parse_tuple(pt['pos'])[:2] for pt in cam['lower_interior_margin_2d']],
        dtype=np.float32,
    )  # (8, 2)
    pupil_xy = np.array(_parse_tuple(cam['pupil_2d'])[:2], dtype=np.float32)

    return {
        'gaze_xyz':     gaze_xyz,
        'pupil_diam_px': pupil_diam_px,
        'upper_pts':    upper_pts,
        'lower_pts':    lower_pts,
        'pupil_xy':     pupil_xy,
    }


def _build_synthetic_gt(label: dict, flipped: bool, pad: int = 0) -> dict:
    """GT tensors from a parsed synthetic label; parallel to _build_gt.

    Synthetic images use standard top-left origin (y increases downward),
    so no y-inversion is needed.
    """
    gx, gy, gz = label['gaze_xyz']
    pw = RAW_INPUT_W + 2 * pad
    ph = RAW_INPUT_H + 2 * pad

    gt = {
        'gaze_vector': torch.tensor(
            [-float(gx) if flipped else float(gx), float(gy), float(gz)],
            dtype=torch.float32,
        ),
        'pupil_diameter': torch.tensor(
            float(label['pupil_diam_px']) / MAX_PUPIL_DIAMETER, dtype=torch.float32,
        ),
        'pupil_valid': torch.tensor(True),
        'lid_valid':   torch.tensor(True),
        'eye_state':   torch.tensor(0, dtype=torch.int8),  # 0 = synthetic sentinel (GIW uses 1-5)
    }

    upper = label['upper_pts'].copy()
    lower = label['lower_pts'].copy()
    if flipped:
        upper = upper[::-1]
        lower = lower[::-1]

    def _norm(pts: np.ndarray) -> np.ndarray:
        x = (pts[:, 0] + pad) / pw
        y = (pts[:, 1] + pad) / ph
        if flipped:
            x = 1.0 - x
        return np.stack([x, y], axis=-1).astype(np.float32)

    px = (label['pupil_xy'][0] + pad) / pw
    py = (label['pupil_xy'][1] + pad) / ph
    if flipped:
        px = 1.0 - px

    pupil = torch.tensor([px, py], dtype=torch.float32)
    gt['landmarks'] = torch.cat([
        torch.from_numpy(_norm(upper)),
        torch.from_numpy(_norm(lower)),
        pupil.unsqueeze(0),
    ], dim=0)  # (17, 2)

    return gt


class SyntheticDataset(Dataset):
    """640×480 rendered eye images from synthetic_v2 with JSON GT.

    Layout:
      {root}/images/{id}.jpg
      {root}/labels/{id}.json
    """

    def __init__(self, root: str = SYNTHETIC_DIR, transforms=None):
        self.transforms = transforms
        self._img_dir = os.path.join(root, 'images')
        self._lbl_dir = os.path.join(root, 'labels')
        self._ids = sorted(
            int(os.path.splitext(f)[0])
            for f in os.listdir(self._lbl_dir)
            if f.endswith('.json')
        )
        print(f'Synthetic | {len(self._ids)} frames loaded.')

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int):
        stem = self._ids[idx]
        img = cv2.imread(os.path.join(self._img_dir, f'{stem}.jpg'))
        if img is None:
            raise RuntimeError(f'missing synthetic image {stem}.jpg')
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        with open(os.path.join(self._lbl_dir, f'{stem}.json')) as f:
            data = json.load(f)
        label = _parse_synthetic_label(data)

        if self.transforms is not None:
            img, flipped, pad = self.transforms(img)
        else:
            img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
            img = to_gray_tensor(img)
            flipped = False
            pad = 0

        return img, _build_synthetic_gt(label, flipped=flipped, pad=pad)


# ── Shared (interleaved) dataset ──────────────────────────────────────────────

# Synthetic fraction per phase: 1.0 at phase 0, dropping linearly to 0.3 at phase 3.
_PHASE_SYN_FRAC = [1.0 - 0.7 * p / 3 for p in range(4)]


class SharedDataset(Dataset):
    """Synthetic + real-GIW dataset with per-epoch reshuffling.

    Each epoch is laid out as:
      1. N_syn synthetic samples drawn at random (shuffled)
      2. N_real GIW frames organised as shuffled same-state contiguous runs

    Call reshuffle() at the start of each epoch to regenerate the ordering.
    The DataLoader should use shuffle=False (SequentialSampler).

    Synthetic-to-real schedule (linear):
      phase 0 → 100 % synthetic / 0 % real
      phase 1 → ~77 % synthetic / ~23 % real
      phase 2 → ~53 % synthetic / ~47 % real
      phase 3 →  30 % synthetic / 70 % real

    epoch_size: total items per epoch (defaults to len(giw_subset)).
      For phase 0, if giw_subset is empty you must supply epoch_size explicitly.
    """

    def __init__(
        self,
        giw_subset,
        synthetic: SyntheticDataset,
        phase: int,
        epoch_size: int | None = None,
    ):
        assert 0 <= phase <= 3, f'phase must be 0-3, got {phase}'
        self._giw  = giw_subset
        self._syn  = synthetic
        self._phase = phase

        syn_frac   = _PHASE_SYN_FRAC[phase]
        total      = epoch_size if epoch_size is not None else len(giw_subset)
        self._n_syn  = round(total * syn_frac)
        self._n_real = total - self._n_syn

        self._giw_runs   = _build_giw_runs(giw_subset) if self._n_real > 0 else []
        self._syn_order:  list[int] = []
        self._real_order: list[int] = []
        self.reshuffle()

    # ------------------------------------------------------------------

    def reshuffle(self, seed: int | None = None) -> None:
        """Regenerate epoch ordering.  Call once per epoch before iterating."""
        rng = random.Random(seed)

        syn_pool = list(range(len(self._syn)))
        rng.shuffle(syn_pool)
        self._syn_order = syn_pool[:self._n_syn]

        self._real_order = []
        if self._n_real > 0:
            runs = list(self._giw_runs)
            rng.shuffle(runs)
            flat = [i for run in runs for i in run]
            self._real_order = flat[:self._n_real]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._syn_order) + len(self._real_order)

    def __getitem__(self, idx: int):
        n = len(self._syn_order)
        if idx < n:
            return self._syn[self._syn_order[idx]]
        return self._giw[self._real_order[idx - n]]
    
cam_transforms = v2.Compose([
    v2.Resize((192, 256)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5])
])
    
class WebcamDataset(Dataset):
    def __init__(self, root: str = WEBCAM_DIR, transforms=None):
        self.paths = list(Path(root).rglob('*.jpg'))
        self.transform = transforms

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')  # grayscale to match IR
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)
