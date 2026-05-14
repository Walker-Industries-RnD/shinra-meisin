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

REAL_DIR      = os.path.expanduser('~/Downloads/shinra-meisin/captures/')
SYNTHETIC_DIR = os.path.expanduser('~/Downloads/datasets/synthetic_v2/')
WEBCAM_DIR    = os.path.expanduser('~/Downloads/datasets/HybridGaze/')

RAW_INPUT_W = 640
RAW_INPUT_H = 480

INIT_SCALE = 0.5  # 640×480 → 320x240
INPUT_W = int(RAW_INPUT_W * INIT_SCALE)
INPUT_H = int(RAW_INPUT_H * INIT_SCALE)

# Pupil diameter normaliser.  Real captures are saved in raw pixels (the
# capture script's MAX_PUPIL_DIAMETER=1), so dividing by this value puts them
# in the same broad scale as the GIW-era model checkpoints.
MAX_PUPIL_DIAMETER = 148.82
MAX_REFLECT_PAD = 36  # max reflect-pad in source-pixel space (synthetic 640×480; real 320×240)


# ── Real GT construction ─────────────────────────────────────────────────────

def _build_real_gt(label, flipped=False, pad=0, src_w=INPUT_W, src_h=INPUT_H):
    """Convert a capture JSON dict to model-ready tensors.

    Captures store landmarks already normalised to the saved image
    (src_w × src_h), pupil_diameter as raw pixels, and gaze_vector as a
    unit 3-vector with image-plane convention (+x right, +y down, +z toward
    camera).  We re-normalise landmarks onto the (reflect-padded) canvas and
    sign-flip gaze.x under hflip — parallel to _build_synthetic_gt.
    """
    gx, gy, gz = (float(v) for v in label['gaze_vector'])
    pw = src_w + 2 * pad
    ph = src_h + 2 * pad

    gt = {
        'gaze_vector': torch.tensor(
            [-gx if flipped else gx, gy, gz], dtype=torch.float32,
        ),
        'pupil_diameter': torch.tensor(
            float(label['pupil_diameter']) / MAX_PUPIL_DIAMETER, dtype=torch.float32,
        ),
        'pupil_valid': torch.tensor(bool(label.get('pupil_valid', True))),
        'lid_valid':   torch.tensor(bool(label.get('lid_valid',   True))),
        'eye_state':   torch.tensor(int(label['eye_state']), dtype=torch.int8),
    }

    lm = np.asarray(label['landmarks'], dtype=np.float32)   # (17, 2) in [0, 1]
    upper = lm[:8].copy()
    lower = lm[8:16].copy()
    pupil = lm[16].copy()

    # Normalize to a consistent left→right ordering before any augmentation.
    # eye0 (right eye) arrives right→left; eye1 (left eye) arrives left→right.
    # Detecting which: if upper[0].x > upper[7].x, it's right→left — reverse it.
    if upper[0, 0] > upper[7, 0]:
        upper = upper[::-1].copy()
        lower = lower[::-1].copy()

    # Now the existing flip augmentation is correct: it reverses into the
    # new image space after a horizontal flip, maintaining left→right in all cases.
    if flipped:
        upper = upper[::-1]
        lower = lower[::-1]

    def _renorm(pts):
        x = (pts[..., 0] * src_w + pad) / pw
        y = (pts[..., 1] * src_h + pad) / ph
        if flipped:
            x = 1.0 - x
        return np.stack([x, y], axis=-1).astype(np.float32)

    gt['landmarks'] = torch.cat([
        torch.from_numpy(_renorm(upper)),
        torch.from_numpy(_renorm(lower)),
        torch.from_numpy(_renorm(pupil)).unsqueeze(0),
    ], dim=0)   # (17, 2)

    return gt


# ── Transforms ────────────────────────────────────────────────────────────────

to_gray_tensor = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

_photometric = v2.Compose([
    v2.ColorJitter(brightness=0.3, contrast=0.3),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])


class RealTransform:
    def __call__(self, img):
        # Reflect-pad before resize so the eye appears zoomed out; landmarks
        # are adjusted in the dataset's GT builder using the returned pad value.
        pad = random.randint(0, MAX_REFLECT_PAD)
        if pad > 0:
            img = TF.pad(img, padding=pad, padding_mode='reflect')
        img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
        flipped = random.random() < 0.5
        if flipped:
            img = TF.horizontal_flip(img)
        img = _photometric(img)
        return img, flipped, pad


real_transforms = RealTransform()


# ── Dataset ───────────────────────────────────────────────────────────────────

class RealDataset(Dataset):
    """Captured webcam frames with per-image JSON GT, from capture_dataset.py.

    Layout:
      {root}/{run_ts}/{eye0|eye1}/{fi:06d}.png
      {root}/{run_ts}/{eye0|eye1}/{fi:06d}.json

    Each (run, eye) pair is one "session" so EyeStateBatchSampler keeps the
    same-state temporal runs of an individual eye intact.  Sessions expose an
    `eye_state` array indexed by frame-in-session, plus a `pngs` list giving
    the on-disk image paths.
    """

    def __init__(self, root: str = REAL_DIR, transforms=None):
        self.transforms = transforms
        self._sessions = []   # list of dicts: {'pngs', 'eye_state', 'src_wh'}
        self._index    = []   # (session_idx, frame_idx)

        for sess_dir in sorted(glob.glob(os.path.join(root, '*', 'eye*'))):
            pngs = sorted(glob.glob(os.path.join(sess_dir, '*.png')))
            if not pngs:
                continue

            states = []
            hard_flags = []
            kept_pngs = []
            for p in pngs:
                jp = os.path.splitext(p)[0] + '.json'
                if not os.path.exists(jp):
                    continue
                with open(jp) as f:
                    d = json.load(f)
                    states.append(int(d['eye_state']))
                    hard_flags.append(bool(d.get('hard', False)))
                kept_pngs.append(p)
            if not kept_pngs:
                continue

            probe = cv2.imread(kept_pngs[0], cv2.IMREAD_GRAYSCALE)
            src_wh = (probe.shape[1], probe.shape[0]) if probe is not None else (INPUT_W, INPUT_H)

            si = len(self._sessions)
            self._sessions.append({
                'pngs':      kept_pngs,
                'eye_state': np.asarray(states, dtype=np.uint8),
                'hard':      np.asarray(hard_flags, dtype=bool),
                'src_wh':    src_wh,
            })
            for fi in range(len(kept_pngs)):
                self._index.append((si, fi))

        print(f'Real | {len(self._index)} frames loaded.')

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        si, fi = self._index[idx]
        sess = self._sessions[si]
        png  = sess['pngs'][fi]

        img = cv2.imread(png)
        if img is None:
            raise RuntimeError(f'missing capture {png}')
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)

        with open(os.path.splitext(png)[0] + '.json') as f:
            label = json.load(f)

        if self.transforms is not None:
            img, flipped, pad = self.transforms(img)
        else:
            img = TF.resize(img, [INPUT_H, INPUT_W], antialias=True)
            img = to_gray_tensor(img)
            flipped = False
            pad = 0

        src_w, src_h = sess['src_wh']
        return img, _build_real_gt(label, flipped=flipped, pad=pad,
                                   src_w=src_w, src_h=src_h)


# ── Batch sampler ─────────────────────────────────────────────────────────────

def _build_real_runs(dataset) -> list[list[int]]:
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


def _get_subset_weights(subset, hard_weight: float = 2.0) -> np.ndarray:
    """Per-frame sampling weights: hard_weight for hard frames, 1.0 for normal."""
    if isinstance(subset, Subset):
        base = subset.dataset
        indices = list(subset.indices)
    else:
        base = subset
        indices = list(range(len(subset)))
    weights = np.ones(len(indices), dtype=np.float32)
    for local_i, global_i in enumerate(indices):
        si, fi = base._index[global_i]
        if base._sessions[si]['hard'][fi]:
            weights[local_i] = hard_weight
    return weights


class EyeStateBatchSampler(Sampler):
    """Streams contiguous same-state runs in a shuffled order, slicing into
    fixed-size batches.  A batch may straddle two runs; the next batch picks
    up where the previous one left off.  Run order is reshuffled each epoch.

    Works with both RealDataset directly and torch.utils.data.Subset wrapping one.
    """

    def __init__(self, dataset, batch_size: int, drop_last: bool = True):
        self._runs = _build_real_runs(dataset)
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


def session_split(dataset: 'RealDataset', train_frac: float = 0.7, seed: int = 0):
    """Split RealDataset into train/val Subsets at the session level.

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
    # gaze direction in scene space.  NOTE: verify axis convention matches the
    # capture script's (+x right, +y down, +z toward camera) before trusting
    # fine-grained gaze error metrics.
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
    """GT tensors from a parsed synthetic label; parallel to _build_real_gt.

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
        'eye_state':   torch.tensor(0, dtype=torch.int8),  # 0 = synthetic sentinel (real uses 3-5)
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
    """Synthetic + real-capture dataset with per-epoch reshuffling.

    The synthetic-to-real ratio is fixed for the entire phase:

      Phase targets:
        phase 0 → 1.00   phase 1 → ~0.77
        phase 2 → ~0.53  phase 3 →  0.30

    Call reshuffle() at the start of each epoch.
    The DataLoader should use shuffle=False (SequentialSampler).

    epoch_size: total items per epoch (defaults to len(real_subset)).
      For phase 0, if real_subset is empty you must supply epoch_size explicitly.
    """

    def __init__(
        self,
        real_subset,
        synthetic: SyntheticDataset,
        phase: int,
        epoch_size: int | None = None,
        hard_weight: float = 2.0,
    ):
        assert 0 <= phase <= 3, f'phase must be 0-3, got {phase}'
        self._real       = real_subset
        self._syn        = synthetic
        self._phase      = phase
        self._epoch_size = epoch_size if epoch_size is not None else len(real_subset)
        self._real_runs  = _build_real_runs(real_subset) if len(real_subset) > 0 else []
        self._real_weights = (
            _get_subset_weights(real_subset, hard_weight)
            if len(real_subset) > 0 else np.array([], dtype=np.float32)
        )
        self._syn_perm:  list[int] = []
        self._real_perm: list[int] = []
        self._is_syn:    np.ndarray = np.zeros(0, dtype=bool)
        self._cycle_pos: np.ndarray = np.zeros(0, dtype=np.int64)
        self.reshuffle(epoch_progress=0.0)

    # ------------------------------------------------------------------

    def reshuffle(self, epoch_progress: float = 0.0, seed: int | None = None) -> None:
        """Regenerate epoch ordering.  Call once per epoch before iterating.

        We keep just one shuffled permutation per pool (size = pool size) and
        a per-slot (type, cycle-pos) pair.  Source indices are derived in
        __getitem__ via `perm[pos % len(perm)]`, so a small real pool can
        cycle as many times as the phase ratio demands without materialising
        the cycled sequence — peak extra memory stays at ~9 bytes/slot
        regardless of how often real wraps.
        """
        rng    = random.Random(seed)
        np_rng = np.random.default_rng(rng.getrandbits(32))

        total    = self._epoch_size
        syn_frac = _PHASE_SYN_FRAC[self._phase]
        n_syn  = round(total * syn_frac)
        n_real = total - n_syn

        # Base permutations: one entry per unique source sample.
        syn_perm = list(range(len(self._syn)))
        rng.shuffle(syn_perm)
        self._syn_perm = syn_perm

        if n_real and self._real_runs:
            runs = list(self._real_runs)
            rng.shuffle(runs)
            flat = [i for run in runs for i in run]
            if len(self._real_weights) > 0:
                w = self._real_weights[flat]
                w = w / w.sum()
                self._real_perm = list(np_rng.choice(flat, size=len(flat), replace=True, p=w))
            else:
                self._real_perm = flat
        else:
            self._real_perm = []

        # Compact per-slot schedule.  is_syn is `total` bytes; cycle_pos is
        # `total` int64s — both linear in epoch_size, independent of how
        # many times either pool cycles.
        is_syn = np.zeros(total, dtype=bool)
        if n_syn:
            is_syn[np_rng.choice(total, size=n_syn, replace=False)] = True
        self._is_syn    = is_syn
        self._cycle_pos = np.where(
            is_syn, np.cumsum(is_syn) - 1, np.cumsum(~is_syn) - 1,
        ).astype(np.int64)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._epoch_size

    def __getitem__(self, idx: int):
        pos = int(self._cycle_pos[idx])
        if self._is_syn[idx]:
            return self._syn[self._syn_perm[pos % len(self._syn_perm)]]
        return self._real[self._real_perm[pos % len(self._real_perm)]]


cam_transforms = v2.Compose([
    v2.Resize((INPUT_H, INPUT_W)),
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
