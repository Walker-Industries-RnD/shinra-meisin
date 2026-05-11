"""Calibration-driven webcam capture for the shinra-meisin training format.

Drops 640x480 grayscale PNGs (each crop centered on a single pupil so both
eyes typically fall within the frame) plus a sibling JSON containing GT
fields compatible with input.py:

  landmarks      — (17, 2) in [0, 1]: 8 upper lid + 8 lower lid + pupil center
  gaze_vector    — (3,) unit vector, anatomical eyeball model
  pupil_diameter — scalar, geometric mean of ellipse axes / MAX_PUPIL_DIAMETER
  eye_state      — 3 Fixation | 4 Saccade | 5 SmoothPursuit (GIW encoding)
  pupil_valid    — always True for saved frames
  lid_valid      — always True for saved frames

Run, then press Q to toggle logging on (press again to pause).  The
calibration overlay guides your gaze through fixation → saccade →
smooth-pursuit states.  Esc to abort.
"""

import argparse
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from pynput import keyboard

MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
              'face_landmarker/float16/1/face_landmarker.task')
MODEL_PATH = Path(__file__).parent / 'face_landmarker.task'


# ── Constants ─────────────────────────────────────────────────────────────────

CROP_W, CROP_H       = 640, 480
FRAMES_PER_PHASE     = 1000
FRAMES_PER_POSITION  = 50
MAX_PUPIL_DIAMETER   = 1

ES_FIXATION = 3
ES_SACCADE  = 4
ES_PURSUIT  = 5


def _pingpong(t):
    """Triangle wave in [0, 1]: 0→1 over half period, then 1→0."""
    u = t % 2.0
    return u if u <= 1.0 else 2.0 - u

# MediaPipe FaceMesh contour indices (outer canthus → inner canthus).
# 9-pt arcs are linspaced down to 8, dropping the middle element to keep both
# corners — mirrors input.py's `_LID_IDX = np.round(np.linspace(0, 16, 8))`.
_PICK8 = [0, 1, 2, 3, 5, 6, 7, 8]

_R_UPPER_FULL = [33, 246, 161, 160, 159, 158, 157, 173, 133]
_R_LOWER_FULL = [33,   7, 163, 144, 145, 153, 154, 155, 133]
_L_UPPER_FULL = [263, 466, 388, 387, 386, 385, 384, 398, 362]
_L_LOWER_FULL = [263, 249, 390, 373, 374, 380, 381, 382, 362]

R_UPPER = [_R_UPPER_FULL[i] for i in _PICK8]
R_LOWER = [_R_LOWER_FULL[i] for i in _PICK8]
L_UPPER = [_L_UPPER_FULL[i] for i in _PICK8]
L_LOWER = [_L_LOWER_FULL[i] for i in _PICK8]

R_OUTER, R_INNER = 33,  133
L_OUTER, L_INNER = 263, 362

R_IRIS_CENTER, R_IRIS_BOUND = 468, [469, 470, 471, 472]
L_IRIS_CENTER, L_IRIS_BOUND = 473, [474, 475, 476, 477]

EYE_SPECS = {
    'eye0': dict(upper=R_UPPER, lower=R_LOWER, outer=R_OUTER, inner=R_INNER,
                 iris_c=R_IRIS_CENTER, iris_b=R_IRIS_BOUND),
    'eye1': dict(upper=L_UPPER, lower=L_LOWER, outer=L_OUTER, inner=L_INNER,
                 iris_c=L_IRIS_CENTER, iris_b=L_IRIS_BOUND),
}


# ── Pupil ellipse detection ──────────────────────────────────────────────────

def detect_pupil(frame_bgr, iris_pts):
    """Threshold the iris-bounded ROI on HSV V-channel, fit ellipse on the
    largest dark blob.  Operating on V (= max(B,G,R)) gives much cleaner
    pupil/iris separation than grayscale luminance because a pigmented iris
    keeps a high value in its dominant channel while the pupil is dark across
    all channels.

    Returns (cx, cy, axis_w, axis_h, angle_deg) in original image coords
    or None on failure.
    """
    margin = 6
    x0 = max(0, int(iris_pts[:, 0].min()) - margin)
    y0 = max(0, int(iris_pts[:, 1].min()) - margin)
    x1 = min(frame_bgr.shape[1], int(iris_pts[:, 0].max()) + margin)
    y1 = min(frame_bgr.shape[0], int(iris_pts[:, 1].max()) + margin)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None

    roi_bgr = frame_bgr[y0:y1, x0:x1]
    v = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    roi = cv2.GaussianBlur(v, (5, 5), 0)
    thr = int(np.percentile(roi, 20))
    _, mask = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5 or cv2.contourArea(cnt) < 8:
        return None
    (cx, cy), (a, b), ang = cv2.fitEllipse(cnt)
    return (cx + x0, cy + y0, a, b, ang)


def iris_fallback(iris_center, iris_bound):
    diam = float(2 * np.mean(np.linalg.norm(iris_bound - iris_center, axis=1)))
    return (float(iris_center[0]), float(iris_center[1]), diam, diam, 0.0)


# ── Gaze vector ──────────────────────────────────────────────────────────────

def gaze_vector(pupil_xy, outer_xy, inner_xy, radius_factor=0.5):
    """Spherical eyeball model: unit gaze derived from in-socket pupil offset.

    Convention: +x right, +y down (image axes); +z out of face toward camera.
    Centered pupil → (0, 0, 1).
    """
    ex = 0.5 * (outer_xy[0] + inner_xy[0])
    ey = 0.5 * (outer_xy[1] + inner_xy[1])
    eye_w = math.hypot(outer_xy[0] - inner_xy[0], outer_xy[1] - inner_xy[1])
    R = max(eye_w * radius_factor, 1.0)
    dx = (pupil_xy[0] - ex) / R
    dy = (pupil_xy[1] - ey) / R
    mag = math.hypot(dx, dy)
    if mag > 0.99:
        s = 0.99 / mag
        dx *= s
        dy *= s
    dz = math.sqrt(max(0.0, 1.0 - dx * dx - dy * dy))
    return np.array([dx, dy, dz], dtype=np.float32)


# ── Crop centred on pupil ────────────────────────────────────────────────────

def crop_around(frame, cx, cy):
    """Return (crop, origin_x, origin_y) where origin is the crop's top-left in
    original-frame coords (may be negative; the crop reflects past the edge)."""
    h, w = frame.shape[:2]
    x0 = int(round(cx - CROP_W / 2))
    y0 = int(round(cy - CROP_H / 2))
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x0 + CROP_W - w)
    pad_b = max(0, y0 + CROP_H - h)
    if pad_l or pad_r or pad_t or pad_b:
        frame = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        xi, yi = x0 + pad_l, y0 + pad_t
    else:
        xi, yi = x0, y0
    return frame[yi:yi + CROP_H, xi:xi + CROP_W], x0, y0


# ── GT construction ──────────────────────────────────────────────────────────

def build_gt(lms_xy, gray, color, side, eye_state):
    """Returns (gt_dict, crop_origin_xy, pupil_px_in_frame) or None."""
    spec = EYE_SPECS[side]
    iris_b_pts = lms_xy[spec['iris_b']]

    # Pupil centre: stick to MediaPipe iris (robust).
    cx, cy, iris_diam, _, _ = iris_fallback(lms_xy[spec['iris_c']], iris_b_pts)

    # Pupil diameter: dark-contour fitEllipse on the colour frame (V-channel)
    # so the pupil separates cleanly from a pigmented iris.  Falls back to
    # iris diameter if no plausible dark blob is found.
    ell = detect_pupil(color, iris_b_pts)
    if ell is not None:
        _, _, a, b, _ = ell
        diameter_px = math.sqrt(a * b)
    else:
        diameter_px = iris_diam

    crop, ox, oy = crop_around(gray, cx, cy)

    upper = lms_xy[spec['upper']]
    lower = lms_xy[spec['lower']]
    pupil = np.array([[cx, cy]], dtype=np.float32)

    lm = np.concatenate([upper, lower, pupil], axis=0).astype(np.float32)  # (17, 2)
    lm[:, 0] = (lm[:, 0] - ox) / CROP_W
    lm[:, 1] = (lm[:, 1] - oy) / CROP_H

    gv = gaze_vector((cx, cy), lms_xy[spec['outer']], lms_xy[spec['inner']])

    gt = {
        'landmarks':      lm.tolist(),
        'gaze_vector':    gv.tolist(),
        'pupil_diameter': float(diameter_px / MAX_PUPIL_DIAMETER),
        'eye_state':      int(eye_state),
        'pupil_valid':    True,
        'lid_valid':      True,
    }
    return gt, crop, (cx, cy)


# ── Calibration overlay ──────────────────────────────────────────────────────

class Calibration:
    PHASES       = [ES_FIXATION, ES_SACCADE, ES_PURSUIT]
    PHASE_NAMES  = {ES_FIXATION: 'FIXATION', ES_SACCADE: 'SACCADE', ES_PURSUIT: 'SMOOTH PURSUIT'}
    TRAJECTORIES = ['sine', 'parabola', 'spiral']

    def __init__(self, w, h):
        self.W, self.H = w, h
        self.phase_idx = 0
        self.count = 0
        self.pos_count = 0
        self.trail = []
        self.elapsed = 0.0          # logical time within current position
        self._last_tick = time.time()
        self._reposition()

    def tick(self, paused):
        """Advance logical clock only when not paused, so pursuit freezes with Q off."""
        now = time.time()
        if not paused:
            self.elapsed += now - self._last_tick
        self._last_tick = now

    @property
    def state(self):
        return self.PHASES[self.phase_idx]

    @property
    def phase_name(self):
        return self.PHASE_NAMES[self.state]

    def done(self):
        return self.phase_idx >= len(self.PHASES) - 1 and self.count >= FRAMES_PER_PHASE

    def advance(self):
        self.count += 1
        self.pos_count += 1
        if self.pos_count >= FRAMES_PER_POSITION:
            self._reposition()
        if self.count >= FRAMES_PER_PHASE and self.phase_idx + 1 < len(self.PHASES):
            self.phase_idx += 1
            self.count = 0
            self._reposition()

    def _reposition(self):
        self.pos_count = 0
        self.trail = []
        self.elapsed = 0.0
        if self.state == ES_FIXATION:
            gx, gy = random.randrange(12), random.randrange(12)
            sx, sy = int(self.W * 0.2), int(self.H * 0.2)
            cw, ch = int(self.W * 0.6), int(self.H * 0.6)
            self.point = (int(sx + (gx + 0.5) * cw / 12),
                          int(sy + (gy + 0.5) * ch / 12))
        elif self.state == ES_SACCADE:
            m = 60
            self.point_a = (random.randint(m, self.W - m), random.randint(m, self.H - m))
            self.point_b = (random.randint(m, self.W - m), random.randint(m, self.H - m))
        else:
            self.trajectory = random.choice(self.TRAJECTORIES)

    def pursuit_point(self):
        t = self.elapsed
        if self.trajectory == 'sine':
            u = _pingpong(t * 0.15)            # 0→1→0 sweep
            x = self.W * 0.1 + u * self.W * 0.8
            y = self.H * 0.5 + math.sin(u * 4 * math.pi) * self.H * 0.25
        elif self.trajectory == 'parabola':
            nx = _pingpong(t * 0.12) * 2 - 1   # -1→+1→-1
            x = self.W * 0.5 + nx * self.W * 0.4
            y = self.H * 0.3 + (nx * nx) * self.H * 0.5
        else:  # spiral, outside-in then restart
            cycle = (t * 0.1) % 1.0
            r = min(self.W, self.H) * 0.4 * (1.0 - cycle)
            theta = cycle * 6 * math.pi
            x = self.W * 0.5 + r * math.cos(theta)
            y = self.H * 0.5 + r * math.sin(theta)
        return (int(x), int(y))

    def draw(self, img):
        st = self.state
        if st == ES_FIXATION:
            cv2.circle(img, self.point, 10, (0, 255, 0), -1)
            cv2.circle(img, self.point, 16, (0, 255, 0), 1)
        elif st == ES_SACCADE:
            cv2.circle(img, self.point_a, 10, (0, 165, 255), -1)
            cv2.circle(img, self.point_b, 10, (0, 165, 255), -1)
            cv2.line(img, self.point_a, self.point_b, (0, 100, 200), 1, cv2.LINE_AA)
        else:
            pt = self.pursuit_point()
            self.trail.append(pt)
            if len(self.trail) > 400:
                self.trail = self.trail[-400:]
            for i in range(1, len(self.trail)):
                cv2.line(img, self.trail[i - 1], self.trail[i],
                         (60, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(img, pt, 12, (0, 255, 255), -1)


# ── Q toggle listener ────────────────────────────────────────────────────────

class QToggle:
    """Press Q at any point during the run to flip logging on; press again to pause."""

    def __init__(self):
        self.on = False
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        if getattr(key, 'char', None) in ('q', 'Q'):
            self.on = not self.on


# ── Main loop ────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--cam-w',  type=int, default=1280)
    ap.add_argument('--cam-h',  type=int, default=720)
    ap.add_argument('--out',    type=str,
                    default=str(Path(__file__).parent / 'captures'))
    ap.add_argument('--status-scale', type=float, default=0.6,
                    help='font scale for the status line (try 2.0+ to read glasses-off)')
    return ap.parse_args()


def landmark_xy(landmarks, w, h):
    """MediaPipe normalized landmarks → (N, 2) array in pixel coords."""
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)


def _ensure_model():
    if MODEL_PATH.exists():
        return
    print(f'downloading face_landmarker.task → {MODEL_PATH}')
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def _make_detector():
    _ensure_model()
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def main():
    args = parse_args()
    run_dir = Path(args.out).expanduser() / time.strftime('%Y%m%d_%H%M%S')
    for side in EYE_SPECS:
        (run_dir / side).mkdir(parents=True, exist_ok=True)
    counters = {side: 0 for side in EYE_SPECS}

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_h)
    ok, probe = cap.read()
    if not ok:
        raise SystemExit('cannot read from webcam')
    H, W = probe.shape[:2]

    detector = _make_detector()
    t_start  = time.time()

    calib = Calibration(W, H)
    q = QToggle()

    win = 'capture'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t_start) * 1000)
            res = detector.detect_for_video(mp_image, ts_ms)

            calib.tick(paused=not q.on)

            captures = {}
            if res.face_landmarks:
                face_lms = res.face_landmarks[0]
                lms_xy = landmark_xy(face_lms, W, H)
                for side in EYE_SPECS:
                    gt, crop, pupil_px = build_gt(lms_xy, gray, frame, side, calib.state)
                    captures[side] = (gt, crop, pupil_px)
            else:
                face_lms = None

            overlay = frame.copy()
            calib.draw(overlay)
            for side, (gt, _, pupil_px) in captures.items():
                cv2.circle(overlay, (int(pupil_px[0]), int(pupil_px[1])),
                           3, (0, 255, 255), -1)
                spec = EYE_SPECS[side]
                for idx in spec['upper'] + spec['lower']:
                    p = face_lms[idx]
                    cv2.circle(overlay, (int(p.x * W), int(p.y * H)),
                               1, (255, 80, 80), -1)
            status = (f'{calib.phase_name}  '
                      f'phase {calib.count}/{FRAMES_PER_PHASE}  '
                      f'pos {calib.pos_count}/{FRAMES_PER_POSITION}  '
                      f'Q={"ON" if q.on else "OFF"}')
            scale = args.status_scale
            thick = max(1, int(round(scale * 1.5)))
            (_, text_h), baseline = cv2.getTextSize(
                status, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            y = H - baseline - 6
            cv2.rectangle(overlay, (0, y - text_h - 6), (W, H), (0, 0, 0), -1)
            cv2.putText(overlay, status, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick,
                        cv2.LINE_AA)
            cv2.imshow(win, overlay)

            if q.on and len(captures) == 2:
                for side, (gt, crop, _) in captures.items():
                    n = counters[side]
                    stem = f'{n:06d}'
                    cv2.imwrite(str(run_dir / side / f'{stem}.png'), crop)
                    with open(run_dir / side / f'{stem}.json', 'w') as f:
                        json.dump(gt, f)
                    counters[side] += 1
                calib.advance()

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or calib.done():
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

    print(f'wrote {counters} to {run_dir}')


if __name__ == '__main__':
    main()
