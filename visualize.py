import argparse
import os, glob, random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from model import ShinraCNN
from dataset import SyntheticDS

device = torch.device('cpu')

GAZE_ARROW_LEN = 90       # display length of the gaze arrow in pixels
GAZE_FLIP_Y = True        # UnityEyes look_vec has +y up, image +y is down
WEBCAM_CROP = 256         # side length of the center crop fed to the model in webcam mode

# Synthetic image dimensions: the model's pixel-space outputs (eyelid points,
# pupil diameter) reference this frame regardless of input resize.
TRAIN_W, TRAIN_H = 640, 480

# Standard pipeline: 3-ch RGB in → grayscale → 224×224 → normalized.
synth_inference_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])

# Webcam pipeline: input is already a 1-channel grayscale crop, just resize+normalize.
gray_inference_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.5], [0.5]),
])


def find_latest_checkpoint():
    phase_dirs = sorted(
        [d for d in os.listdir('.') if d.startswith('phase_') and os.path.isdir(d)],
        key=lambda d: int(d.split('_')[1]), reverse=True,
    )
    for phase_dir in phase_dirs:
        ckpts = sorted(
            glob.glob(os.path.join(phase_dir, 'shinra_checkpoint_*.pth')),
            key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]),
        )
        if ckpts:
            return ckpts[-1]
    return None


def smooth_curve(points, degree=2):
    # Least-squares polynomial fit — does not overshoot between samples.
    # Eyelids are roughly parabolic, so degree=2 fits cleanly without ringing.
    pts = np.asarray(points, dtype=np.float64)
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    xs, ys = pts[:, 0], pts[:, 1]
    if len(xs) < 2:
        return pts
    deg = min(degree, len(xs) - 1)
    coeffs = np.polyfit(xs, ys, deg)
    xx = np.linspace(xs.min(), xs.max(), 120)
    return np.column_stack([xx, np.polyval(coeffs, xx)])


def load_model():
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError('No phase_*/shinra_checkpoint_*.pth checkpoint found.')
    print(f'Loading checkpoint: {ckpt_path}')
    shinra = ShinraCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    shinra.load_state_dict(ckpt['shinra'])
    shinra.eval()
    return shinra


def infer(shinra, img_tensor, already_gray=False):
    """img_tensor: uint8 [C, H, W]. Returns (pupil[3], eyelid[8,2], gaze[3])."""
    transform = gray_inference_transforms if already_gray else synth_inference_transforms
    x = transform(img_tensor).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pupil_pred, eyelid_pred, gaze_pred = shinra(x)
    pupil = pupil_pred[0].cpu().numpy()[0]
    eyelid = eyelid_pred[0].cpu().numpy()[0].reshape(8, 2)
    gaze = gaze_pred[0].cpu().numpy()[0]
    return pupil, eyelid, gaze


def synthetic_mode(shinra):
    ds = SyntheticDS(transforms=None)
    print(f'Synthetic mode. Dataset size: {len(ds)}. Close window for next sample.')

    while True:
        idx = random.randrange(len(ds))
        raw_img, gt = ds[idx]
        img_np = raw_img.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:
            img_np = img_np[..., 0]
        H_img, W_img = img_np.shape[:2]

        pupil, eyelid_pts, gaze = infer(shinra, raw_img)

        gt_pupil = gt['pupil'].numpy()
        gt_eyelid = gt['eyelid_shape'].numpy()
        gt_gaze = gt['gaze_vector'].numpy()
        mse_pupil = float(np.mean((pupil - gt_pupil) ** 2))
        mse_eyelid = float(np.mean((eyelid_pts.flatten() - gt_eyelid) ** 2))
        mse_gaze = float(np.mean((gaze - gt_gaze) ** 2))

        pupil_x = float(pupil[0]) * W_img
        pupil_y = float(pupil[1]) * H_img
        diameter = float(pupil[2])
        upper, lower = eyelid_pts[:4], eyelid_pts[4:]

        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)

        ax.scatter(eyelid_pts[:, 0], eyelid_pts[:, 1],
                   c='black', s=35, zorder=4, edgecolors='white', linewidths=0.5)
        for lid_pts in (upper, lower):
            curve = smooth_curve(lid_pts)
            ax.plot(curve[:, 0], curve[:, 1], color='black', linewidth=1.6, zorder=3)

        ax.plot([pupil_x - diameter / 2, pupil_x + diameter / 2],
                [pupil_y, pupil_y], color='white', linewidth=2.5, zorder=5)

        ax.scatter([pupil_x], [pupil_y], c='blue', s=45, zorder=6,
                   edgecolors='white', linewidths=0.5)
        ax.annotate(f'({pupil_x:.0f}, {pupil_y:.0f})',
                    (pupil_x, pupil_y), xytext=(10, -10), textcoords='offset points',
                    color='blue', fontsize=10, fontweight='bold')

        gx, gy = gaze[0], (-gaze[1] if GAZE_FLIP_Y else gaze[1])
        ax.annotate('', xy=(pupil_x + gx * GAZE_ARROW_LEN, pupil_y + gy * GAZE_ARROW_LEN),
                    xytext=(pupil_x, pupil_y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.2, mutation_scale=22))

        ax.set_xlim(0, W_img); ax.set_ylim(H_img, 0)
        ax.set_title(f'idx={idx}   (close window for next sample)')
        ax.axis('off')

        mse_text = (
            'Per-head MSE (pred vs ground truth)\n'
            f'  pupil   {mse_pupil:.4e}\n'
            f'  eyelid  {mse_eyelid:.4e}\n'
            f'  gaze    {mse_gaze:.4e}'
        )
        ax.text(0.012, 0.985, mse_text,
                transform=ax.transAxes, va='top', ha='left',
                fontsize=10, family='monospace', color='white',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=6))
        plt.tight_layout()
        plt.show()


def draw_overlays_cv2(frame, pupil, eyelid_pts, gaze, origin_x, origin_y, crop_size):
    # Map model outputs (trained in TRAIN_W × TRAIN_H pixel space) into the crop region.
    sx = crop_size / TRAIN_W
    sy = crop_size / TRAIN_H

    px = origin_x + float(pupil[0]) * crop_size
    py = origin_y + float(pupil[1]) * crop_size
    diameter = float(pupil[2]) * sx

    pts = np.empty((8, 2), dtype=np.float32)
    pts[:, 0] = origin_x + eyelid_pts[:, 0] * sx
    pts[:, 1] = origin_y + eyelid_pts[:, 1] * sy

    for lid_pts in (pts[:4], pts[4:]):
        curve = smooth_curve(lid_pts).astype(np.int32)
        cv2.polylines(frame, [curve], isClosed=False, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 0), -1, cv2.LINE_AA)

    cv2.line(frame,
             (int(px - diameter / 2), int(py)),
             (int(px + diameter / 2), int(py)),
             (255, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(frame, (int(px), int(py)), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(px), int(py)), 4, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(frame, f'({int(px)}, {int(py)})',
                (int(px) + 8, int(py) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    gx, gy = gaze[0], (-gaze[1] if GAZE_FLIP_Y else gaze[1])
    cv2.arrowedLine(frame, (int(px), int(py)),
                    (int(px + gx * GAZE_ARROW_LEN), int(py + gy * GAZE_ARROW_LEN)),
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)


def webcam_mode(shinra, camera_idx):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open camera index {camera_idx}')
    print(f'Webcam mode (camera {camera_idx}). Press q to quit.')

    while True:
        ok, frame = cap.read()
        if not ok:
            print('Failed to read frame from camera.')
            break
        H, W = frame.shape[:2]
        if W < WEBCAM_CROP or H < WEBCAM_CROP:
            print(f'Camera resolution {W}×{H} smaller than required {WEBCAM_CROP}×{WEBCAM_CROP} crop.')
            break

        x0 = (W - WEBCAM_CROP) // 2
        y0 = (H - WEBCAM_CROP) // 2

        crop_bgr = frame[y0:y0 + WEBCAM_CROP, x0:x0 + WEBCAM_CROP]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Render the grayscale crop back into the displayed frame so the user
        # sees exactly the region the model is processing, in the colorspace it sees it.
        frame[y0:y0 + WEBCAM_CROP, x0:x0 + WEBCAM_CROP] = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)

        gray_tensor = torch.from_numpy(crop_gray).unsqueeze(0)  # [1, H, W] uint8
        pupil, eyelid_pts, gaze = infer(shinra, gray_tensor, already_gray=True)

        draw_overlays_cv2(frame, pupil, eyelid_pts, gaze,
                          origin_x=x0, origin_y=y0, crop_size=WEBCAM_CROP)

        cv2.rectangle(frame, (x0, y0), (x0 + WEBCAM_CROP, y0 + WEBCAM_CROP),
                      (0, 255, 255), 1)
        cv2.putText(frame, 'q: quit', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Shinra-Meisin live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize Shinra-Meisin eye-tracking predictions.')
    parser.add_argument('--webcam', action='store_true',
                        help='Run on a live webcam feed (256×256 center crop) instead of synthetic samples.')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index for --webcam mode (default 0).')
    args = parser.parse_args()

    shinra = load_model()
    if args.webcam:
        webcam_mode(shinra, args.camera)
    else:
        synthetic_mode(shinra)


if __name__ == '__main__':
    main()
