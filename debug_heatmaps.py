#!/usr/bin/env python3
"""
debug_heatmaps.py — navigate Shinra-Meisin heatmap tensors at full resolution.

Panels per sample:
  [Overview group]  source image, pred overlay (max across channels), GT overlay
  [Channels group]  pupil overlaid on each of the 16 landmark channels, then
                    standalone heatmaps for those 16 channels

Controls:
  ← / →  or scroll wheel   previous / next panel
  Tab                       jump to the start of the next group
  q                         close window (advance to next sample if --n > 1)

Usage:
    python debug_heatmaps.py              # one random sample
    python debug_heatmaps.py --idx 42     # specific dataset index
    python debug_heatmaps.py --n 4        # cycle through N random samples
    python debug_heatmaps.py --no-model   # GT heatmaps only (no checkpoint needed)
"""

import argparse
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from model import ShinraCNN
from dataset import SyntheticDS
from visualize import find_latest_checkpoint, decode_heatmaps, synth_inference_transforms, MAX_IRIS_DIAMETER

device = torch.device('cpu')

CHANNEL_NAMES = (
    ['pupil']
    + [f'upper_{i}' for i in range(8)]
    + [f'lower_{i}' for i in range(8)]
)

PRED_COLOR = 'cyan'
GT_COLOR   = 'lime'
DISP       = 224   # display size for overlay panels


# ── model ──────────────────────────────────────────────────────────────────────

def load_model():
    ckpt_path = find_latest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError('No phase_*/shinra_checkpoint_*.pth found.')
    print(f'Checkpoint: {ckpt_path}')
    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    shinra = ShinraCNN(backbone, out_channels=17).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    shinra.load_state_dict(ckpt['shinra'])
    shinra.eval()
    return shinra


def run_inference(shinra, raw_img):
    x = synth_inference_transforms(raw_img).unsqueeze(0).to(device)
    with torch.no_grad():
        (diam_pred, _), hm, _, dec_feats = shinra(x, return_decoder_feats=True)
    diameter = diam_pred[0].squeeze().item()
    decoder_feats = [f[0].numpy() for f in dec_feats]  # list of (C, H, W)
    return torch.sigmoid(hm[0]).cpu().numpy(), diameter, decoder_feats  # (17, H, W), scalar, feats


# ── panels ──────────────────────────────────────────────────────────────────────

def _info_text(ax, text, loc='bottom'):
    """Readable monospace annotation in a corner of an axes."""
    x  = 0.01
    y  = 0.01 if loc == 'bottom' else 0.99
    va = 'bottom' if loc == 'bottom' else 'top'
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9,
            family='monospace', color='white', va=va, ha='left',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=4))


def _draw_source(ax, img_np, pred_center=None, pred_diam_px=None, gt_center=None, gt_diam_px=None):
    ax.imshow(img_np, cmap='gray')
    h, w = img_np.shape[:2]
    _info_text(ax, f'{w}×{h}')

    if gt_center is not None and gt_diam_px is not None:
        gx, gy = gt_center
        half = gt_diam_px / 2
        ax.plot([gx - half, gx + half], [gy - 4, gy - 4],
                color=GT_COLOR, linewidth=2.0, zorder=5)
        ax.scatter([gx], [gy], c=GT_COLOR, s=40, zorder=6, label='GT pupil')

    if pred_center is not None and pred_diam_px is not None:
        px, py = pred_center
        half = pred_diam_px / 2
        ax.plot([px - half, px + half], [py, py],
                color=PRED_COLOR, linewidth=2.0, zorder=5)
        ax.scatter([px], [py], c=PRED_COLOR, s=40, zorder=6, label='pred pupil')

    if pred_center is not None or gt_center is not None:
        ax.legend(fontsize=8, loc='upper right', framealpha=0.6)

    ax.axis('off')


def _draw_overlay(ax, img_np, max_hm, dots, note):
    """dots: list of (xs, ys, color, marker, label)"""
    ax.imshow(img_np, cmap='gray',  extent=[0, DISP, DISP, 0])
    ax.imshow(max_hm, cmap='hot', alpha=0.6, vmin=0, vmax=1,
              extent=[0, DISP, DISP, 0], interpolation='bilinear')
    for (xs, ys, color, marker, label) in dots:
        ax.scatter(xs, ys, c=color, s=40, marker=marker,
                   zorder=5, linewidths=0.9, label=label)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.6)
    ax.set_xlim(0, DISP)
    ax.set_ylim(DISP, 0)
    _info_text(ax, note)
    ax.axis('off')


def _resize_hm(hm, target_h, target_w):
    t = torch.from_numpy(hm).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()


def _draw_heatmap(ax, hm, gt_hm=None):
    H, W   = hm.shape
    py, px = np.unravel_index(hm.argmax(), hm.shape)
    im = ax.imshow(hm, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    half = hm.max() * 0.5
    if half > 0.01:
        ax.contour(hm, levels=[half], colors=['white'], linewidths=0.9, alpha=0.7, zorder=4)
    ax.plot(px, py, '+', color='white', markersize=12, markeredgewidth=1.5, zorder=5,
            label='pred peak')

    if gt_hm is not None:
        gt_H, gt_W = gt_hm.shape
        gt_py, gt_px = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
        gt_px_s = gt_px * W / gt_W
        gt_py_s = gt_py * H / gt_H
        gt_half = gt_hm.max() * 0.5
        if gt_half > 0.01:
            gt_overlay = gt_hm if (gt_H == H and gt_W == W) else _resize_hm(gt_hm, H, W)
            ax.contour(gt_overlay, levels=[gt_half], colors=['palegreen'],
                       linewidths=0.9, alpha=0.8, zorder=4)
        ax.plot(gt_px_s, gt_py_s, 'x', color='palegreen', markersize=10,
                markeredgewidth=1.5, zorder=5, label='GT peak')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.6)

    _info_text(ax, f'peak ({px},{py})   max {hm.max():.4f}   mean {hm.mean():.5f}   [{W}×{H}]')
    ax.axis('off')


def _draw_channel(ax, hm_ch, gt_peak):
    H, W = hm_ch.shape
    flat   = hm_ch.argmax()
    py, px = np.unravel_index(flat, hm_ch.shape)

    im = ax.imshow(hm_ch, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # half-max contour — shows the shape and spread of the active region
    half = hm_ch.max() * 0.5
    if half > 0.01:
        ax.contour(hm_ch, levels=[half], colors=[PRED_COLOR],
                   linewidths=0.9, alpha=0.8, zorder=4)

    ax.plot(px, py, '+', color=PRED_COLOR, markersize=12,
            markeredgewidth=1.5, zorder=5, label='pred peak')

    if gt_peak is not None:
        gx, gy = gt_peak
        ax.plot(gx, gy, 'x', color=GT_COLOR, markersize=10,
                markeredgewidth=1.5, zorder=5, label='GT peak')
        gt_str = f'   gt ({gx:.1f},{gy:.1f})'
    else:
        gt_str = ''

    ax.legend(fontsize=8, loc='upper right', framealpha=0.6)
    _info_text(ax,
        f'pred ({px},{py}){gt_str}   max {hm_ch.max():.4f}   mean {hm_ch.mean():.5f}   [{W}×{H}]')
    ax.axis('off')


def _draw_channel_with_pupil(ax, hm_ch, pupil_hm, gt_peak_ch, gt_peak_pupil):
    """Render hm_ch (hot) with pupil_hm (cool) blended on top."""
    H, W = hm_ch.shape
    py, px = np.unravel_index(hm_ch.argmax(), hm_ch.shape)

    im = ax.imshow(hm_ch, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pupil_rs = pupil_hm if pupil_hm.shape == hm_ch.shape else _resize_hm(pupil_hm, H, W)
    ax.imshow(pupil_rs, cmap='hot', vmin=0, vmax=1, alpha=0.45, interpolation='nearest')

    half = hm_ch.max() * 0.5
    if half > 0.01:
        ax.contour(hm_ch, levels=[half], colors=[PRED_COLOR],
                   linewidths=0.9, alpha=0.8, zorder=4)

    pupil_half = pupil_rs.max() * 0.5
    if pupil_half > 0.01:
        ax.contour(pupil_rs, levels=[pupil_half], colors=['magenta'],
                   linewidths=0.9, alpha=0.8, zorder=4)

    ax.plot(px, py, '+', color=PRED_COLOR, markersize=12,
            markeredgewidth=1.5, zorder=5, label='ch pred peak')

    pup_py, pup_px = np.unravel_index(pupil_rs.argmax(), pupil_rs.shape)
    ax.plot(pup_px, pup_py, 'D', color='magenta', markersize=7,
            markeredgewidth=1.5, zorder=5, label='pupil pred peak')

    if gt_peak_ch is not None:
        gx, gy = gt_peak_ch
        ax.plot(gx, gy, 'x', color=GT_COLOR, markersize=10,
                markeredgewidth=1.5, zorder=5, label='ch GT peak')
    if gt_peak_pupil is not None:
        gx, gy = gt_peak_pupil
        ax.plot(gx, gy, 's', color='magenta', markersize=8,
                markeredgewidth=1.5, zorder=5, fillstyle='none', label='pupil GT peak')

    ax.legend(fontsize=8, loc='upper right', framealpha=0.6)
    _info_text(ax,
        f'pred ({px},{py})   max {hm_ch.max():.4f}   mean {hm_ch.mean():.5f}   [{W}×{H}]')
    ax.axis('off')


def _draw_decoder_grid(ax, dec_feats):
    """Grid of all decoder block outputs (max-projection across channels) as a mosaic."""
    n    = len(dec_feats)
    cols = 2
    rows = (n + cols - 1) // cols
    cell = 72
    sep  = 3
    canvas = np.zeros((rows * cell + (rows - 1) * sep,
                       cols * cell + (cols - 1) * sep), dtype=np.float32)

    for i, feat in enumerate(dec_feats):
        r, c = divmod(i, cols)
        y0   = r * (cell + sep)
        x0   = c * (cell + sep)
        proj = feat.max(axis=0)               # (H, W)
        mn, mx = proj.min(), proj.max()
        if mx > mn:
            proj = (proj - mn) / (mx - mn)
        canvas[y0:y0 + cell, x0:x0 + cell] = _resize_hm(proj, cell, cell)

    ax.imshow(canvas, cmap='hot', vmin=0, vmax=1, interpolation='nearest', aspect='equal')

    for i, feat in enumerate(dec_feats):
        r, c = divmod(i, cols)
        y0   = r * (cell + sep)
        x0   = c * (cell + sep)
        C, H, W = feat.shape
        ax.text(x0 + cell / 2, y0 + 3, f'blk{i}  {C}ch  {W}×{H}',
                color='white', fontsize=7, ha='center', va='top', family='monospace',
                bbox=dict(facecolor='black', alpha=0.55, edgecolor='none', pad=1))

    ax.set_title('Decoder intermediaries  (max-proj per block)', fontsize=9, color='#dddddd', pad=4)
    ax.axis('off')


def build_overview_panels(img_np, hm_pred, hm_gt, lm_pred, lm_gt, have_pred, diam_pred=None, diam_gt=None):
    """Source image + max-activation overlay (pred heatmap with GT region annotated)."""
    panels = []

    h, w = img_np.shape[:2]

    gt_center = gt_diam_px = pred_center = pred_diam_px = None
    if lm_gt is not None:
        gx = lm_gt[0, 0] * w / 640
        gy = lm_gt[0, 1] * h / 480
        gt_center = (gx, gy)
        if diam_gt is not None:
            gt_diam_px = diam_gt * MAX_IRIS_DIAMETER * w / 640
    if have_pred and lm_pred is not None:
        px = lm_pred[0, 0] * w / 640
        py = lm_pred[0, 1] * h / 480
        pred_center = (px, py)
        if diam_pred is not None:
            pred_diam_px = diam_pred * MAX_IRIS_DIAMETER * w / 640

    panels.append({
        'title': 'Source image',
        'draw':  lambda ax, _img=img_np, _pc=pred_center, _pdp=pred_diam_px, _gc=gt_center, _gdp=gt_diam_px:
                     _draw_source(ax, _img, _pc, _pdp, _gc, _gdp),
    })

    max_gt = hm_gt.max(axis=0)
    if have_pred:
        max_pred = hm_pred.max(axis=0)
        panels.append({
            'title': 'Pred  —  max across 17 ch  +  GT region (green)',
            'draw':  lambda ax, _mp=max_pred, _mg=max_gt: _draw_heatmap(ax, _mp, _mg),
        })
    else:
        panels.append({
            'title': 'GT  —  max across 17 ch',
            'draw':  lambda ax, _mg=max_gt: _draw_heatmap(ax, _mg),
        })

    return panels


def build_channel_panels(hm_pred, hm_gt, have_pred, dec_feats=None):
    """Per-channel pred heatmaps; pupil channel shown overlaid on each other channel."""
    if not have_pred:
        return []

    _, H_pred, W_pred = hm_pred.shape
    _, H_gt,   W_gt   = hm_gt.shape

    # GT hard-argmax per channel, scaled into pred heatmap pixel space
    gt_peaks = []
    for c in range(17):
        gy, gx = np.unravel_index(hm_gt[c].argmax(), hm_gt[c].shape)
        gt_peaks.append((gx * W_pred / W_gt, gy * H_pred / H_gt))

    dec_draw = (lambda ax, _df=dec_feats: _draw_decoder_grid(ax, _df)) if dec_feats else None

    # Pupil (ch 0) overlaid on each of the other 16 channels
    pupil_panels = [
        {
            'title':        f'Pred  ch {c:02d}  {CHANNEL_NAMES[c]}  +  pupil overlay',
            'draw':         lambda ax, _c=c: _draw_channel_with_pupil(
                                ax, hm_pred[_c], hm_pred[0], gt_peaks[_c], gt_peaks[0]
                            ),
            'draw_decoder': dec_draw,
        }
        for c in range(1, 17)
    ]

    # Standalone panels for channels 1-16 (pupil channel omitted as standalone)
    standalone_panels = [
        {
            'title':        f'Pred  ch {c:02d}  {CHANNEL_NAMES[c]}',
            'draw':         lambda ax, _c=c: _draw_channel(ax, hm_pred[_c], gt_peaks[_c]),
            'draw_decoder': dec_draw,
        }
        for c in range(1, 17)
    ]

    return pupil_panels + standalone_panels


# ── navigator ──────────────────────────────────────────────────────────────────

class Navigator:
    def __init__(self, overview_panels, channel_panels, sample_label):
        # Groups stored separately so Tab can jump between them.
        # Filter out empty groups (e.g. channels when --no-model is set).
        self._groups = [g for g in (overview_panels, channel_panels) if g]
        self._group  = 0   # which group is active
        self.idx     = 0   # panel index within the active group
        self._sample_label = sample_label

        self.fig = plt.figure(figsize=(14, 7))
        self.fig.patch.set_facecolor('#111111')
        self._scroll_dir    = 0
        self._scroll_last   = 0.0
        self._scroll_active = False
        self._scroll_timer  = self.fig.canvas.new_timer(interval=110)
        self._scroll_timer.add_callback(self._scroll_tick)

        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)
        self.fig.canvas.mpl_connect('scroll_event',       self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_button)
        self._render()
        plt.show()

    @property
    def _panels(self):
        return self._groups[self._group]

    def _step(self, delta):
        self.idx = (self.idx + delta) % len(self._panels)
        self._render()

    def _jump_group(self):
        self._group = (self._group + 1) % len(self._groups)
        self.idx = 0
        self._render()

    def _trigger_scroll(self, direction):
        self._scroll_dir  = direction
        self._scroll_last = time.monotonic()
        if not self._scroll_active:
            self._scroll_active = True
            self._scroll_timer.start()

    def _scroll_tick(self):
        if time.monotonic() - self._scroll_last > 0.25:
            self._scroll_timer.stop()
            self._scroll_active = False
            return
        self._step(self._scroll_dir)

    def _on_key(self, event):
        if event.key in ('right', 'd'):
            self._step(+1)
        elif event.key in ('left', 'a'):
            self._step(-1)
        elif event.key == 'tab':
            self._jump_group()
        elif event.key == 'q':
            plt.close(self.fig)

    def _on_scroll(self, event):
        if event.button == 'up':
            self._trigger_scroll(-1)
        elif event.button == 'down':
            self._trigger_scroll(+1)

    def _on_button(self, event):
        # Wayland/XWayland delivers scroll as button 4 (up) / 5 (down)
        if event.button == 4:
            self._trigger_scroll(-1)
        elif event.button == 5:
            self._trigger_scroll(+1)

    def _render(self):
        # clf() wipes all axes cleanly, avoiding the size-steal that
        # plt.colorbar(ax=...) causes when axes are merely cleared.
        self.fig.clf()
        self.fig.patch.set_facecolor('#111111')

        group_label = 'Overview' if self._group == 0 else 'Channels'
        self.fig.text(0.5, 0.005,
                      f'[{group_label}]   ← → or scroll to navigate    Tab switch group    q close',
                      ha='center', va='bottom', fontsize=8, color='#666666')

        panel = self._panels[self.idx]

        if panel.get('draw_decoder'):
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(1, 2, figure=self.fig, wspace=0.06)
            self.ax = self.fig.add_subplot(gs[0])
            self.ax.set_facecolor('#111111')
            ax_dec = self.fig.add_subplot(gs[1])
            ax_dec.set_facecolor('#111111')
            panel['draw'](self.ax)
            panel['draw_decoder'](ax_dec)
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('#111111')
            panel['draw'](self.ax)

        self.ax.set_title(
            f'{self._sample_label}   [{self.idx + 1}/{len(self._panels)}]   {panel["title"]}',
            fontsize=9, color='#dddddd', pad=8,
        )
        self.fig.canvas.draw_idle()


# ── stats (terminal) ───────────────────────────────────────────────────────────

def print_stats(idx, hm_pred, hm_gt, have_pred):
    print(f'\n=== idx {idx} ===')
    if have_pred:
        print(f'  pred shape : {hm_pred.shape}')
    print(f'  GT   shape : {hm_gt.shape}')

    cols = f'  {"channel":<10}  '
    if have_pred:
        cols += f'{"pred_max":>8}  {"pred_mean":>9}  {"pred_peak":>10}  '
    cols += f'{"gt_max":>7}  {"gt_mean":>8}  {"gt_peak":>10}'
    print(cols)

    for c in range(17):
        row = f'  {CHANNEL_NAMES[c]:<10}  '
        if have_pred:
            ch = hm_pred[c]
            flat = ch.argmax()
            py, px = np.unravel_index(flat, ch.shape)
            row += f'{ch.max():>8.4f}  {ch.mean():>9.5f}  ({px:3d},{py:3d})    '
        ch = hm_gt[c]
        flat = ch.argmax()
        py, px = np.unravel_index(flat, ch.shape)
        row += f'{ch.max():>7.4f}  {ch.mean():>8.5f}  ({px:3d},{py:3d})'
        print(row)


# ── entry point ────────────────────────────────────────────────────────────────

def plot_sample(shinra, ds, idx):
    raw_img, gt = ds[idx]

    img_np = raw_img.permute(1, 2, 0).numpy()
    if img_np.ndim == 3 and img_np.shape[2] == 1:
        img_np = img_np[..., 0]

    hm_gt  = gt['eye_heatmaps'].numpy()
    lm_gt  = decode_heatmaps(gt['eye_heatmaps'], out_w=640, out_h=480).numpy()

    have_pred = shinra is not None
    dec_feats = None
    if have_pred:
        hm_pred, diam_pred, dec_feats = run_inference(shinra, raw_img)
        lm_pred = decode_heatmaps(torch.from_numpy(hm_pred), out_w=640, out_h=480).numpy()
    else:
        hm_pred = lm_pred = diam_pred = None

    diam_gt = float(gt['pupil_diameter'])

    print_stats(idx, hm_pred, hm_gt, have_pred)

    ph, pw = (hm_pred.shape[1], hm_pred.shape[2]) if have_pred else (0, 0)
    gh, gw = hm_gt.shape[1], hm_gt.shape[2]
    label  = (f'idx {idx}   pred {pw}×{ph}   gt {gw}×{gh}'
              if have_pred else f'idx {idx}   gt {gw}×{gh}')

    overview_panels = build_overview_panels(img_np, hm_pred, hm_gt, lm_pred, lm_gt, have_pred, diam_pred, diam_gt)
    channel_panels  = build_channel_panels(hm_pred, hm_gt, have_pred, dec_feats)

    Navigator(overview_panels, channel_panels, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',      type=int,  default=None)
    parser.add_argument('--n',        type=int,  default=1)
    parser.add_argument('--no-model', action='store_true')
    args = parser.parse_args()

    shinra = None if args.no_model else load_model()
    ds     = SyntheticDS(transforms=None)
    print(f'Dataset size: {len(ds)}')

    indices = (
        [args.idx] if args.idx is not None
        else [random.randrange(len(ds)) for _ in range(args.n)]
    )
    for idx in indices:
        plot_sample(shinra, ds, idx)


if __name__ == '__main__':
    main()
