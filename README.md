# Shinra-Meisin
### *神羅明眸 — "The All-Seeing Eminence Of Shadow"*

**Shinra-Meisin** is an open-source, extensible tracking pipeline developed by **Walker Industries R&D** in partnership with **EOZVR**. The architecture is built around a shared CNN backbone with swappable prediction heads, so the same training stack can in principle be retargeted to mouth, tongue, or full-body pose once matching datasets are available.

Today, in its current form, it is a **binocular eye tracker** for VR/AR headsets, designed to run in real time on embedded hardware. The eye-tracking build is the tracking core of the **VectorGear** XR headset and the model behind **SM-Tracker**, a plug-and-play snap-on add-on for popular consumer HMDs (Meta Quest 3/3S, BigScreen Beyond 2, Pimax Dream Air).

---

## What it does (eye-tracking build)

Every frame, Shinra-Meisin looks at a grayscale image of one eye (NIR for production, RGB→gray for webcam dev) and predicts three things:

- **Gaze direction** — a 3D unit vector in image-plane convention (+x right, +y down, +z toward the camera)
- **Pupil diameter** — a single scalar, normalised against a maximum-pupil prior so synthetic and real captures live on the same scale
- **17 eyelid + pupil landmarks** — 8 upper-lid points, 8 lower-lid points, and the pupil centre — predicted as a soft-argmax over a 17-channel heatmap for sub-pixel localisation

Every regression output is paired with a **learned log-variance**, so the model emits its own per-frame uncertainty rather than a flat confidence. Both eyes are inferred independently, at up to **90 fps** on a Rockchip RK3576 NPU once quantised. Tracking output is published over **OSC/UDP** so it drops straight into [VRCFaceTracking](https://github.com/benaclejames/VRCFaceTracking) and similar consumers.

---

## Why open source

Reliable eye tracking has been locked behind expensive proprietary hardware for too long. The model architecture, the training pipeline, and the synthetic-data tooling should be something the community can build on. The plug-and-play hardware product funds the research. The research belongs to everyone.

---

## How it works — the non-technical version

Shinra-Meisin learns to read eyes the way a student would: it starts with millions of synthetic examples where every label is perfect, then mixes in real captures, then mixes in unlabeled webcam imagery so it stops over-fitting to the renderer's quirks.

1. **Synthetic training.** Photorealistic eye renders from [UnityEyes2](https://github.com/alexanderdsmith/UnityEyes2) come with exact analytical labels — pupil position, gaze direction, eyelid shape — computed straight from the 3D scene. No human annotation required.
2. **Mixed-domain fine-tuning.** Real captures (recorded with `capture_dataset.py`) are mixed in at a phase-dependent ratio, with the synthetic fraction dropping from 1.00 in phase 0 to 0.30 in phase 3. Hard-flagged frames are over-sampled.
3. **Domain-adversarial alignment.** A small domain classifier with a gradient-reversal layer pushes the backbone to produce features that look the same whether the input is synthetic or a real unlabeled webcam frame. This kicks in in phase 2 and later, using the HybridGaze webcam set as the unlabeled side.
4. **Progressive thaw.** The backbone is unfrozen one segment at a time, from heads-only at phase 0 down to the shallowest layers at phase 3, with a separate (lower) learning rate per thawed segment.
5. **Embedded deployment.** The trained model is quantised to INT8 and deployed via RKNN onto the RK3576 NPU, fitting an 11 ms per-frame budget.

---

## Architecture — the technical version

Shinra-Meisin is a **multi-task CNN** built on a **YOLO26n backbone**, modified at the stem to accept a single-channel grayscale input. Rather than running the full detection model, only the first 11 backbone layers are kept and tapped at three depths:

| Tap | Source layer | Role |
|---|---|---|
| `early` | after layer 2 | high-resolution spatial detail for landmark localisation |
| `mid`   | after layer 4 | medium-resolution context, fused with `early` for the heatmap head |
| `deep`  | after layer 10 | low-resolution global context for gaze and pupil diameter |

Input is grayscale at **320×240** (raw 640×480 captures resized by `INIT_SCALE=0.5`). The three taps and the heads connect like this:

```
Input (240×320×1, grayscale / NIR)
        │
        ▼
   ┌───────────────────────────────────────────────┐
   │ YOLO26n backbone, layers 0–10 (modified stem) │
   │  - first Conv replaced with 1-channel variant │
   │  - BatchNorms frozen in eval until thawed     │
   └───────────────────────────────────────────────┘
        │            │                │
        │            │                │
        ▼            ▼                ▼
     early tap    mid tap          deep tap
     (64 ch,      (128 ch,         (256 ch,
      /4 stride)   /8 stride)       /32 stride)
        │            │                │
        │  upsample to early res      │
        └────┬───────┘                │
             │                        │
             ▼                        │
       fused features                 │
       (192 ch, early res)            │
             │                        │
             ▼                ┌───────┴───────┐
       HeatmapHead            │               │
       (2× DepthwiseSep)      ▼               ▼
             │             GazeHead       DiameterHead
             │           (GAP → MLP →    (GAP → MLP →
             ▼            3D vec + lvar)  scalar + lvar)
       17-ch logits
       at (H/4 × W/4)
             │
       softmax + soft-argmax
       over registered (x,y) grid
             │
             ▼
       17 sub-pixel (x, y)
       coordinates in [0, 1]

       (DomainClassifier, phase 2+: reads the fused features
        through a gradient-reversal layer and tries to tell
        synthetic from webcam. Backbone gets the reversed
        gradient and learns to make the two indistinguishable.)
```

The skip between `early` and `mid` is a single FPN-style fusion: `mid` is nearest-upsampled to `early`'s spatial size and concatenated channel-wise (`64 + 128 = 192`). The `deep` tap is left to gaze and diameter, which both want global context more than spatial precision.

### Heads

**HeatmapHead** is two depthwise-separable 3×3 convolutions (`192 → 96 → 17`) producing one logit map per landmark. A coordinate buffer is registered on the head itself, so soft-argmax is a single elementwise multiply + reduce:

```
logits (B, 17, H, W)
  │
  │   H, W = INPUT_H/4, INPUT_W/4 = 60, 80
  │
  ▼
flatten spatial → (B, 17, H·W)
  │
  ÷ temperature τ (=0.1)         ← sharpens the peak
  │
softmax over the spatial axis    ← (B, 17, H·W) sums to 1 per channel
  │
reshape → soft_map (B, 17, H, W)
  │
  ├──────── × x_grid (H, W ∈ [0,1])   sum over (H, W)  →  pred_x  (B, 17)
  └──────── × y_grid (H, W ∈ [0,1])   sum over (H, W)  →  pred_y  (B, 17)
                                        │
                                        ▼
                              landmark coords (B, 17, 2) ∈ [0, 1]²
```

At training time the same logits are also matched directly against a Gaussian heatmap rendered from the ground-truth coordinates (`σ = 2.5` pixels on the head grid). The heatmap head's total loss is a blend:

```
landmark_loss = 0.25 * focal_loss(logits, gaussian_gt)
              + 0.75 * smooth_L1(soft_argmax_pred, coords_gt)
```

The focal term shapes the distribution, the coord term pulls the centroid where it should go. Both terms benefit each other: the focal loss anchors the spatial layout, smooth-L1 anchors the actual answer.

**GazeHead** and **DiameterHead** are identical in shape (`AdaptiveAvgPool2d → 256 → 128 → out`, plus a parallel `128 → 1` log-variance head). Gaze outputs a 3-vector, diameter outputs a scalar. Both read from the `deep` tap (`/32`, 256 channels) since gaze and pupil scale are global properties of the eye, not anchored to a specific pixel.

### Uncertainty-aware loss

Every regression head emits a log-variance alongside its prediction, and that log-variance feeds the heteroscedastic loss:

$$\mathcal{L}_i = \exp(-\log\sigma_i^2)\,(y_i - \hat{y}_i)^2 + \log\sigma_i^2$$

When the model is uncertain, the squared-error term is down-weighted by the precision and the model pays a penalty proportional to its own admitted uncertainty. This is what stops it from being equally confident about a sharp, well-lit pupil and a blink frame, and it is what the downstream state machine and the ROS-side filter consume.

### Domain-adversarial branch

`dann.py` defines a small classifier (GAP → 64 → 1) sitting behind a `GradientReversalFunction`. During training, the classifier reads the same fused features the heatmap head reads — once for the labeled (synthetic + real-capture) batch, once for the unlabeled webcam batch — and is trained with binary cross-entropy to tell them apart. The gradient-reversal layer flips the sign of the gradient flowing back into the backbone, so the backbone is pushed in the opposite direction: toward features the classifier can't distinguish. The DANN's gradient-reversal `λ` ramps up over the phase with the standard `2/(1+e^{-10p}) - 1` schedule.

This branch is gated on phase: it activates from phase 2 onward, the same point at which the second backbone segment is unfrozen.

### Progressive training

The backbone is frozen at construction time. `ShinraCNN.thaw()` is a generator that yields one optimizer parameter-group set per phase, peeling layers off the freeze list one segment at a time and assigning each segment its own learning rate (deepest gets the highest, shallowest the lowest):

| Phase | Newly unfrozen | DANN active | Synthetic / real ratio |
|---|---|---|---|
| 0 | None — heads only | no | 1.00 / 0.00 |
| 1 | `deep` segment (layers 5–10) | no | ~0.77 / 0.23 |
| 2 | `mid` segment (layers 3–4) | yes | ~0.53 / 0.47 |
| 3 | `early` segment (layers 0–2) | yes | 0.30 / 0.70 |

Each phase runs up to 20 epochs with cosine annealing and early stopping at patience 5. Checkpoints are dropped into `phase_<N>/shinra_checkpoint_<epoch>.pth`, and `train.py` will resume from the latest checkpoint it can find.

---

## Datasets

The training stack consumes three sources:

- **Synthetic** (`SyntheticDataset`) — UnityEyes2 renders at 640×480 with full analytical labels: gaze vector, pupil pixel diameter, eyelid margin polylines, pupil 2D position. Lives in `~/Downloads/datasets/synthetic_v2/`.
- **Real captures** (`RealDataset`) — frames recorded via `capture_dataset.py`. Each session is one eye recording, stored as `captures/{run_ts}/{eye0|eye1}/{fi:06d}.{png,json}`. The JSON carries gaze, pupil diameter in raw pixels, the same 17-landmark layout, an eye-state code, and an optional `hard` flag for hard-mining. Sessions are split at the session level (not the frame level) by `session_split()` so train/val never share frames from the same recording.
- **Unlabeled webcam** (`WebcamDataset`) — drawn from the HybridGaze public set; used only by the DANN branch as the unlabeled "real" domain.

`SharedDataset` interleaves synthetic and real-capture frames at a phase-dependent ratio with hard-flagged frames sampled at `hard_weight=1.6×`. The mix is fixed for the entire phase and reshuffled at the start of every epoch. `EyeStateBatchSampler` keeps contiguous same-eye-state runs intact within a batch, which matters for any temporal smoothing on the consumer side.

---

## Project structure

```
shinra-meisin/
├── model.py              # ShinraCNN — backbone, taps, fusion, heads, DANN wiring, thaw schedule
├── train.py              # Phased training loop, heteroscedastic + focal + smooth-L1 losses, checkpointing
├── input.py              # Real / Synthetic / Webcam / Shared datasets, transforms, batch sampler, session split
├── dann.py               # GradientReversalFunction + DomainClassifier
├── early_stopping.py     # Patience-based early stop
├── capture_dataset.py    # Real-data capture tool (webcam → annotated frames)
├── hard_mine.py          # Mark hard frames for over-sampling
├── mark_hard.py          # Manual hard-flag toggler
├── verify_captures.py    # Sanity-check capture JSON + image pairs
├── extract_frames.py     # Frame extraction from raw video
├── random_crop_captures.py
├── transplant_checkpoint.py  # Move weights between backbone layouts
├── visualize.py          # Per-sample prediction overlay
├── test_shared_composition.py
├── yolo26n.pt            # YOLO26n backbone weights (pretrained)
└── requirements.txt
```

---

## Current status

The current build trains end-to-end on synthetic + real captures, with the DANN branch wired up and ready to engage in phase 2. Gaze is converging well; the heatmap head's coordinate term is doing most of the work, with focal acting as a regulariser on the spatial distribution.

**Immediate roadmap**
- ROI state machine + downstream Kalman filter
- RKNN INT8 quantisation and RK3576 deployment
- VRCFaceTracking C# module

**Later**
- Mouth / lip / tongue tracking — same backbone, new heads, new dataset
- In-house FOSS dataset captured across BSB2, Quest 3/3S, and Pimax Dream Air
- Full-body inside-out tracking for our WIP controllers
- BCI data layer with VRChat OSC + Resonite and an [Eclipse](https://github.com/Walker-Industries-RnD/Eclipse/tree/main/EclipseProject) layer

---

## Requirements

PyTorch 2.11, torchvision 0.26, Ultralytics 8.4.x (for the YOLO26n loader), OpenCV, NumPy, Pillow, scipy, polars. Full pinned set in `requirements.txt`.

```
torch==2.11.0
torchvision==0.26.0
ultralytics==8.4.47
opencv-python==4.13.0.92
numpy==2.4.4
```

---

## License

Copyright (c) 2026 Walker Industries R&D. All rights reserved.

This is a work-in-progress prototype. You may view the source for personal evaluation only. No license is granted (express or implied) for:

- copying
- modification
- distribution
- commercial use
- derivative works
- or any other form of exploitation

It'll be open-sourced when it's actually ready and the examples are in place. Until then: look, don't touch. Seriously.

<img src="https://github.com/Walker-Industries-RnD/Malicious-Affiliation-Ban/blob/main/WIBan.png?raw=true" align="center" style="margin-left: 20px; margin-bottom: 20px;"/>

> Read more here: https://github.com/Walker-Industries-RnD/Malicious-Affiliation-Ban/

> Unauthorized use of the artwork — including but not limited to copying, distribution, modification, or inclusion in any machine-learning training dataset — is strictly prohibited and will be prosecuted to the fullest extent of the law.
