# Shinra-Meisin
### *神羅明眸 — "All-Seeing"*

**Shinra-Meisin** is an open-source, extensible image-to-prediction tracking pipeline developed by **Walker Industries R&D** in partnership with **EOZVR**. Its first application is binocular eye tracking for VR/AR headsets, designed to run in real time on embedded hardware, but the architecture is deliberately general: given the right dataset and output heads, the same system can learn to track mouths, tongues, or full body pose.

Shinra-Meisin is the tracking core of the **VectorGear** XR headset and the underlying model behind **SM-Tracker**, a plug-and-play snap-on add-on for popular consumer HMDs (Meta Quest 3/3S, BigScreen Beyond 2, Pimax Dream Air).

---

## What Does It Do?

Every frame, Shinra-Meisin looks at an infrared image of one eye and predicts:

- **Where the eye is pointing** — a 3D gaze direction vector
- **Where the pupil is** — 2D position and diameter
- **The shape of the eyelids** — 8 landmark points tracing the upper and lower lid
- **How open the eye is** — an openness scalar from fully closed to fully open

It does this for both eyes simultaneously at up to **90 frames per second** on a compact embedded processor (Rockchip RK3576), with each prediction accompanied by a **confidence estimate** — the model knows when it's uncertain and says so, rather than silently publishing bad data.

Tracking output is published over **OSC/UDP**, making it immediately compatible with social VR platforms like VRChat via [VRCFaceTracking](https://github.com/benaclejames/VRCFaceTracking).

---

## Why Open Source?

Reliable eye tracking has historically been locked behind expensive, proprietary hardware. Shinra-Meisin is FOSS because the underlying software — the model architecture, the training pipeline, the synthetic data tools — should be a foundation the community can build on. The plug-and-play hardware product funds the research; the research belongs to everyone.

---

## How It Works — The Non-Technical Version

Shinra-Meisin is like a student learning to read eyes by studying millions of synthetic examples first, then graduating to real ones.

**Step 1 — Synthetic training.** Rather than needing thousands of real annotated eye images upfront, Shinra-Meisin first trains on photorealistic synthetic eyes generated with [UnityEyes](https://github.com/UnityEyes). Every pixel of every synthetic frame comes with exact ground truth labels — precise pupil position, gaze direction, eyelid shape — computed analytically from the 3D scene. No human annotation required.

**Step 2 — Domain adaptation.** Synthetic images and real infrared camera images look different. A separate network (SRCGAN) learns to bridge this gap, making synthetic images look like real near-infrared captures. This reduces the amount of expensive real labeled data needed to make the tracker work on actual hardware.

**Step 3 — Real-world fine-tuning.** The model is progressively introduced to real IR datasets, refining its predictions from "roughly correct on synthetic" to "precise on real hardware."

**Step 4 — Embedded deployment.** The trained model is quantized to INT8 and deployed via RKNN onto the RK3576 NPU, where it runs the full inference pipeline within an 11ms budget per frame.

---

## Architecture — The Technical Version

Shinra-Meisin is a **multi-task CNN** built on a MobileNetV3-Small backbone modified for single-channel infrared input. The backbone is split into three explicitly tapped stages, each feeding a different output head:

```
Input (224×224×1 IR)
        │
   ┌────▼────┐
   │  Early  │  layers 0–3 → 28×28×24  ──────────────────┐ skip
   └────┬────┘                                             │
        │                                                  │
   ┌────▼────┐                                             │
   │   Mid   │  layers 4–6 → 14×14×40  ─────────┐ skip   │
   └────┬────┘                                   │         │
        │                                        │         │
   ┌────▼────┐                                   │         │
   │  Deep   │  layers 7–12 → 1×1×576            │         │
   └────┬────┘                                   │         │
        │                                        │         │
        ├──► Gaze Head (3D unit vector)           │         │
        │                                        │         │
        │  pointwise 1×1 conv (576→40)           │         │
        │  bilinear upsample (1×1 → 14×14)       │         │
        │  concat with mid skip ──────────────────┘         │
        │  depthwise+pointwise → 14×14×48                   │
        │       │                                           │
        │       ├──► Eyelid Head (8×2 landmark points)      │
        │       │                                           │
        │       │  bilinear upsample (14×14 → 28×28)       │
        │       │  concat with early skip ──────────────────┘
        │       │  depthwise+pointwise → 28×28×16
        │       │       │
        │       │       └──► Pupil Head (2D position + diameter)
```

### U-Net Style Spatial Decoders

The key architectural choice is a **U-Net inspired decoder** for the spatial prediction heads (pupil and eyelid). A standard approach would collapse spatial feature maps with global average pooling before prediction — fast, but it destroys the precise location information that pupil and eyelid landmark prediction require.

Instead:
- The **deep** stage (576 channels, 1×1) is projected down via a cheap pointwise convolution and upsampled back to match the **mid** stage resolution (14×14). The two are concatenated and mixed with a depthwise separable convolution, producing the eyelid decoder features.
- Those eyelid features are then upsampled again to match the **early** stage (28×28), concatenated with the early skip connection, and mixed again — producing the pupil decoder features.

This "passing the torch" pattern lets each head predict from spatially-rich features at the resolution appropriate to its task, while still benefiting from the semantic context built up in the deeper layers.

### Uncertainty-Aware Predictions

Every regression head outputs not just a prediction, but a **log-variance** — a learned estimate of its own uncertainty on each input. This is used in the heteroscedastic multi-task loss:

$$\mathcal{L}_i = \frac{1}{2\sigma_i^2} \mathcal{L}_i^{task} + \frac{1}{2} \log\sigma_i^2$$

In plain terms: when the model is uncertain, it automatically reduces the weight of that head's loss contribution rather than being penalized for honest uncertainty. At inference time, these uncertainty estimates drive the state machine and are published downstream so consuming applications can make informed decisions about when to trust the tracker's output.

### Progressive Training

The backbone is trained in phases, unfreezing from deep to shallow:

| Phase | Backbone state | Dataset |
|---|---|---|
| 0 | Fully frozen | Synthetic only |
| 1 | Deep unfrozen | Synthetic only |
| 2 | Mid unfrozen | Synthetic + real IR (Dikablis) |
| 3 | Early unfrozen | Full mixing, per-head dataset selection |
| 4 | All unfrozen | Hardcase fine-tuning (optional) |

---

## Project Structure

```
shinra-meisin/
├── model.py        # ShinraCNN architecture — backbone, decoders, heads
├── train.py        # Training loop — heteroscedastic loss, phased unfreezing, checkpointing
├── dataset.py      # SyntheticDS dataset class and augmentation transforms
├── synthetic.py    # Label conversion from UnityEyes JSON format to tensors
├── visualize.py    # Per-sample prediction visualization with MSE overlay
├── early_stopping.py
└── main.py         # Quick backbone inspection utility
```

---

## Current Status

Shinra-Meisin is in active development. The current checkpoint (epoch 34) has been trained on synthetic UnityEyes data with the U-Net decoder architecture. Gaze prediction is converging well; pupil and eyelid heads are showing improvement with the spatial decoder over the earlier GAP-only approach.

**Immediate roadmap:**
- SRCGAN domain adaptation (synthetic → real NIR)
- Real IR fine-tuning on Dikablis / OpenEDS 2019
- ROI state machine and Kalman filter implementation
- RKNN INT8 quantization and RK3576 deployment
- VRCFaceTracking C# module

**Later:**
- Mouth / lip / tongue tracking (same backbone, new heads and dataset)
- Crowdfunded FOSS dataset collected across target HMDs
- Full body inside-out tracking

---

## Datasets

Shinra-Meisin currently trains on synthetic data generated with **UnityEyes2**. Real IR fine-tuning plans include:
- **VRGaze** — primary source for SRCGAN domain adaptation

An in-house dataset covering eye and mouth modalities across BSB2, Quest 3/3S, and Pimax Dream Air hardware is planned for public FOSS release once prototype hardware is available. Pending publication status.

---

## Requirements

```
torch
torchvision
opencv-python
numpy
```

See `requirements.txt` for the full pinned dependency list.

---

## License
<img src="https://github.com/Walker-Industries-RnD/Malicious-Affiliation-Ban/blob/main/WIBan.png?raw=true" align="center" style="margin-left: 20px; margin-bottom: 20px;"/>

> Unauthorized use of the artwork — including but not limited to copying, distribution, modification, or inclusion in any machine-learning training dataset — is strictly prohibited and will be prosecuted to the fullest extent of the law.
