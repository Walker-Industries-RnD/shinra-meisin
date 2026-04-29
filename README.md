# Shinra-Meisin
### *神羅明眸 — "The All-Seeing Eminence Of Shadow"*

**The Shinra-Meisin** is an open-source, extensible tracking pipeline developed by **Walker Industries R&D** in partnership with **EOZVR**. Its first application is the binocular eye tracking system for VR/AR headsets, designed to run in real time on embedded hardware, but the architecture is deliberately general: given the right dataset and output heads, the same system can learn to track mouths, tongues, or full body pose.

Shinra-Meisin is the tracking core of the **VectorGear** XR headset and the underlying model behind **SM-Tracker**, a plug-and-play snap-on add-on for popular consumer HMDs (Meta Quest 3/3S, BigScreen Beyond 2, Pimax Dream Air).

---

## What Does It Do?

Every frame, Shinra-Meisin looks at an infrared image of one eye and predicts:

- **Where the eye is pointing** — a 3D gaze direction vector
- **Where the pupil is** — 2D position and diameter
- **The shape of the eyelids** — 8 landmark points tracing the upper and lower lid
- **How open the eye is** — an openness scalar from fully closed to fully open

It does this for both eyes simultaneously at up to **90 frames per second** on a compact embedded processor (Rockchip RK3576), with each prediction accompanied by a **confidence estimate** — the model knows when it's uncertain and says so, rather than silently publishing bad data. It also can be streamed onto a computer VIA Wifi6, allowing for very lightweight systems.

Tracking output is published over **OSC/UDP**, making it immediately compatible with social VR platforms like VRChat via [VRCFaceTracking](https://github.com/benaclejames/VRCFaceTracking).

---

## Why Open Source?

Reliable eye tracking has historically been locked behind expensive, proprietary hardware. Shinra-Meisin is FOSS because the underlying software — the model architecture, the training pipeline, the synthetic data tools — should be a foundation the community can build on. The plug-and-play hardware product funds the research; the research belongs to everyone.

---

## How It Works — The Non-Technical Version

Shinra-Meisin is like a student learning to read eyes by studying millions of synthetic examples first, then graduating to real ones.

**Step 1 — Synthetic training.** Rather than needing thousands of real annotated eye images upfront, Shinra-Meisin first trains on photorealistic synthetic eyes generated with [UnityEyes2](https://github.com/alexanderdsmith/UnityEyes2). Every pixel of every synthetic frame comes with exact ground truth labels — precise pupil position, gaze direction, eyelid shape — computed analytically from the 3D scene. No human annotation required.

**Step 2 — Domain adaptation.** Synthetic images and real infrared camera images look different. A separate network (SRCGAN) learns to bridge this gap, making synthetic images look like real near-infrared captures. This reduces the amount of expensive real labeled data needed to make the tracker work on actual hardware.

**Step 3 — Real-world fine-tuning.** The model is progressively introduced to real IR datasets, refining its predictions from "roughly correct on synthetic" to "precise on real hardware."

**Step 4 — Embedded deployment.** The trained model is quantized to INT8 and deployed via RKNN onto the RK3576 NPU, where it runs the full inference pipeline within an 11ms budget per frame.

---

## Architecture — The Technical Version

Shinra-Meisin is a **multi-task CNN** built on a MobileNetV3-Small backbone modified for single-channel infrared input. Rather than hardcoding a fixed set of skip connections, the model **dynamically segments the backbone at every strided convolution** — one segment per resolution-halving layer, one skip connection saved per segment. This makes the decoder chain automatically adapt to different backbone choices without manual wiring.

On MobileNetV3-Small, `backbone.features` (everything before the GAP and classifier) contains five strided layers, producing five segments and a bottleneck at **7×7×96**:

```
Input (224×224×1 IR)
        │
   ┌────▼────┐
   │  Seg 0  │  stride-2  →  112×112×16  ────────────────────────────────────────┐ skip₀
   └────┬────┘                                                                     │
        │                                                                          │
   ┌────▼────┐                                                                     │
   │  Seg 1  │  stride-2  →   56×56×16   ──────────────────────────────┐ skip₁   │
   └────┬────┘                                                           │          │
        │                                                                │          │
   ┌────▼────┐                                                           │          │
   │  Seg 2  │  stride-2  →   28×28×24   ─────────────────────┐ skip₂  │          │
   └────┬────┘                                                  │         │          │
        │                                                       │         │          │
   ┌────▼────┐                                                  │         │          │
   │  Seg 3  │  stride-2  →   14×14×40   ──────────┐ skip₃    │         │          │
   └────┬────┘                                      │            │         │          │
        │                                           │            │         │          │
   ┌────▼────┐                                      │            │         │          │
   │  Seg 4  │  stride-2  →    7×7×96  (bottleneck) │            │         │          │
   └────┬────┘                                      │            │         │          │
        │                                           │            │         │          │
        ├──► Gaze Head  (GAP → 96-d FC → 3D unit vector + σ²)   │         │          │
        │                                           │            │         │          │
        │  upsample  7→14,  cat skip₃ ──────────────┘            │         │          │
        │  DepthwiseSepConv  →  14×14×40                          │         │          │
        │       │                                                  │         │          │
        │  upsample 14→28,  cat skip₂ ──────────────────────────── ┘         │          │
        │  DepthwiseSepConv  →  28×28×24                                      │          │
        │       │                                                              │          │
        │  upsample 28→56,  cat skip₁ ──────────────────────────────────────── ┘          │
        │  DepthwiseSepConv  →  56×56×16                                                  │
        │       │                                                                          │
        │  upsample 56→112, cat skip₀ ──────────────────────────────────────────────────── ┘
        │  DepthwiseSepConv  →  112×112×16
        │       │
        │       ├──► Heatmap Head  (1×1 conv  →  112×112×17)
        │       │      ch 0      :  pupil center
        │       │      ch 1–8   :  upper eyelid (8 landmarks)
        │       │      ch 9–16  :  lower eyelid (8 landmarks)
        │       │      [inference: soft-argmax  →  (B, 17, 2) coordinates]
        │       │
        │       └──► Diameter Head  (GAP → 16-d MLP → scalar + σ²)
```

### Dynamic, Extensible U-Net Decoder Chain

The spatial decoder is built from a chain of `DecodeBlock` modules — one per consecutive pair of adjacent skip connections, assembled automatically from the backbone's stride points. Each `DecodeBlock` bilinearly upsamples its input to match the next skip's spatial size, concatenates the two, and refines through a depthwise separable convolution. The chain passes features progressively from the deep bottleneck all the way back to the shallowest resolution, accumulating fine spatial detail at each step.

Because the segments and blocks are derived programmatically rather than wired by hand, the decoder extends for free: swapping in a deeper backbone, or one with more strided layers, produces a longer chain with more skip connections without touching any other code.

The single unified **heatmap head** — a 1×1 convolution at the end of the decoder — outputs all 17 landmark channels at 112×112. At inference, each channel's spatial distribution is decoded to a 2D coordinate via **soft-argmax** (weighted centroid over a softmax-normalized heatmap), giving sub-pixel localization without any argmax discontinuity. Pupil diameter is predicted from the same final decoder features through a separate pooling + MLP branch.

The **gaze head** operates entirely independently, reading from the bottleneck via global average pooling before the decoder chain runs — it benefits from the backbone's full semantic depth but is never influenced by the decoder or the spatial heads.

### Uncertainty-Aware Predictions

Every regression head outputs not just a prediction, but a **log-variance** — a learned estimate of its own uncertainty on each input. This is used in the heteroscedastic multi-task loss:

$$\mathcal{L}_i = \frac{1}{2\sigma_i^2} \mathcal{L}_i^{task} + \frac{1}{2} \log\sigma_i^2$$

In plain terms: when the model is uncertain, it automatically reduces the weight of that head's loss contribution rather than being penalized for honest uncertainty. At inference time, these uncertainty estimates drive the state machine and are published downstream so consuming applications can make informed decisions about when to trust the tracker's output.

### Progressive Training

The backbone is unfrozen one segment at a time, from deepest to shallowest. Each phase adds the next segment to the optimizer with its own learning rate group, so earlier, more general features are always trained at a lower rate than the newly thawed layer:

| Phase | Newly unfrozen segment | Dataset |
|---|---|---|
| 0 | None — output heads only | Synthetic only |
| 1 | Seg 4 — bottleneck (7×7×96) | Synthetic only |
| 2 | Seg 3 — (14×14×40) | Synthetic only |
| 3 | Seg 2 — (28×28×24) | Synthetic + real IR |
| 4 | Seg 1 — (56×56×16) | Full mixing |
| 5 | Seg 0 — shallowest (112×112×16) | Hardcase fine-tuning (optional) |

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

Shinra-Meisin is in active development. The current checkpoint has been trained on synthetic UnityEyes data with the dynamic U-Net decoder architecture. Gaze prediction is converging well; the unified heatmap head (pupil center + 17 eyelid landmarks) is showing improvement with the full spatial decoder chain over earlier GAP-only approaches.

**Immediate roadmap:**
- SRCGAN domain adaptation (synthetic → real NIR)
- Real IR fine-tuning on Dikablis / OpenEDS 2019
- ROI state machine and Kalman filter implementation
- RKNN INT8 quantization and RK3576 deployment
- VRCFaceTracking C# module

**Later:**
- Mouth / lip / tongue tracking (same backbone, new heads and dataset)
- Inhouse curated FOSS dataset collected across target HMDs
- Full body inside-out tracking for our WIP controllers
- BCI Data layer with VRChat OSC + Resonite and [Eclipse](https://github.com/Walker-Industries-RnD/Eclipse/tree/main/EclipseProject) Layer

---

## Datasets

Shinra-Meisin currently trains on synthetic data generated with **UnityEyes2**. Real IR fine-tuning plans include:
- **VRGaze** — primary source for SRCGAN domain adaptation

An in-house dataset covering eye and mouth modalities across BSB2, Quest 3/3S, and Pimax Dream Air hardware is planned for public FOSS release once prototype hardware is available. **NO CONSUMER DATA WILL BE COLLECTED FOR TRAINING.** Pending publication status.

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


Copyright (c) 2026 Walker Industries R&D All rights reserved.

This is a work-in-progress prototype. You may view the source code for personal evaluation purposes only. NO license is granted (express or implied) for:

- copying
- modification
- distribution
- commercial use
- derivative works
- or any other form of exploitation
  
It'll be open-sourced when it's actually ready and has examples ready. Until then: look, don't touch. Seriously.


<img src="https://github.com/Walker-Industries-RnD/Malicious-Affiliation-Ban/blob/main/WIBan.png?raw=true" align="center" style="margin-left: 20px; margin-bottom: 20px;"/>

> Read more here: https://github.com/Walker-Industries-RnD/Malicious-Affiliation-Ban/



> Unauthorized use of the artwork — including but not limited to copying, distribution, modification, or inclusion in any machine-learning training dataset — is strictly prohibited and will be prosecuted to the fullest extent of the law.
