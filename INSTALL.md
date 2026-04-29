# Visualizing Shinra-Meisin
This is an early work-in-progress tracking pipeline accustomed to synthetic eye data ([dataset](https://huggingface.co/datasets/jpena-173/sm-eyes-v1)). However, it is possible to judge its current progress with input eye imagery. Here is what you need to do:

1. Download sm-eyes-v1 (Parquet UnityEyes2-based dataset)
2. Install requirements - python3 -m pip install -r requirements.txt
3. Run visualize.py/debug_heatmap.py. Here are their descriptions:

visualize.py
Controls:
  q                         close window (advance to next sample if --n > 1)
Usage:
    python visualize.py             frame-by-frame annotation (pupil position, eyelid landmarks, gaze vector)
    python visualize.py --webcam    run inference on a 256x256 center crop of attached webcam. overlay shows heatmap of max activations, along with annotations

debug_heatmap.py
Controls:
  ← / →  or scroll wheel   previous / next panel
  Tab                       jump to the start of the next group
  q                         close window (advance to next sample if --n > 1)

Usage:
    python debug_heatmaps.py              # one random sample
    python debug_heatmaps.py --idx 42     # specific dataset index
    python debug_heatmaps.py --n 4        # cycle through N random samples
    python debug_heatmaps.py --no-model   # GT heatmaps only (no checkpoint needed)