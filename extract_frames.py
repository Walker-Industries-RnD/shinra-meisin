#!/usr/bin/env python3
"""Pre-extract GIW video frames to a PNG cache.

Run once before training. Frames are written as lossless PNGs named by their
true 0-based frame number, so input.py can use the filename directly as the GT
row index regardless of stride.

Resumable — scans existing PNGs to find the last written frame and continues
from there.

Output layout:
    {root}/{outer}/{session}/frames/{eye}/{fi:06d}.png

Usage:
    python extract_frames.py --root /path/to/GIW
    python extract_frames.py --root /path/to/GIW --stride 3 --jobs 4
"""

import argparse
import glob
import os
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'


def frames_dir(video_path):
    session_dir = os.path.dirname(video_path)
    eye_name    = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(session_dir, 'frames', eye_name)


def extract_video(video_path, bar, stride):
    """Extract every `stride`-th frame from one video, resuming from existing PNGs.

    Filenames encode the true 0-based frame number so input.py can index GT
    arrays directly without any remapping.

    Returns (video_path, n_written, n_total).
    """
    out_dir = frames_dir(video_path)

    session_name = os.path.basename(os.path.dirname(video_path))
    eye_name     = os.path.splitext(os.path.basename(video_path))[0]
    desc         = f'{session_name}/{eye_name}'

    cap      = cv2.VideoCapture(video_path)
    n_video  = max(8, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    n_expect = (n_video + stride - 1) // stride  # frames we intend to write

    existing_pngs = glob.glob(os.path.join(out_dir, '*.jpg'))
    n_existing    = len(existing_pngs)

    # MJPEG metadata drift scales with video length; use a relative threshold.
    if n_existing >= n_expect * 0.995:
        cap.release()
        return video_path, 0, n_existing

    os.makedirs(out_dir, exist_ok=True)

    if n_existing > 0:
        # Resume: seek to the frame after the last one written.
        last_fi = max(int(os.path.splitext(os.path.basename(p))[0]) for p in existing_pngs)
        fi = last_fi + stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    else:
        fi = 0

    bar.reset(total=n_expect)
    bar.set_description(desc)
    bar.n = n_existing
    bar.refresh()

    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.imwrite(os.path.join(out_dir, f'{fi:06d}.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        n_written += 1
        bar.update(1)

        fi += stride
        # Skip stride-1 frames using grab() — cheaper than read() (no decode).
        for _ in range(stride - 1):
            if not cap.grab():
                break

    cap.release()
    return video_path, n_written, n_existing + n_written


def main():
    parser = argparse.ArgumentParser(
        description='Pre-extract GIW video frames to PNG cache'
    )
    parser.add_argument('--root', required=True,
                        help='Root GIW directory (contains subject sub-folders)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Write every Nth frame (default: 1 = all frames). '
                             'Filenames encode the true frame number so GT '
                             'alignment in input.py is automatic.')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Parallel worker threads (default: 1)')
    args = parser.parse_args()

    if args.stride < 1:
        parser.error('--stride must be >= 1')

    videos = sorted(glob.glob(os.path.join(args.root, '*', '*', 'eye?.mp4')))
    if not videos:
        print(f'No eye?.mp4 files found under {args.root}')
        return

    tqdm.write(f'Found {len(videos)} video(s)  |  stride={args.stride}  |  jobs={args.jobs}')

    bar_pool = queue.Queue()
    bars = [tqdm(total=0, position=i, leave=True, dynamic_ncols=True)
            for i in range(args.jobs)]
    for bar in bars:
        bar_pool.put(bar)

    def run(video_path):
        bar = bar_pool.get()
        try:
            return extract_video(video_path, bar, args.stride)
        finally:
            bar.set_description('idle')
            bar.refresh()
            bar_pool.put(bar)

    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, v): v for v in videos}
        for fut in as_completed(futures):
            path, n_written, n_total = fut.result()
            done += 1
            rel = os.path.relpath(path, args.root)
            if n_written == 0:
                label = 'skipped (already complete)'
            elif n_total - n_written > 0:
                label = f'resumed — {n_written} frames written ({n_total} total)'
            else:
                label = f'{n_written} frames written'
            tqdm.write(f'[{done}/{len(videos)}] {rel} — {label}')

    for bar in bars:
        bar.close()

    tqdm.write('Done.')


if __name__ == '__main__':
    main()
