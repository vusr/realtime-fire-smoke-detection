#!/usr/bin/env python3
"""
Prediction Visualizer for Fire & Smoke Detection
=================================================

Reads a predictions JSON file produced by inference_pytorch.py or
inference_pytorch_mps.py and overlays bounding boxes onto the original
video, saving an annotated output video.

Usage:
    python visualize_predictions.py --video /path/to/video.mp4 \
                                    --predictions /path/to/predictions.json \
                                    --output annotated_video.mp4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Visual style ──────────────────────────────────────────────────────────────
CLASS_STYLES: dict[str, dict] = {
    'fire':  {'color': (0,  80, 255), 'label_bg': (0,  50, 200)},  # red-orange
    'smoke': {'color': (180, 180, 180), 'label_bg': (100, 100, 100)},  # grey
}
DEFAULT_STYLE = {'color': (0, 255, 0), 'label_bg': (0, 180, 0)}

BBOX_THICKNESS   = 2
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.55
FONT_THICKNESS   = 1
LABEL_PADDING    = 4   # pixels of padding around label text


def load_predictions(json_path: str) -> tuple[dict, list, dict]:
    """Load predictions JSON and return video_info, detections list, and a
    frame-indexed lookup dict for O(1) access per frame."""
    with open(json_path) as f:
        data = json.load(f)

    video_info  = data.get('video_info', {})
    detections  = data.get('detections', [])

    # Build frame → objects lookup
    frame_lookup: dict[int, list] = {}
    for entry in detections:
        frame_lookup[int(entry['frame'])] = entry['objects']

    return video_info, detections, frame_lookup


def draw_detections(frame: np.ndarray, objects: list) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of the frame."""
    out = frame.copy()

    for obj in objects:
        cls        = obj.get('class', 'unknown')
        confidence = obj.get('confidence', 0.0)
        x1, y1, x2, y2 = [int(round(v)) for v in obj['bbox']]

        style     = CLASS_STYLES.get(cls, DEFAULT_STYLE)
        box_color = style['color']
        bg_color  = style['label_bg']

        # Bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, BBOX_THICKNESS)

        # Label text
        label = f"{cls}  {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        # Label background — sit above the box; clamp to frame top
        lx1 = x1
        ly1 = max(0, y1 - th - 2 * LABEL_PADDING - baseline)
        lx2 = x1 + tw + 2 * LABEL_PADDING
        ly2 = max(th + 2 * LABEL_PADDING, y1)

        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), bg_color, -1)
        cv2.putText(
            out, label,
            (lx1 + LABEL_PADDING, ly2 - LABEL_PADDING - baseline),
            FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA,
        )

    return out


def overlay_hud(frame: np.ndarray, frame_idx: int, timestamp: float,
                n_fire: int, n_smoke: int) -> np.ndarray:
    """Optionally draw a small info overlay in the top-right corner."""
    h, w = frame.shape[:2]
    lines = [
        f"Frame {frame_idx}",
        f"Time  {timestamp:.2f}s",
    ]
    if n_fire:
        lines.append(f"Fire  {n_fire}")
    if n_smoke:
        lines.append(f"Smoke {n_smoke}")

    margin    = 10
    line_h    = 20
    panel_w   = 130
    panel_h   = len(lines) * line_h + margin
    px1       = w - panel_w - margin
    py1       = margin
    overlay   = frame.copy()

    cv2.rectangle(overlay, (px1, py1), (w - margin, py1 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (px1 + 6, py1 + (i + 1) * line_h - 4),
                    FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def annotate_video(
    video_path:   str,
    json_path:    str,
    output_path:  str,
    show_hud:     bool = True,
    codec:        str  = 'mp4v',
) -> None:
    """
    Main annotation loop — reads every frame, draws predictions where they
    exist, and writes the result to output_path.
    """
    video_info, _, frame_lookup = load_predictions(json_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video : {video_path}")
    print(f"  Resolution : {width}x{height}  |  FPS : {fps}  |  Frames : {total_frames}")
    print(f"Predictions : {json_path}")
    print(f"  Frames with detections : {len(frame_lookup)}")
    print(f"Output : {output_path}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Cannot create output video: {output_path}")
        cap.release()
        sys.exit(1)

    annotated_frames = 0
    frame_idx        = 0

    with tqdm(total=total_frames, desc="Annotating", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            objects = frame_lookup.get(frame_idx, [])

            if objects:
                frame = draw_detections(frame, objects)
                annotated_frames += 1

                if show_hud:
                    n_fire  = sum(1 for o in objects if o['class'] == 'fire')
                    n_smoke = sum(1 for o in objects if o['class'] == 'smoke')
                    frame   = overlay_hud(frame, frame_idx, frame_idx / fps,
                                          n_fire, n_smoke)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix({'annotated': annotated_frames})

    cap.release()
    writer.release()

    print(f"\nDone.")
    print(f"  Frames annotated : {annotated_frames} / {frame_idx}")
    print(f"  Output saved to  : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Annotate a video with fire/smoke detections from a predictions JSON.'
    )
    parser.add_argument(
        '--video', type=str, required=True,
        help='Path to the original input video file',
    )
    parser.add_argument(
        '--predictions', type=str, required=True,
        help='Path to the predictions JSON file (from inference_pytorch*.py)',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path for the annotated output video (default: <video_stem>_annotated.mp4)',
    )
    parser.add_argument(
        '--no-hud', dest='show_hud', action='store_false', default=True,
        help='Disable the frame/time/count overlay in the corner',
    )
    parser.add_argument(
        '--codec', type=str, default='mp4v',
        help='FourCC codec code for VideoWriter (default: mp4v)',
    )

    args = parser.parse_args()

    video_path = args.video
    json_path  = args.predictions

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    if not Path(json_path).exists():
        print(f"Error: Predictions JSON not found: {json_path}")
        sys.exit(1)

    if args.output is None:
        stem        = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_annotated.mp4")
    else:
        output_path = args.output

    annotate_video(
        video_path  = video_path,
        json_path   = json_path,
        output_path = output_path,
        show_hud    = args.show_hud,
        codec       = args.codec,
    )


if __name__ == '__main__':
    main()
