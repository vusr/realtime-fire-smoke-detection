#!/usr/bin/env python3
"""
PyTorch Inference Script for Fire & Smoke Detection
====================================================

Now supports:
- Apple Silicon (MPS)
- CUDA
- CPU fallback

Usage:
    python inference_pytorch.py --video /path/to/video.mp4 --output results/
"""

import argparse
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


class PyTorchInference:
    """PyTorch inference engine for YOLO object detection."""

    def __init__(self, model_path: str, class_names: List[str],
                 half: bool = True, device: Optional[str] = None):

        self.class_names = class_names

        # -------- Device Selection Priority --------
        if device:
            self.device = torch.device(device)
        else:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        # FP16 only works properly on CUDA
        self.half = half and self.device.type == "cuda"

        print(f"Loading PyTorch model from {model_path}...")

        try:
            from ultralytics import YOLO

            self.model = YOLO(model_path)
            self.model.to(self.device)

            if self.half:
                self.model.model.half()

            self.input_size = 640  # YOLO default

            print("✓ Model loaded successfully")
            print(f"  Device: {self.device}")
            print(f"  Precision: {'FP16' if self.half else 'FP32'}")
            print(f"  Input size: {self.input_size}")

        except ImportError:
            print("Error: ultralytics package not found.")
            print("Install with: pip install ultralytics")
            sys.exit(1)

    def infer_batch(self, images: List[np.ndarray],
                    conf_thresh: float = 0.25,
                    iou_thresh: float = 0.65) -> List[List[Dict]]:

        batch_detections = []

        with torch.no_grad():
            results = self.model.predict(
                images,
                conf=conf_thresh,
                iou=iou_thresh,
                half=self.half,
                verbose=False,
                device=str(self.device)  # IMPORTANT for MPS
            )

        for result in results:
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'class': self.class_names[cls_id],
                        'confidence': float(conf),
                        'bbox': [
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3])
                        ]
                    })

            batch_detections.append(detections)

        return batch_detections


def process_video(video_path: str, model_path: str, output_dir: str,
                  conf_thresh: float = 0.25,
                  iou_thresh: float = 0.65,
                  batch_size: int = 16,
                  skip_frames: int = 1,
                  half: bool = True):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    class_names = ['smoke', 'fire']
    inference = PyTorchInference(model_path, class_names, half=half)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"\nProcessing with batch_size={batch_size}, skip_frames={skip_frames}")

    all_detections = []
    frame_buffer = []
    frame_indices = []

    total_detections = 0
    fire_count = 0
    smoke_count = 0
    processing_times = []

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            frame_buffer.append(frame)
            frame_indices.append(frame_idx)

            if len(frame_buffer) == batch_size:
                start_time = time.time()

                batch_detections = inference.infer_batch(
                    frame_buffer, conf_thresh, iou_thresh
                )

                processing_time = time.time() - start_time
                processing_times.append(processing_time)

                for f_idx, detections in zip(frame_indices, batch_detections):
                    if detections:
                        timestamp = f_idx / fps
                        all_detections.append({
                            'frame': f_idx,
                            'timestamp': timestamp,
                            'objects': detections
                        })

                        total_detections += len(detections)
                        for det in detections:
                            if det['class'] == 'fire':
                                fire_count += 1
                            else:
                                smoke_count += 1

                frame_buffer.clear()
                frame_indices.clear()

                current_fps = batch_size / processing_time
                pbar.set_postfix({
                    'FPS': f'{current_fps:.1f}',
                    'Detections': total_detections
                })

            frame_idx += 1
            pbar.update(1)

        if frame_buffer:
            batch_detections = inference.infer_batch(
                frame_buffer, conf_thresh, iou_thresh
            )

            for f_idx, detections in zip(frame_indices, batch_detections):
                if detections:
                    timestamp = f_idx / fps
                    all_detections.append({
                        'frame': f_idx,
                        'timestamp': timestamp,
                        'objects': detections
                    })

                    total_detections += len(detections)
                    for det in detections:
                        if det['class'] == 'fire':
                            fire_count += 1
                        else:
                            smoke_count += 1

    finally:
        pbar.close()
        cap.release()

    avg_fps = (
        len(processing_times) * batch_size / sum(processing_times)
        if processing_times else 0
    )

    json_output = {
        'video_info': {
            'path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'resolution': [width, height],
            'processed_frames': frame_idx
        },
        'detections': all_detections,
        'summary': {
            'total_detections': total_detections,
            'fire_count': fire_count,
            'smoke_count': smoke_count,
            'avg_fps': avg_fps,
            'frames_with_detections': len(all_detections)
        }
    }

    json_path = output_path / 'predictions.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)

    print(f"\n✓ JSON output saved to {json_path}")
    print("\nProcessing Complete!")
    print(f"Total Detections: {total_detections}")
    print(f"Average Processing FPS: {avg_fps:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Fire & Smoke Detection - PyTorch Inference'
    )

    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str,
                        default='models/yolo26l_best.pt')
    parser.add_argument('--output', type=str, default='output/')
    parser.add_argument('--conf-thresh', type=float, default=0.25)
    parser.add_argument('--iou-thresh', type=float, default=0.65)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--half', action='store_true', default=True)
    parser.add_argument('--no-half', dest='half', action='store_false')

    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Device info
    if torch.backends.mps.is_available():
        print("✓ Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        print("✓ Using CUDA GPU")
    else:
        print("⚠ Running on CPU (no GPU detected)")

    process_video(
        args.video,
        args.model,
        args.output,
        args.conf_thresh,
        args.iou_thresh,
        args.batch_size,
        args.skip_frames,
        args.half
    )


if __name__ == '__main__':
    main()
