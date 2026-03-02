#!/usr/bin/env python3
"""
PyTorch Inference Script for Fire & Smoke Detection
====================================================

Pure PyTorch inference with FP16 support for systems without TensorRT.
Designed for processing long video feeds efficiently with batch processing.

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
                 half: bool = True, device: str = 'cuda'):
        """
        Initialize PyTorch inference engine.
        
        Args:
            model_path: Path to PyTorch model file (.pt)
            class_names: List of class names
            half: Use FP16 precision
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.class_names = class_names
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.half = half and self.device.type == 'cuda'
        
        print(f"Loading PyTorch model from {model_path}...")
        
        # Load model using ultralytics YOLO
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            if self.half:
                self.model.model.half()
            
            # Get model input size
            self.input_size = 640  # YOLO default
            
            print(f"✓ Model loaded successfully")
            print(f"  Device: {self.device}")
            print(f"  Precision: {'FP16' if self.half else 'FP32'}")
            print(f"  Input size: {self.input_size}")
            
        except ImportError:
            print("Error: ultralytics package not found. Install with: pip install ultralytics")
            sys.exit(1)
    
    def preprocess(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess images for inference.
        
        Args:
            images: List of BGR images (OpenCV format)
            
        Returns:
            Preprocessed batch as torch tensor
        """
        batch = []
        for img in images:
            # Letterbox resize
            img_resized = self.letterbox(img, (self.input_size, self.input_size))
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            img_norm = img_rgb.astype(np.float32) / 255.0
            # HWC to CHW
            img_chw = np.transpose(img_norm, (2, 0, 1))
            batch.append(img_chw)
        
        # Convert to tensor
        batch_tensor = torch.from_numpy(np.array(batch)).to(self.device)
        
        if self.half:
            batch_tensor = batch_tensor.half()
        
        return batch_tensor
    
    @staticmethod
    def letterbox(img: np.ndarray, new_shape: tuple, 
                  color=(114, 114, 114)) -> np.ndarray:
        """
        Resize image with aspect ratio preservation (letterbox).
        
        Args:
            img: Input image
            new_shape: Target (width, height)
            color: Fill color for borders
            
        Returns:
            Resized image
        """
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                 cv2.BORDER_CONSTANT, value=color)
        return img
    
    def infer_batch(self, images: List[np.ndarray], conf_thresh: float = 0.25,
                   iou_thresh: float = 0.65) -> List[List[Dict]]:
        """
        Run inference on a batch of images using ultralytics YOLO.
        
        Args:
            images: List of BGR images
            conf_thresh: Confidence threshold
            iou_thresh: NMS IoU threshold
            
        Returns:
            List of detections per image
        """
        batch_detections = []
        
        # Run inference using ultralytics
        with torch.no_grad():
            results = self.model.predict(
                images,
                conf=conf_thresh,
                iou=iou_thresh,
                half=self.half,
                verbose=False,
                device=self.device
            )
        
        # Parse results
        for result in results:
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls_id in zip(boxes, confidences, class_ids):
                    detections.append({
                        'class': self.class_names[cls_id],
                        'confidence': float(conf),
                        'bbox': [float(box[0]), float(box[1]), 
                                float(box[2]), float(box[3])]
                    })
            
            batch_detections.append(detections)
        
        return batch_detections


def process_video(video_path: str, model_path: str, output_dir: str,
                 conf_thresh: float = 0.25, iou_thresh: float = 0.65,
                 batch_size: int = 16, skip_frames: int = 1, 
                 half: bool = True):
    """
    Process video file and save detections.
    
    Args:
        video_path: Path to input video
        model_path: Path to PyTorch model
        output_dir: Output directory for results
        conf_thresh: Confidence threshold
        iou_thresh: NMS IoU threshold
        batch_size: Batch size for processing
        skip_frames: Process every Nth frame
        half: Use FP16 precision
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_names = ['smoke', 'fire']
    
    # Initialize inference engine
    inference = PyTorchInference(model_path, class_names, half=half)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"\nProcessing with batch_size={batch_size}, skip_frames={skip_frames}")
    
    # Storage for results
    all_detections = []
    frame_buffer = []
    frame_indices = []
    
    # Statistics
    total_detections = 0
    fire_count = 0
    smoke_count = 0
    processing_times = []
    
    # Process video
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            
            frame_buffer.append(frame)
            frame_indices.append(frame_idx)
            
            # Process batch when full
            if len(frame_buffer) == batch_size:
                start_time = time.time()
                
                # Inference
                batch_detections = inference.infer_batch(
                    frame_buffer, conf_thresh, iou_thresh
                )
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Save results
                for idx, (f_idx, detections) in enumerate(zip(frame_indices, batch_detections)):
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
                
                # Clear buffer
                frame_buffer = []
                frame_indices = []
                
                # Update progress
                current_fps = batch_size / processing_time
                pbar.set_postfix({'FPS': f'{current_fps:.1f}', 
                                'Detections': total_detections})
            
            frame_idx += 1
            pbar.update(1)
        
        # Process remaining frames
        if frame_buffer:
            batch_detections = inference.infer_batch(
                frame_buffer, conf_thresh, iou_thresh
            )
            
            for idx, (f_idx, detections) in enumerate(zip(frame_indices, batch_detections)):
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
    
    # Calculate statistics
    avg_fps = len(processing_times) * batch_size / sum(processing_times) if processing_times else 0
    
    # Save JSON output
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
    
    # Save CSV output
    csv_path = output_path / 'predictions.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        
        for detection in all_detections:
            frame = detection['frame']
            timestamp = detection['timestamp']
            for obj in detection['objects']:
                writer.writerow([
                    frame,
                    f"{timestamp:.2f}",
                    obj['class'],
                    f"{obj['confidence']:.3f}",
                    f"{obj['bbox'][0]:.1f}",
                    f"{obj['bbox'][1]:.1f}",
                    f"{obj['bbox'][2]:.1f}",
                    f"{obj['bbox'][3]:.1f}"
                ])
    print(f"✓ CSV output saved to {csv_path}")
    
    # Save summary
    summary_path = output_path / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Fire & Smoke Detection - Processing Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Processed Frames: {frame_idx}\n")
        f.write(f"Video FPS: {fps}\n")
        f.write(f"Processing FPS: {avg_fps:.2f}\n\n")
        f.write(f"Total Detections: {total_detections}\n")
        f.write(f"  Fire: {fire_count}\n")
        f.write(f"  Smoke: {smoke_count}\n")
        f.write(f"Frames with Detections: {len(all_detections)}\n\n")
        f.write(f"Confidence Threshold: {conf_thresh}\n")
        f.write(f"IoU Threshold: {iou_thresh}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Skip Frames: {skip_frames}\n")
        f.write(f"Precision: {'FP16' if half else 'FP32'}\n")
    print(f"✓ Summary saved to {summary_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total Detections: {total_detections}")
    print(f"  Fire: {fire_count}")
    print(f"  Smoke: {smoke_count}")
    print(f"Frames with Detections: {len(all_detections)}")
    print(f"Average Processing FPS: {avg_fps:.2f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Fire & Smoke Detection - PyTorch Inference'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--model', type=str, 
                       default='models/yolo26l_best.pt',
                       help='Path to PyTorch model file')
    parser.add_argument('--output', type=str, default='output/',
                       help='Output directory for predictions')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou-thresh', type=float, default=0.65,
                       help='NMS IoU threshold (default: 0.65)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference (default: 16)')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (default: 1)')
    parser.add_argument('--half', action='store_true', default=True,
                       help='Use FP16 precision (default: True)')
    parser.add_argument('--no-half', dest='half', action='store_false',
                       help='Use FP32 precision instead of FP16')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU will be slow.")
        print("Consider using a GPU for faster inference.")
    
    # Process video
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
