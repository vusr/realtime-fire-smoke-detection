#!/usr/bin/env python3
"""
TensorRT Inference Script for Fire & Smoke Detection
=====================================================

Optimized inference using TensorRT INT8 quantized YOLOv11 model.
Designed for processing long video feeds efficiently with batch processing.

Usage:
    python inference_tensorrt.py --video /path/to/video.mp4 --output results/
"""

import argparse
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tqdm import tqdm


class TensorRTInference:
    """TensorRT inference engine for YOLO object detection."""
    
    def __init__(self, engine_path: str, class_names: List[str]):
        """
        Initialize TensorRT inference engine.
        
        Args:
            engine_path: Path to TensorRT engine file (.engine)
            class_names: List of class names
        """
        self.class_names = class_names
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine from {engine_path}...")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get input/output details
        self.input_shape = self.engine.get_tensor_shape("images")
        self.batch_size = self.input_shape[0]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        
        # Allocate buffers
        self.allocate_buffers()
        
        print(f"✓ Engine loaded successfully")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Batch size: {self.batch_size}")
        
    def allocate_buffers(self):
        """Allocate GPU buffers for inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Allocate host and device buffers
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
    
    def preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess images for inference.
        
        Args:
            images: List of BGR images (OpenCV format)
            
        Returns:
            Preprocessed batch as numpy array
        """
        batch = []
        for img in images:
            # Letterbox resize
            img_resized = self.letterbox(img, (self.input_w, self.input_h))
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            img_norm = img_rgb.astype(np.float32) / 255.0
            # HWC to CHW
            img_chw = np.transpose(img_norm, (2, 0, 1))
            batch.append(img_chw)
        
        # Pad batch if needed
        while len(batch) < self.batch_size:
            batch.append(np.zeros_like(batch[0]))
        
        return np.array(batch, dtype=np.float32)
    
    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int], 
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
    
    def infer(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed batch.
        
        Args:
            batch: Preprocessed image batch
            
        Returns:
            Raw model output
        """
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], batch.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), 
                self.bindings[i]
            )
        
        self.context.execute_v2(self.bindings)
        
        # Copy output to CPU
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # Reshape output
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        return output
    
    def postprocess(self, output: np.ndarray, conf_thresh: float = 0.25,
                   iou_thresh: float = 0.65) -> List[List[Dict]]:
        """
        Post-process model output to get detections.
        
        Args:
            output: Raw model output
            conf_thresh: Confidence threshold
            iou_thresh: NMS IoU threshold
            
        Returns:
            List of detections per image in batch
        """
        batch_detections = []
        
        for batch_idx in range(output.shape[0]):
            detections = []
            preds = output[batch_idx]  # (num_boxes, 4 + num_classes)
            
            # YOLO output format: [x_center, y_center, width, height, class_scores...]
            if len(preds.shape) == 2:
                boxes = preds[:, :4]
                scores = preds[:, 4:]
            else:
                # Handle different output formats
                continue
            
            # Get class with max score for each box
            class_scores = scores.max(axis=1)
            class_ids = scores.argmax(axis=1)
            
            # Filter by confidence
            mask = class_scores >= conf_thresh
            boxes = boxes[mask]
            class_scores = class_scores[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) == 0:
                batch_detections.append([])
                continue
            
            # Convert from center format to corner format
            x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            # Apply NMS
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
            keep_indices = self.nms(boxes_xyxy, class_scores, iou_thresh)
            
            # Build detection list
            for idx in keep_indices:
                detections.append({
                    'class': self.class_names[class_ids[idx]],
                    'confidence': float(class_scores[idx]),
                    'bbox': [float(x1[idx]), float(y1[idx]), 
                            float(x2[idx]), float(y2[idx])]
                })
            
            batch_detections.append(detections)
        
        return batch_detections
    
    @staticmethod
    def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Boxes in xyxy format (N, 4)
            scores: Confidence scores (N,)
            iou_thresh: IoU threshold
            
        Returns:
            Indices of kept boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        
        return keep


def process_video(video_path: str, engine_path: str, output_dir: str,
                 conf_thresh: float = 0.25, iou_thresh: float = 0.65,
                 batch_size: int = 16, skip_frames: int = 1):
    """
    Process video file and save detections.
    
    Args:
        video_path: Path to input video
        engine_path: Path to TensorRT engine
        output_dir: Output directory for results
        conf_thresh: Confidence threshold
        iou_thresh: NMS IoU threshold
        batch_size: Batch size for processing
        skip_frames: Process every Nth frame
    """
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_names = ['smoke', 'fire']
    
    # Initialize inference engine
    inference = TensorRTInference(engine_path, class_names)
    
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
                
                # Preprocess
                batch = inference.preprocess(frame_buffer)
                
                # Inference
                output = inference.infer(batch)
                
                # Postprocess
                batch_detections = inference.postprocess(output, conf_thresh, iou_thresh)
                
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
            batch = inference.preprocess(frame_buffer)
            output = inference.infer(batch)
            batch_detections = inference.postprocess(output, conf_thresh, iou_thresh)
            
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
        description='Fire & Smoke Detection - TensorRT Inference'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--model', type=str, 
                       default='models/yolo26l_int8_bs16.engine',
                       help='Path to TensorRT engine file')
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
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Process video
    process_video(
        args.video,
        args.model,
        args.output,
        args.conf_thresh,
        args.iou_thresh,
        args.batch_size,
        args.skip_frames
    )


if __name__ == '__main__':
    main()
