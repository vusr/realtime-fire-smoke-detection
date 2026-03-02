# Model Files

This directory contains the trained fire and smoke detection models.

## Files

### 1. yolo26l_int8_bs16.engine
- **Type**: TensorRT INT8 quantized model
- **Batch size**: 16
- **Size**: ~31 MB
- **Performance**: 459 img/s on NVIDIA L4 GPU
- **mAP@50**: 0.7667
- **mAP@50:95**: 0.4444
- **Use**: Primary inference model for TensorRT script

**Usage:**
```bash
python inference_tensorrt.py --model models/yolo26l_int8_bs16.engine --video video.mp4
```

### 2. yolo26l_best.pt
- **Type**: PyTorch FP32 model
- **Size**: ~53 MB
- **Performance**: 129.7 img/s on NVIDIA L4 GPU (FP32), 143.7 img/s (FP16)
- **mAP@50**: 0.7894
- **mAP@50:95**: 0.4805
- **Use**: Fallback model for systems without TensorRT

**Usage:**
```bash
python inference_pytorch.py --model models/yolo26l_best.pt --video video.mp4
```

### 3. dataset.yaml
- **Type**: Configuration file
- **Content**: Class names and dataset paths
- **Use**: Reference for class indices

```yaml
nc: 2
names:
  0: smoke
  1: fire
```

## Model Selection Guide

### Use TensorRT Model When:
- ✓ TensorRT is installed
- ✓ Maximum performance needed (3.5× faster)
- ✓ Running on RTX 4080 or similar GPU
- ✓ Processing long videos

### Use PyTorch Model When:
- ✓ TensorRT not available
- ✓ Development/debugging
- ✓ Need highest accuracy (slight edge over INT8)
- ✓ Cross-platform compatibility needed

## Rebuilding TensorRT Engine

If you need to rebuild the TensorRT engine for your specific GPU:

```bash
# Export PyTorch model to ONNX
python -c "from ultralytics import YOLO; YOLO('models/yolo26l_best.pt').export(format='onnx')"

# Convert ONNX to TensorRT (using trtexec)
trtexec --onnx=models/yolo26l_best.onnx \
        --saveEngine=models/yolo26l_int8_bs16.engine \
        --int8 \
        --batch=16 \
        --workspace=4096
```

## Expected Performance on RTX 4080

| Model | Resolution | Expected FPS | Latency |
|-------|-----------|--------------|---------|
| TRT INT8 | 1080p | 60+ FPS | ~1.5 ms |
| TRT INT8 | 2K | 35-40 FPS | ~2.0 ms |
| PyTorch FP16 | 1080p | 40-50 FPS | ~3.0 ms |
| PyTorch FP16 | 2K | 20-25 FPS | ~5.0 ms |

## Notes

- Model files are not tracked in git due to their size
- Download models from the releases page or train your own
- TensorRT engines are GPU-specific (rebuild if deploying to different GPU architecture)
- PyTorch model works across all CUDA-capable GPUs
