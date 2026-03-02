# Quick Start Guide

Get started with fire and smoke detection in 5 minutes.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (RTX 4080 recommended)
- 4+ GB free disk space

## Installation

### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Inference

### Using PyTorch (Easiest)

```bash
python inference_pytorch.py \
  --video /path/to/your/video.mp4 \
  --output results/
```

### Using TensorRT (Fastest)

```bash
python inference_tensorrt.py \
  --video /path/to/your/video.mp4 \
  --output results/
```

## View Results

After processing, check the `results/` directory:

- `predictions.json` - Structured detection data
- `predictions.csv` - Tabular format for analysis
- `summary.txt` - Processing statistics

**Example output structure:**
```json
{
  "frame": 1523,
  "timestamp": 50.77,
  "objects": [
    {
      "class": "fire",
      "confidence": 0.89,
      "bbox": [345, 123, 567, 289]
    }
  ]
}
```

## Common Use Cases

### 1. Process Single Video

```bash
python inference_pytorch.py --video dashcam.mp4 --output results/dashcam/
```

### 2. High Confidence Detections Only

```bash
python inference_tensorrt.py \
  --video video.mp4 \
  --conf-thresh 0.5 \
  --output results/
```

### 3. Fast Processing (Skip Frames)

For long videos, process every 2nd frame:

```bash
python inference_tensorrt.py \
  --video long_video.mp4 \
  --skip-frames 2 \
  --output results/
```

### 4. Batch Process Multiple Videos

**Linux/Mac:**
```bash
for video in videos/*.mp4; do
  python inference_tensorrt.py \
    --video "$video" \
    --output "results/$(basename $video .mp4)/"
done
```

**Windows:**
```powershell
Get-ChildItem videos\*.mp4 | ForEach-Object {
  python inference_tensorrt.py --video $_.FullName --output "results\$($_.BaseName)\"
}
```

## Docker Quick Start

### Build Image

```bash
docker build -t fire-detection .
```

### Run Inference

```bash
docker run --gpus all \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/results:/data/output \
  fire-detection \
  python inference_tensorrt.py \
    --video /data/videos/test.mp4 \
    --output /data/output/
```

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for detailed Docker usage.

## Performance Tips

### For 2K Video @ 30 FPS:
- Use TensorRT model: `inference_tensorrt.py`
- Batch size: 16 (default)
- Expected: 35-40 FPS on RTX 4080

### For 4K Video:
- Reduce batch size: `--batch-size 8`
- Or skip frames: `--skip-frames 2`

### If Running Out of Memory:
```bash
python inference_tensorrt.py \
  --video video.mp4 \
  --batch-size 4 \
  --output results/
```

## Troubleshooting

### CUDA out of memory
- Reduce batch size: `--batch-size 4`
- Close other GPU applications

### TensorRT not found
- Use PyTorch inference instead: `inference_pytorch.py`
- Or install TensorRT: `pip install nvidia-tensorrt`

### Slow performance
- Check GPU is being used: `nvidia-smi`
- Use TensorRT instead of PyTorch (3× faster)
- Ensure CUDA drivers are up to date

### Model file not found
- Check `models/` directory exists
- Verify model files are present:
  - `models/yolo26l_best.pt` (PyTorch)
  - `models/yolo26l_int8_bs16.engine` (TensorRT)

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for production deployment
- Review evaluation results in `YOLO26l/YOLO26l_Evaluation/`

## Support

For issues:
1. Check `nvidia-smi` shows your GPU
2. Verify Python version: `python --version` (need 3.10+)
3. Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
4. Review error logs in console output

## Performance Expectations

| GPU | Resolution | FPS (TensorRT) | FPS (PyTorch) |
|-----|-----------|----------------|---------------|
| RTX 4080 | 1080p | 60+ | 40-50 |
| RTX 4080 | 2K | 35-40 | 20-25 |
| RTX 4080 | 4K | 15-20 | 8-12 |
| RTX 3080 | 1080p | 50+ | 35-40 |
| RTX 3080 | 2K | 30-35 | 18-22 |

## Example Output

```
Processing with batch_size=16, skip_frames=1
Processing: 100%|████████████| 18000/18000 [07:50<00:00, 38.3 frame/s]

==================================
Processing Complete!
==================================
Total Detections: 127
  Fire: 83
  Smoke: 44
Frames with Detections: 89
Average Processing FPS: 38.3
==================================
```

Happy detecting! 🔥
