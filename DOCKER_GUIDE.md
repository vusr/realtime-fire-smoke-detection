# Docker Deployment Guide

Complete guide for deploying the fire detection system using Docker on RTX 4080 or similar GPUs.

## Prerequisites

### 1. Install Docker

**Ubuntu/Debian:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**Windows:**
- Download Docker Desktop from https://www.docker.com/products/docker-desktop
- Install WSL2 backend
- Enable GPU support in Docker Desktop settings

### 2. Install NVIDIA Docker Runtime

Required for GPU access in containers.

**Ubuntu/Debian:**
```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

### 3. Verify GPU Access

Test that Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your RTX 4080 listed.

## Building the Docker Image

### Method 1: Build from Dockerfile

```bash
# Navigate to project directory
cd /path/to/fire-detection-project

# Build the image (this will take 10-15 minutes)
docker build -t fire-detection:latest .

# Check image size
docker images fire-detection
```

### Method 2: Using Docker Compose

```bash
docker-compose build
```

## Running Inference

### Option 1: Command Line

**Basic usage:**
```bash
docker run --gpus all \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  fire-detection:latest \
  python inference_tensorrt.py \
    --video /data/videos/test_video.mp4 \
    --output /data/output/
```

**With custom parameters:**
```bash
docker run --gpus all \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  fire-detection:latest \
  python inference_tensorrt.py \
    --video /data/videos/dashcam.mp4 \
    --output /data/output/dashcam_results/ \
    --conf-thresh 0.3 \
    --batch-size 16 \
    --skip-frames 2
```

**Using PyTorch instead of TensorRT:**
```bash
docker run --gpus all \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  fire-detection:latest \
  python inference_pytorch.py \
    --video /data/videos/test_video.mp4 \
    --output /data/output/
```

### Option 2: Docker Compose

Edit `docker-compose.yml` to specify your video and parameters, then:

```bash
# Run once
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 3: Interactive Mode

For debugging or batch processing multiple videos:

```bash
# Start container with shell access
docker run --gpus all -it \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  fire-detection:latest \
  /bin/bash

# Inside container, run inference on multiple videos
python inference_tensorrt.py --video /data/videos/video1.mp4 --output /data/output/video1/
python inference_tensorrt.py --video /data/videos/video2.mp4 --output /data/output/video2/

# Exit
exit
```

## Volume Mounting

### Input Videos (Read-Only)
```bash
-v /host/path/to/videos:/data/videos:ro
```

The `:ro` flag mounts as read-only for safety.

### Output Results
```bash
-v /host/path/to/output:/data/output
```

Results will be written to the host system.

### Custom Model Files (Optional)
```bash
-v /host/path/to/models:/app/models
```

Override built-in models with your own.

## Performance Optimization

### For RTX 4080

**Recommended settings for 2K video:**
```bash
docker run --gpus all \
  --shm-size=2g \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  fire-detection:latest \
  python inference_tensorrt.py \
    --video /data/videos/2k_video.mp4 \
    --output /data/output/ \
    --batch-size 16 \
    --conf-thresh 0.25
```

**For 4K video or limited VRAM:**
```bash
docker run --gpus all \
  --shm-size=2g \
  -v /path/to/videos:/data/videos:ro \
  -v /path/to/output:/data/output \
  fire-detection:latest \
  python inference_tensorrt.py \
    --video /data/videos/4k_video.mp4 \
    --output /data/output/ \
    --batch-size 8 \
    --skip-frames 2
```

### Memory Settings

Increase shared memory if processing large batches:
```bash
--shm-size=4g  # 4 GB shared memory
```

### GPU Selection

If you have multiple GPUs:
```bash
# Use specific GPU
docker run --gpus '"device=0"' ...

# Use multiple GPUs
docker run --gpus '"device=0,1"' ...
```

## Batch Processing

Process multiple videos automatically:

**Linux/Mac:**
```bash
#!/bin/bash
for video in /path/to/videos/*.mp4; do
  basename=$(basename "$video" .mp4)
  echo "Processing $basename..."
  
  docker run --gpus all \
    -v /path/to/videos:/data/videos:ro \
    -v /path/to/output:/data/output \
    fire-detection:latest \
    python inference_tensorrt.py \
      --video "/data/videos/$(basename "$video")" \
      --output "/data/output/$basename/"
done
```

**Windows PowerShell:**
```powershell
Get-ChildItem C:\path\to\videos\*.mp4 | ForEach-Object {
    $basename = $_.BaseName
    Write-Host "Processing $basename..."
    
    docker run --gpus all `
      -v C:\path\to\videos:/data/videos:ro `
      -v C:\path\to\output:/data/output `
      fire-detection:latest `
      python inference_tensorrt.py `
        --video "/data/videos/$($_.Name)" `
        --output "/data/output/$basename/"
}
```

## Troubleshooting

### Issue: "docker: Error response from daemon: could not select device driver"

**Solution:** Install nvidia-docker2 runtime:
```bash
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size:
```bash
--batch-size 8  # or even 4
```

### Issue: TensorRT engine not compatible

**Solution:** The TensorRT engine is GPU-specific. Rebuild inside container:

```bash
# Start interactive container
docker run --gpus all -it fire-detection:latest /bin/bash

# Inside container, rebuild engine
cd /app
python -c "
from ultralytics import YOLO
model = YOLO('models/yolo26l_best.pt')
model.export(format='engine', batch=16, int8=True, workspace=4)
"

# Copy out the new engine
exit

# Copy from container to host
docker cp <container_id>:/app/models/yolo26l_best.engine ./models/
```

### Issue: Slow inference performance

**Checklist:**
1. Verify GPU is being used: Check `nvidia-smi` while running
2. Use TensorRT instead of PyTorch
3. Increase batch size if memory allows
4. Check if CPU is bottleneck (reduce video I/O)
5. Ensure you're not using `--skip-frames` unless needed

### Issue: Video codec not supported

**Solution:** Install additional codecs in container:

Create custom Dockerfile:
```dockerfile
FROM fire-detection:latest

# Install additional codecs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*
```

Build:
```bash
docker build -t fire-detection:codecs -f Dockerfile.codecs .
```

## Production Deployment

### Best Practices

1. **Pin versions** in requirements.txt
2. **Use specific image tags** instead of `:latest`
3. **Health checks**:
   ```yaml
   healthcheck:
     test: ["CMD", "python", "-c", "import torch; assert torch.cuda.is_available()"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```
4. **Resource limits**:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 8G
       reservations:
         memory: 4G
   ```
5. **Logging**:
   ```bash
   docker run --log-driver json-file --log-opt max-size=10m ...
   ```

### Kubernetes Deployment

For production-scale deployment, see `kubernetes/` directory (if available) or contact maintainers.

## Cleanup

Remove containers and images:

```bash
# Stop all containers
docker stop $(docker ps -aq)

# Remove containers
docker rm $(docker ps -aq)

# Remove image
docker rmi fire-detection:latest

# Clean up system
docker system prune -a
```

## Support

For issues or questions:
- Check GitHub Issues
- Review logs: `docker logs <container-id>`
- Verify GPU access: `nvidia-smi`
- Check Docker GPU access: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
