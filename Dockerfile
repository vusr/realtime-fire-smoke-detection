# Fire and Smoke Detection - Docker Image
# Optimized for RTX 4080 GPU with CUDA 12.x and TensorRT support

# Use NVIDIA PyTorch base image with CUDA 12.1 and TensorRT
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference scripts
COPY inference_tensorrt.py /app/
COPY inference_pytorch.py /app/

# Create directories for models and data
RUN mkdir -p /app/models /data/videos /data/output

# Copy model files (these should be added during build or mounted at runtime)
COPY models/ /app/models/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY

# Default command shows help
CMD ["python", "inference_tensorrt.py", "--help"]

# To run inference, override CMD:
# docker run --gpus all -v /path/to/videos:/data/videos -v /path/to/output:/data/output \
#   fire-detection python inference_tensorrt.py --video /data/videos/test.mp4 --output /data/output/
