# Fire and Smoke Detection - Docker Image
# Optimized for NVIDIA L4 GPU with CUDA 13.0 and TensorRT support

# Use NVIDIA PyTorch base image with CUDA 13.0
FROM nvcr.io/nvidia/pytorch:25.01-py3

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
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy inference scripts
COPY inference_tensorrt.py /app/
COPY inference_pytorch.py /app/

# Create directories for models and data
RUN mkdir -p /app/models /data/videos /data/output

# Copy model files (these should be added during build or mounted at runtime)
COPY models/ /app/models/

# Set environment variables for CUDA 13.0
ENV PYTHONUNBUFFERED=1
ENV CUDA_MODULE_LOADING=LAZY
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH}

# Default command shows help
CMD ["python", "inference_tensorrt.py", "--help"]

# To run inference, override CMD:
# docker run --gpus all -v /path/to/videos:/data/videos -v /path/to/output:/data/output \
#   fire-detection python inference_tensorrt.py --video /data/videos/test.mp4 --output /data/output/
