#!/bin/bash
# Setup script for fire detection project

echo "=================================="
echo "Fire Detection Setup"
echo "=================================="

# Check Python version
echo -e "\n[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA availability
echo -e "\n[2/5] Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "✗ NVIDIA GPU not detected or nvidia-smi not available"
    echo "  Note: CPU inference will be very slow"
fi

# Install Python dependencies
echo -e "\n[3/5] Installing Python dependencies..."
echo "This may take a few minutes..."

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing other dependencies..."
pip3 install -r requirements.txt

# Verify installations
echo -e "\n[4/5] Verifying installations..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 
python3 -c "import cv2; print(f'✓ OpenCV {cv2.__version__} installed')"
python3 -c "import ultralytics; print(f'✓ Ultralytics installed')"

# Check TensorRT
echo -e "\nChecking TensorRT..."
if python3 -c "import tensorrt" 2>/dev/null; then
    python3 -c "import tensorrt as trt; print(f'✓ TensorRT {trt.__version__} installed')"
else
    echo "✗ TensorRT not installed"
    echo "  For TensorRT inference, install from: https://developer.nvidia.com/tensorrt"
    echo "  Or run: pip3 install nvidia-tensorrt"
fi

# Check model files
echo -e "\n[5/5] Checking model files..."
if [ -f "models/yolo26l_best.pt" ]; then
    echo "✓ PyTorch model found"
else
    echo "✗ PyTorch model not found: models/yolo26l_best.pt"
fi

if [ -f "models/yolo26l_int8_bs16.engine" ]; then
    echo "✓ TensorRT model found"
else
    echo "✗ TensorRT model not found: models/yolo26l_int8_bs16.engine"
    echo "  This is expected if running on a different GPU"
    echo "  The engine will need to be rebuilt for your specific GPU"
fi

echo -e "\n=================================="
echo "Setup Complete!"
echo "=================================="
echo -e "\nQuick Start:"
echo "  1. Place your video in a known location"
echo "  2. Run inference:"
echo "     python3 inference_pytorch.py --video /path/to/video.mp4 --output results/"
echo "     OR (if TensorRT available):"
echo "     python3 inference_tensorrt.py --video /path/to/video.mp4 --output results/"
echo ""
