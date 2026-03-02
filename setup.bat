@echo off
REM Setup script for fire detection project (Windows)

echo ==================================
echo Fire Detection Setup (Windows)
echo ==================================

REM Check Python version
echo.
echo [1/5] Checking Python version...
python --version

REM Check CUDA availability
echo.
echo [2/5] Checking CUDA availability...
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
) else (
    echo NVIDIA GPU not detected or nvidia-smi not available
    echo Note: CPU inference will be very slow
)

REM Install Python dependencies
echo.
echo [3/5] Installing Python dependencies...
echo This may take a few minutes...

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM Install other dependencies
echo Installing other dependencies...
pip install -r requirements.txt

REM Verify installations
echo.
echo [4/5] Verifying installations...
python -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import cv2; print(f'✓ OpenCV {cv2.__version__} installed')"
python -c "import ultralytics; print('✓ Ultralytics installed')"

REM Check TensorRT
echo.
echo Checking TensorRT...
python -c "import tensorrt" 2>nul
if %ERRORLEVEL% EQU 0 (
    python -c "import tensorrt as trt; print(f'✓ TensorRT {trt.__version__} installed')"
) else (
    echo ✗ TensorRT not installed
    echo   For TensorRT inference, install from: https://developer.nvidia.com/tensorrt
    echo   Or run: pip install nvidia-tensorrt
)

REM Check model files
echo.
echo [5/5] Checking model files...
if exist "models\yolo26l_best.pt" (
    echo ✓ PyTorch model found
) else (
    echo ✗ PyTorch model not found: models\yolo26l_best.pt
)

if exist "models\yolo26l_int8_bs16.engine" (
    echo ✓ TensorRT model found
) else (
    echo ✗ TensorRT model not found: models\yolo26l_int8_bs16.engine
    echo   This is expected if running on a different GPU
    echo   The engine will need to be rebuilt for your specific GPU
)

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo Quick Start:
echo   1. Place your video in a known location
echo   2. Run inference:
echo      python inference_pytorch.py --video C:\path\to\video.mp4 --output results\
echo      OR (if TensorRT available):
echo      python inference_tensorrt.py --video C:\path\to\video.mp4 --output results\
echo.

pause
