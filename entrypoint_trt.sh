#!/bin/bash
# entrypoint_trt.sh
# =================
# Step 1: Build the TensorRT INT8 engine from the PyTorch .pt model.
#         Skips the build automatically if the engine file already exists
#         (volume-persisted cache), so subsequent container starts are fast.
#
# Step 2: Run inference_tensorrt.py using the freshly-built (or cached) engine.
#         All arguments passed to this script are forwarded to the inference
#         script, e.g.:
#           --video  /data/videos/test.mp4
#           --output /data/output/
#           --conf-thresh 0.3
#           --batch-size 16

set -e

echo ""
echo "============================================================"
echo "  Fire & Smoke Detection — TRT Build + Inference Pipeline"
echo "============================================================"
echo ""

# ── Step 1: Build TRT INT8 engine ────────────────────────────────────────────
echo "[Step 1/2] Building TensorRT INT8 engine from PyTorch model ..."
echo ""
python /app/build_engine.py

echo ""
echo "──────────────────────────────────────────────────────────"
echo ""

# ── Step 2: Run inference ─────────────────────────────────────────────────────
# Use ENGINE_PATH env var (set in Dockerfile / docker-compose) as --model.
# All extra arguments forwarded from CMD are appended here.
RESOLVED_ENGINE="${ENGINE_PATH:-/app/engines/yolo26l_int8_bs16.engine}"

echo "[Step 2/2] Starting TensorRT inference ..."
echo "           Engine : ${RESOLVED_ENGINE}"
echo ""

exec python /app/inference_tensorrt.py \
    --model "${RESOLVED_ENGINE}" \
    "$@"
