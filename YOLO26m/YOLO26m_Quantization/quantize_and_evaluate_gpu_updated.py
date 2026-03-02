#!/usr/bin/env python3
"""
quantize_and_evaluate_gpu.py
=============================
GPU-optimised quantization and evaluation pipeline for the YOLO fire/smoke
detection model on a Google Cloud VM with an NVIDIA L4 GPU (Ada Lovelace,
compute capability 8.9, ~24 GB VRAM — script configured conservatively for
12 GB available).

Precision modes evaluated
--------------------------
  [1] PyTorch  FP32   – baseline, native CUDA
  [2] PyTorch  FP16   – half-precision Tensor Core inference
  [3] TensorRT FP16   – TRT engine with layer fusion + kernel tuning (batch=16)
  [4] TensorRT FP16   – TRT engine with layer fusion + kernel tuning (batch=32)
  [5] TensorRT FP16   – TRT engine with layer fusion + kernel tuning (batch=64)
  [6] TensorRT INT8   – TRT engine with PTQ calibration (batch=16)
  [7] TensorRT INT8   – TRT engine with PTQ calibration (batch=32)
  [8] TensorRT INT8   – TRT engine with PTQ calibration (batch=64)

L4 GPU precision capabilities (Ada Lovelace / CC 8.9)
-------------------------------------------------------
  ✓ FP32, FP16, BF16  – CUDA cores + 4th-gen Tensor Cores
  ✓ INT8              – Tensor Core INT8 (≈4× INT8 TOPS vs FP32 TFLOPS)

Saved artefacts (all under optimized_models/)
----------------------------------------------
  yolo26m_best_fp16_bs16.engine   TensorRT FP16 engine (batch=16)
  yolo26m_best_fp16_bs32.engine   TensorRT FP16 engine (batch=32)
  yolo26m_best_fp16_bs64.engine   TensorRT FP16 engine (batch=64)
  yolo26m_best_int8_bs16.engine   TensorRT INT8 engine (batch=16, with calibration)
  yolo26m_best_int8_bs32.engine   TensorRT INT8 engine (batch=32, with calibration)
  yolo26m_best_int8_bs64.engine   TensorRT INT8 engine (batch=64, with calibration)
  quantization_report.txt         Final comparison report

Required packages — see INSTALL COMMANDS at the bottom of this file.

Usage
-----
    python quantize_and_evaluate_gpu.py
"""

import gc
import math
import shutil
import sys
import warnings
from pathlib import Path
from datetime import datetime

import yaml
import torch

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH    = "yolo26m_best.pt"   # trained PyTorch weights
TEST_DIR      = "/home/smoon23/data/test"
TEMP_TEST_DIR = "/home/smoon23/data/test_eval_temp"  # temporary test directory
DATASET_YAML  = "dataset.yaml"      # written by this script; safe to delete
OUTPUT_DIR    = "optimized_models_yolo26m"  # all exported model files land here
REPORT_FILE   = "quantization_report.txt"   # saved comparison report

IMG_SIZE      = 640
# Batch sizes for PyTorch validation
BATCH_PT      = 64    # PyTorch FP32 / FP16 validation batches
# TensorRT batch sizes - we'll create separate engines for each
TRT_BATCH_SIZES = [16, 32, 64]
# TRT workspace: memory (GiB) TRT may use during engine build.
# Set to None for auto; use a fixed value to stay within 12 GB budget.
TRT_WORKSPACE = 4

DEVICE_GPU    = "0"   # CUDA device index for PyTorch + TRT

# Fraction of the test set used for TRT INT8 calibration (1.0 = full set).
# Reduce to speed up calibration; 0.1–0.2 is often sufficient.
TRT_CALIB_FRACTION = 1.0
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "Precision", "Recall", "mAP50", "mAP50:95",
    "Latency (ms)", "Throughput (img/s)",
]


# ─── Setup ────────────────────────────────────────────────────────────────────

def is_valid_label(label_path: Path) -> bool:
    """
    Check if a label file has valid normalized coordinates.
    Returns False if any coordinate is outside [0, 1] range.
    """
    if not label_path.exists():
        return True  # No label file is okay (background image)
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height
                    coords = [float(x) for x in parts[1:5]]
                    # Check if all coordinates are normalized (0 to 1)
                    if any(c < 0 or c > 1 for c in coords):
                        return False
    except Exception:
        return False
    
    return True


def create_temp_test_set() -> int:
    """
    Create a temporary test directory with exactly a multiple of 64 VALID images.
    Filters out images with corrupt/invalid labels.
    Returns the number of images copied.
    """
    max_batch = max(TRT_BATCH_SIZES)
    
    # Get all images and labels
    images_dir = Path(TEST_DIR) / "images"
    labels_dir = Path(TEST_DIR) / "labels"
    
    all_images = sorted(images_dir.glob("*.jpg"))
    
    print(f"[i] Found {len(all_images)} total images in test set")
    print(f"[→] Filtering out images with corrupt/invalid labels...")
    
    # Filter valid images only
    valid_images = []
    skipped = []
    
    for img_path in all_images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if is_valid_label(label_path):
            valid_images.append(img_path)
        else:
            skipped.append(img_path.name)
    
    if skipped:
        print(f"[!] Skipped {len(skipped)} images with invalid labels:")
        for name in skipped:
            print(f"    - {name}")
    
    print(f"[✓] Found {len(valid_images)} valid images")
    
    # Calculate how many images to use (multiple of 64)
    n_valid = len(valid_images)
    n_target = (n_valid // max_batch) * max_batch
    
    if n_target == 0:
        n_target = max_batch
    
    print(f"[i] Will use {n_target} images (multiple of {max_batch})")
    
    # Create temp directories
    temp_images_dir = Path(TEMP_TEST_DIR) / "images"
    temp_labels_dir = Path(TEMP_TEST_DIR) / "labels"
    
    # Clean up if exists
    if Path(TEMP_TEST_DIR).exists():
        print(f"[i] Removing existing temp directory: {TEMP_TEST_DIR}")
        shutil.rmtree(TEMP_TEST_DIR)
    
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy first n_target valid images and their labels
    print(f"[→] Copying {n_target} valid images and labels to {TEMP_TEST_DIR} ...")
    
    copied = 0
    for img_path in valid_images[:n_target]:
        # Copy image
        shutil.copy2(img_path, temp_images_dir / img_path.name)
        
        # Copy corresponding label if it exists
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, temp_labels_dir / label_path.name)
        
        copied += 1
    
    print(f"[✓] Copied {copied} valid images and labels to temporary directory")
    return copied


def setup() -> int:
    """
    Setup directories, create temp test set, and create dataset YAML.
    Returns the number of test images.
    """
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create temporary test set with exact multiple of 64 valid images
    n_test = create_temp_test_set()
    
    # Create dataset YAML pointing to temp directory
    cfg = {
        "path": "/home/smoon23/data/",
        "val":  "test_eval_temp/images",
        "train": "train/images",
        "test": "test_eval_temp/images",
        "nc":   2,
        "names": {0: "smoke", 1: "fire"},
    }
    with open(DATASET_YAML, "w") as fh:
        yaml.dump(cfg, fh, sort_keys=False)
    
    print(f"[✓] {OUTPUT_DIR}/ ready")
    print(f"[✓] {DATASET_YAML} written (pointing to {TEMP_TEST_DIR})\n")
    
    return n_test


def cleanup() -> None:
    """Remove temporary test directory."""
    if Path(TEMP_TEST_DIR).exists():
        print(f"\n[→] Cleaning up temporary directory: {TEMP_TEST_DIR}")
        shutil.rmtree(TEMP_TEST_DIR)
        print(f"[✓] Cleanup complete")


def clear_gpu_memory() -> None:
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _file_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1e6
    except FileNotFoundError:
        return float("nan")


def _extract(results) -> dict:
    """Flatten an ultralytics ValResults object into a plain metrics dict."""
    box   = results.box
    speed = results.speed  # per-image ms: preprocess / inference / postprocess
    lat   = sum(speed.get(k, 0.0) for k in ("preprocess", "inference", "postprocess"))
    tput  = 1000.0 / lat if lat > 0 else float("nan")
    return {
        "Precision":          round(float(box.mp),    4),
        "Recall":             round(float(box.mr),    4),
        "mAP50":              round(float(box.map50), 4),
        "mAP50:95":           round(float(box.map),   4),
        "Latency (ms)":       round(lat,  2),
        "Throughput (img/s)": round(tput, 2),
    }


def _banner(step: str, label: str, device: str) -> None:
    print(f"\n{'─' * 66}")
    print(f"  {step}  {label}   [{device}]")
    print(f"{'─' * 66}")


def _safe_run(fn, label: str) -> dict:
    """Call fn(); return {} and print a warning if it raises."""
    try:
        result = fn()
        # Clear GPU memory after each evaluation
        clear_gpu_memory()
        return result
    except Exception as exc:
        print(f"\n  ⚠  {label} skipped: {exc}\n")
        import traceback
        traceback.print_exc()
        # Try to clear memory even on error
        clear_gpu_memory()
        return {}


# ─── [1] PyTorch FP32 ─────────────────────────────────────────────────────────

def eval_pytorch_fp32() -> dict:
    from ultralytics import YOLO
    _banner("[1/8]", "PyTorch FP32  (baseline)", f"cuda:{DEVICE_GPU}")
    model = YOLO(MODEL_PATH)
    r = model.val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_PT,
        device=DEVICE_GPU, half=False, verbose=False, plots=False,
    )
    result = _extract(r)
    # Explicitly delete model and clear memory
    del model
    return result


# ─── [2] PyTorch FP16 ─────────────────────────────────────────────────────────

def eval_pytorch_fp16() -> dict:
    from ultralytics import YOLO
    _banner("[2/8]", "PyTorch FP16  (half-precision)", f"cuda:{DEVICE_GPU}")
    model = YOLO(MODEL_PATH)
    r = model.val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_PT,
        device=DEVICE_GPU, half=True, verbose=False, plots=False,
    )
    result = _extract(r)
    # Explicitly delete model and clear memory
    del model
    return result


# ─── [3-8] TensorRT FP16 / INT8 (multiple batch sizes) ───────────────────────

def _trt_path(tag: str, batch_size: int) -> str:
    return str(Path(OUTPUT_DIR) / f"yolo26m_best_{tag}_bs{batch_size}.engine")


def _build_trt_engine(tag: str, batch_size: int, *, half: bool, int8: bool) -> str:
    """Build a TensorRT engine and move it to OUTPUT_DIR (idempotent)."""
    dest = _trt_path(tag, batch_size)
    if Path(dest).exists():
        print(f"  [✓] TRT {tag.upper()} (batch={batch_size}) engine already present: {dest}")
        return dest

    from ultralytics import YOLO
    print(f"\n  [→] Building TensorRT {tag.upper()} engine (batch={batch_size}) "
          f"(layer fusion + kernel tuning — may take several minutes) …")

    kwargs: dict = dict(
        format="engine",
        imgsz=IMG_SIZE,
        batch=batch_size,
        device=DEVICE_GPU,
        half=half,
        int8=int8,
        workspace=TRT_WORKSPACE,
        verbose=False,
    )
    if int8:
        kwargs["data"]     = DATASET_YAML
        kwargs["fraction"] = TRT_CALIB_FRACTION

    model = YOLO(MODEL_PATH)
    exported = Path(str(model.export(**kwargs)))
    shutil.move(str(exported), dest)
    print(f"  [✓] Saved → {dest}  ({_file_mb(dest):.1f} MB)")
    
    # Clean up model and clear memory after export
    del model
    clear_gpu_memory()
    
    return dest


def eval_trt_fp16(batch_size: int, step: str) -> dict:
    from ultralytics import YOLO
    _banner(step, f"TensorRT FP16  (fused engine, batch={batch_size})", f"cuda:{DEVICE_GPU}")
    engine = _build_trt_engine("fp16", batch_size, half=True, int8=False)
    
    model = YOLO(engine)
    r = model.val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=batch_size,
        device=DEVICE_GPU, verbose=False, plots=False,
    )
    result = _extract(r)
    
    # Explicitly delete model and clear memory
    del model
    return result


def eval_trt_int8(batch_size: int, step: str) -> dict:
    from ultralytics import YOLO
    _banner(step, f"TensorRT INT8  (PTQ calibrated, batch={batch_size})", f"cuda:{DEVICE_GPU}")
    engine = _build_trt_engine("int8", batch_size, half=False, int8=True)
    
    model = YOLO(engine)
    r = model.val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=batch_size,
        device=DEVICE_GPU, verbose=False, plots=False,
    )
    result = _extract(r)
    
    # Explicitly delete model and clear memory
    del model
    return result


# ─── Reporting ────────────────────────────────────────────────────────────────

def _pct(new_val: float, base_val: float) -> str:
    if math.isnan(new_val) or math.isnan(base_val) or base_val == 0:
        return "N/A"
    return f"{(new_val - base_val) / abs(base_val) * 100:>+.2f}%"


def print_report(metrics: dict[str, dict], sizes: dict[str, float], n_test: int) -> str:
    """Print and return the report as a string."""
    col_w   = 17
    name_w  = 16
    # columns: Device | Size (MB) | Precision | Recall | mAP50 | mAP50:95 | Latency | Throughput
    all_cols = ["Device", "Size (MB)"] + METRIC_COLS
    total_w  = name_w + col_w * len(all_cols)
    sep      = "─" * total_w
    
    lines = []
    
    lines.append("\n\n" + "═" * total_w)
    lines.append(f"  QUANTIZATION COMPARISON — Fire & Smoke Detection (YOLO26M) — NVIDIA L4 GPU")
    lines.append(f"  Test Set: {n_test} images | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("═" * total_w)
    
    hdr = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in all_cols)
    lines.append(hdr)
    lines.append(sep)

    # Device labels for each configuration
    device_label = {k: f"cuda:{DEVICE_GPU}" for k in metrics.keys()}

    for name, m in metrics.items():
        dev = device_label.get(name, "?")
        sz  = sizes.get(name, float("nan"))
        if not m:
            lines.append(f"  {name:<{name_w - 2}}  (skipped — see warning above)")
            continue
        row = f"{name:<{name_w}}{dev:>{col_w}}{sz:>{col_w}.1f}"
        for col in METRIC_COLS:
            val = m.get(col, float("nan"))
            row += (
                f"{val:>{col_w}.2f}" if ("Latency" in col or "Throughput" in col)
                else f"{val:>{col_w}.4f}"
            )
        lines.append(row)

    lines.append(sep)

    # Delta table versus PyTorch FP32 baseline
    base   = metrics.get("PT-FP32", {})
    base_sz = sizes.get("PT-FP32", float("nan"))
    if base and any(k != "PT-FP32" and v for k, v in metrics.items()):
        lines.append(f"\n  Δ relative to PT-FP32 baseline  (+) = improvement for accuracy / throughput")
        lines.append(sep)
        for name, m in metrics.items():
            if name == "PT-FP32" or not m:
                continue
            sz   = sizes.get(name, float("nan"))
            sz_d = _pct(sz, base_sz)
            row  = f"{name:<{name_w}}{'':>{col_w}}{sz_d:>{col_w}}"
            for col in METRIC_COLS:
                row += f"{_pct(m.get(col, float('nan')), base.get(col, float('nan'))):>{col_w}}"
            lines.append(row)
        lines.append("═" * total_w)

    # Human-readable summary
    lines.append(f"\n  Summary")
    lines.append(sep)
    for name, m in metrics.items():
        if not m:
            continue
        lines.append(
            f"  {name:<15}  "
            f"mAP50:95={m.get('mAP50:95', float('nan')):.4f}  "
            f"latency={m.get('Latency (ms)', float('nan')):.1f} ms  "
            f"throughput={m.get('Throughput (img/s)', float('nan')):.1f} img/s  "
            f"size={sizes.get(name, float('nan')):.1f} MB  "
            f"device={device_label.get(name, '?')}"
        )
    lines.append("═" * total_w + "\n")
    
    report_text = "\n".join(lines)
    print(report_text)
    return report_text


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    banner = "YOLO GPU Quantization Evaluation — Fire/Smoke (NVIDIA L4, CC 8.9)"
    print("╔" + "═" * (len(banner) + 4) + "╗")
    print(f"║  {banner}  ║")
    print("╚" + "═" * (len(banner) + 4) + "╝")
    print(f"  Model  : {MODEL_PATH}  ({_file_mb(MODEL_PATH):.1f} MB)")
    print(f"  Test   : {TEST_DIR}/images/")
    print(f"  Output : {OUTPUT_DIR}/")
    print(f"  Report : {REPORT_FILE}\n")

    try:
        n_test = setup()

        metrics: dict[str, dict] = {}
        sizes:   dict[str, float] = {}

        # ── PyTorch ───────────────────────────────────────────────────────────────
        metrics["PT-FP32"] = _safe_run(eval_pytorch_fp32,  "PyTorch FP32")
        sizes["PT-FP32"]   = _file_mb(MODEL_PATH)

        metrics["PT-FP16"] = _safe_run(eval_pytorch_fp16,  "PyTorch FP16")
        sizes["PT-FP16"]   = _file_mb(MODEL_PATH)   # same .pt file, different compute

        # ── TensorRT FP16 (multiple batch sizes) ─────────────────────────────────
        step_idx = 3
        for batch_size in TRT_BATCH_SIZES:
            key = f"TRT-FP16-B{batch_size}"
            metrics[key] = _safe_run(
                lambda bs=batch_size, s=step_idx: eval_trt_fp16(bs, f"[{s}/8]"),
                f"TensorRT FP16 (batch={batch_size})"
            )
            sizes[key] = _file_mb(_trt_path("fp16", batch_size))
            step_idx += 1

        # ── TensorRT INT8 (multiple batch sizes) ─────────────────────────────────
        for batch_size in TRT_BATCH_SIZES:
            key = f"TRT-INT8-B{batch_size}"
            metrics[key] = _safe_run(
                lambda bs=batch_size, s=step_idx: eval_trt_int8(bs, f"[{s}/8]"),
                f"TensorRT INT8 (batch={batch_size})"
            )
            sizes[key] = _file_mb(_trt_path("int8", batch_size))
            step_idx += 1

        # Generate and save report
        report_text = print_report(metrics, sizes, n_test)
        
        report_path = Path(OUTPUT_DIR) / REPORT_FILE
        with open(report_path, "w") as f:
            f.write(report_text)
        print(f"[✓] Report saved to: {report_path}")
    
    finally:
        # Always cleanup temp directory and clear GPU memory
        cleanup()
        clear_gpu_memory()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()


# =============================================================================
# INSTALL COMMANDS FOR GOOGLE CLOUD VM (Ubuntu 22.04 + CUDA 12.x)
# =============================================================================
#
# 1. System-level CUDA / cuDNN (usually pre-installed on Deep Learning VMs)
#    If not present:
#      sudo apt-get install -y cuda-toolkit-12-x libcudnn8
#
# 2. Python packages
#    pip install --upgrade pip
#    pip install ultralytics          # YOLO model + export + val pipeline
#    pip install pyyaml               # dataset YAML generation
#
# 3. TensorRT  (needed for format="engine" export)
#    TensorRT ships with NVIDIA Deep Learning VM images on GCP.
#    If not present, install via pip (matches your CUDA/TRT version):
#      pip install tensorrt           # installs TRT Python bindings
#    Or via NVIDIA's apt repo:
#      sudo apt-get install -y tensorrt
#    Verify: python -c "import tensorrt; print(tensorrt.__version__)"
#
# 4. Sanity check
#    python -c "
#    import torch, ultralytics, tensorrt
#    print('torch       :', torch.__version__, '| CUDA:', torch.cuda.is_available())
#    print('ultralytics :', ultralytics.__version__)
#    print('tensorrt    :', tensorrt.__version__)
#    "
# =============================================================================
