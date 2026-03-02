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
  [3] TensorRT FP16   – TRT engine with layer fusion + kernel tuning
  [4] TensorRT INT8   – TRT engine with PTQ calibration (~2-4× FP32 speed)
  [5] ONNX     FP32   – ONNX Runtime with CUDA Execution Provider
  [6] ONNX     INT8   – dynamic INT8 weight quantisation, ONNX Runtime (CPU)
  [7] TFLite   FP16   – TFLite float16 (CPU, cross-platform baseline)
  [8] TFLite   INT8   – TFLite dynamic-range INT8 (CPU)

L4 GPU precision capabilities (Ada Lovelace / CC 8.9)
-------------------------------------------------------
  ✓ FP32, FP16, BF16  – CUDA cores + 4th-gen Tensor Cores
  ✓ INT8              – Tensor Core INT8 (≈4× INT8 TOPS vs FP32 TFLOPS)
  ✓ FP8 (E4M3)        – Ada 4th-gen Tensor Cores (TRT ≥ 9.0 only)
  ✗ FP8 via Ultralytics – NOT yet exposed in ultralytics export API; use
                          TensorRT-ModelOptimizer for manual FP8 PTQ if needed.

Saved artefacts (all under optimized_models/)
----------------------------------------------
  yolo26l_best_fp16.engine   TensorRT FP16 engine
  yolo26l_best_int8.engine   TensorRT INT8 engine (with calibration)
  yolo26l_best_fp32.onnx     ONNX FP32 (CUDA EP ready)
  yolo26l_best_int8.onnx     ONNX dynamic-INT8 (CPU)
  yolo26l_best_fp16.tflite   TFLite FP16 (CPU)
  yolo26l_best_int8.tflite   TFLite dynamic-range INT8 (CPU)

Required packages — see INSTALL COMMANDS at the bottom of this file.

Usage
-----
    python quantize_and_evaluate_gpu.py
"""

import math
import shutil
import sys
import warnings
from pathlib import Path

import yaml

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH    = "yolo26l_best.pt"   # trained PyTorch weights
TEST_DIR      = "/home/smoon23/data/test"
DATASET_YAML  = "dataset.yaml"      # written by this script; safe to delete
OUTPUT_DIR    = "optimized_models_yolo26l"  # all exported model files land here

IMG_SIZE      = 640
# Batch sizes – tune based on actual VRAM; conservative defaults for 12 GB
BATCH_PT      = 64    # PyTorch FP32 / FP16 validation batches
BATCH_TRT     = 64    # TRT engine batch (built and validated at this size)
BATCH_ONNX    = 64    # ONNX Runtime batch (CUDA EP for FP32, CPU for INT8)
BATCH_TFLITE  = 1     # TFLite interpreter; always single-image

DEVICE_GPU    = "0"   # CUDA device index for PyTorch + TRT + ONNX FP32
# TRT workspace: memory (GiB) TRT may use during engine build.
# Set to None for auto; use a fixed value to stay within 12 GB budget.
TRT_WORKSPACE = 4

# Fraction of the test set used for TRT INT8 calibration (1.0 = full set).
# Reduce to speed up calibration; 0.1–0.2 is often sufficient.
TRT_CALIB_FRACTION = 1.0
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "Precision", "Recall", "mAP50", "mAP50:95",
    "Latency (ms)", "Throughput (img/s)",
]

# Inference device label shown in the report
DEVICE_LABEL = {
    "PT-FP32":   f"cuda:{DEVICE_GPU}",
    "PT-FP16":   f"cuda:{DEVICE_GPU}",
    "TRT-FP16":  f"cuda:{DEVICE_GPU}",
    "TRT-INT8":  f"cuda:{DEVICE_GPU}",
    "ONNX-FP32": f"cuda:{DEVICE_GPU}",
    "ONNX-INT8": "cpu",
    "TFL-FP16":  "cpu",
    "TFL-INT8":  "cpu",
}


# ─── Setup ────────────────────────────────────────────────────────────────────

def setup() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    cfg = {
        "path": "/home/smoon23/data/",
        "val":  "test/images",
        "train": "train/images",
        "test": "test/images",
        "nc":   2,
        "names": {0: "smoke", 1: "fire"},
    }
    with open(DATASET_YAML, "w") as fh:
        yaml.dump(cfg, fh, sort_keys=False)
    n = len(list(Path(TEST_DIR, "images").glob("*.jpg")))
    print(f"[✓] {OUTPUT_DIR}/ ready")
    print(f"[✓] {DATASET_YAML} written  ({n} test images)\n")


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
        return fn()
    except Exception as exc:
        print(f"\n  ⚠  {label} skipped: {exc}\n")
        return {}


# ─── [1] PyTorch FP32 ─────────────────────────────────────────────────────────

def eval_pytorch_fp32() -> dict:
    from ultralytics import YOLO
    _banner("[1/8]", "PyTorch FP32  (baseline)", f"cuda:{DEVICE_GPU}")
    r = YOLO(MODEL_PATH).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_PT,
        device=DEVICE_GPU, half=False, verbose=False, plots=False,
    )
    return _extract(r)


# ─── [2] PyTorch FP16 ─────────────────────────────────────────────────────────

def eval_pytorch_fp16() -> dict:
    from ultralytics import YOLO
    _banner("[2/8]", "PyTorch FP16  (half-precision)", f"cuda:{DEVICE_GPU}")
    r = YOLO(MODEL_PATH).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_PT,
        device=DEVICE_GPU, half=True, verbose=False, plots=False,
    )
    return _extract(r)


# ─── [3 & 4] TensorRT FP16 / INT8 ────────────────────────────────────────────

def _trt_path(tag: str) -> str:
    return str(Path(OUTPUT_DIR) / f"yolo26l_best_{tag}.engine")


def _build_trt_engine(tag: str, *, half: bool, int8: bool) -> str:
    """Build a TensorRT engine and move it to OUTPUT_DIR (idempotent)."""
    dest = _trt_path(tag)
    if Path(dest).exists():
        print(f"  [✓] TRT {tag.upper()} engine already present: {dest}")
        return dest

    from ultralytics import YOLO
    print(f"\n  [→] Building TensorRT {tag.upper()} engine "
          f"(layer fusion + kernel tuning — may take several minutes) …")

    kwargs: dict = dict(
        format="engine",
        imgsz=IMG_SIZE,
        batch=BATCH_TRT,
        device=DEVICE_GPU,
        half=half,
        int8=int8,
        workspace=TRT_WORKSPACE,
        verbose=False,
    )
    if int8:
        kwargs["data"]     = DATASET_YAML
        kwargs["fraction"] = TRT_CALIB_FRACTION

    exported = Path(str(YOLO(MODEL_PATH).export(**kwargs)))
    shutil.move(str(exported), dest)
    print(f"  [✓] Saved → {dest}  ({_file_mb(dest):.1f} MB)")
    return dest


def eval_trt_fp16() -> dict:
    from ultralytics import YOLO
    _banner("[3/8]", "TensorRT FP16  (fused engine)", f"cuda:{DEVICE_GPU}")
    engine = _build_trt_engine("fp16", half=True, int8=False)
    r = YOLO(engine).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_TRT,
        device=DEVICE_GPU, verbose=False, plots=False,
    )
    return _extract(r)


def eval_trt_int8() -> dict:
    from ultralytics import YOLO
    _banner("[4/8]", "TensorRT INT8  (PTQ calibrated engine)", f"cuda:{DEVICE_GPU}")
    engine = _build_trt_engine("int8", half=False, int8=True)
    r = YOLO(engine).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_TRT,
        device=DEVICE_GPU, verbose=False, plots=False,
    )
    return _extract(r)


# ─── [5] ONNX FP32 (CUDA EP) ─────────────────────────────────────────────────

def _onnx_fp32_path() -> str:
    return str(Path(OUTPUT_DIR) / "yolo26l_best_fp32.onnx")


def _export_onnx_fp32() -> str:
    dest = _onnx_fp32_path()
    if Path(dest).exists():
        print(f"  [✓] ONNX FP32 already present: {dest}")
        return dest

    from ultralytics import YOLO
    print(f"\n  [→] Exporting FP32 ONNX …")
    exported = Path(str(YOLO(MODEL_PATH).export(
        format="onnx", imgsz=IMG_SIZE, half=False,
        dynamic=False, simplify=True, verbose=False,
    )))
    shutil.move(str(exported), dest)
    print(f"  [✓] Saved → {dest}  ({_file_mb(dest):.1f} MB)")
    return dest


def eval_onnx_fp32() -> dict:
    from ultralytics import YOLO
    _banner("[5/8]", "ONNX FP32  (ONNX Runtime – CUDA Execution Provider)",
            f"cuda:{DEVICE_GPU}")
    onnx = _export_onnx_fp32()
    r = YOLO(onnx).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_ONNX,
        device=DEVICE_GPU, verbose=False, plots=False,
    )
    return _extract(r)


# ─── [6] ONNX INT8 Dynamic (CPU) ─────────────────────────────────────────────

def _onnx_int8_path() -> str:
    return str(Path(OUTPUT_DIR) / "yolo26l_best_int8.onnx")


def _quantize_onnx_int8(fp32_path: str) -> str:
    """Dynamic INT8 weight quantisation with ONNX Runtime (no calibration needed)."""
    dest = _onnx_int8_path()
    if Path(dest).exists():
        print(f"  [✓] ONNX INT8 already present: {dest}")
        return dest

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise RuntimeError("Run: pip install onnxruntime-gpu")

    print(f"\n  [→] Applying dynamic INT8 weight quantisation to ONNX …")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=dest,
        weight_type=QuantType.QInt8,
    )
    print(f"  [✓] Saved → {dest}  ({_file_mb(dest):.1f} MB)")
    return dest


def eval_onnx_int8() -> dict:
    from ultralytics import YOLO
    _banner("[6/8]",
            "ONNX INT8  (dynamic weight quant – ONNX Runtime CPU)", "cpu")
    fp32 = _export_onnx_fp32()
    int8 = _quantize_onnx_int8(fp32)
    # QLinearConv ops from dynamic quantisation have limited CUDA EP coverage;
    # CPU gives reliable results for accuracy benchmarking.
    r = YOLO(int8).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_ONNX,
        device="cpu", verbose=False, plots=False,
    )
    return _extract(r)


# ─── [7 & 8] TFLite FP16 / INT8 ──────────────────────────────────────────────

def _tflite_dest(tag: str) -> str:
    return str(Path(OUTPUT_DIR) / f"yolo26l_best_{tag}.tflite")


def _find_tflite(pattern: str) -> Path | None:
    """
    Search for a TFLite file matching *pattern* in common export locations:
      • current directory
      • {stem}_saved_model/ subfolder (ultralytics intermediate artefact)
    """
    stem = Path(MODEL_PATH).stem
    search_dirs = [Path("."), Path(f"{stem}_saved_model")]
    for d in search_dirs:
        hits = list(d.glob(f"*{pattern}*.tflite")) if d.exists() else []
        if hits:
            return hits[0]
    return None


def _export_tflite_variant(tag: str, *, half: bool, int8: bool) -> str:
    """
    Export one TFLite variant and copy it to OUTPUT_DIR (idempotent).
    Returns the path inside OUTPUT_DIR.
    """
    dest = _tflite_dest(tag)
    if Path(dest).exists():
        print(f"  [✓] TFLite {tag.upper()} already present: {dest}")
        return dest

    from ultralytics import YOLO
    label = "FP16" if half else "INT8"
    print(f"\n  [→] Exporting TFLite {label} "
          f"(goes via TF SavedModel — may take a few minutes) …")

    kwargs: dict = dict(
        format="tflite",
        imgsz=IMG_SIZE,
        half=half,
        int8=int8,
        verbose=False,
    )
    if int8:
        kwargs["data"] = DATASET_YAML   # calibration images

    exported_path = YOLO(MODEL_PATH).export(**kwargs)

    # ultralytics returns the path; copy it to our output directory
    if exported_path and Path(str(exported_path)).exists():
        shutil.copy(str(exported_path), dest)
    else:
        # Fallback: search for the generated file by pattern
        pattern = "float16" if half else "integer_quant"
        found = _find_tflite(pattern)
        if found is None:
            raise FileNotFoundError(
                f"Could not locate the exported TFLite {label} file. "
                f"Searched for '*{pattern}*.tflite' in ./ and "
                f"{Path(MODEL_PATH).stem}_saved_model/"
            )
        shutil.copy(str(found), dest)

    print(f"  [✓] Saved → {dest}  ({_file_mb(dest):.1f} MB)")
    return dest


def eval_tflite_fp16() -> dict:
    from ultralytics import YOLO
    _banner("[7/8]", "TFLite FP16  (TFLite interpreter – CPU)", "cpu")
    tflite = _export_tflite_variant("fp16", half=True, int8=False)
    r = YOLO(tflite).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_TFLITE,
        device="cpu", verbose=False, plots=False,
    )
    return _extract(r)


def eval_tflite_int8() -> dict:
    from ultralytics import YOLO
    _banner("[8/8]", "TFLite INT8  (dynamic-range quant – TFLite CPU)", "cpu")
    tflite = _export_tflite_variant("int8", half=False, int8=True)
    r = YOLO(tflite).val(
        data=DATASET_YAML, imgsz=IMG_SIZE, batch=BATCH_TFLITE,
        device="cpu", verbose=False, plots=False,
    )
    return _extract(r)


# ─── Reporting ────────────────────────────────────────────────────────────────

_MODEL_FILE: dict[str, str] = {}   # populated in main()


def _pct(new_val: float, base_val: float) -> str:
    if math.isnan(new_val) or math.isnan(base_val) or base_val == 0:
        return "N/A"
    return f"{(new_val - base_val) / abs(base_val) * 100:>+.2f}%"


def print_report(metrics: dict[str, dict], sizes: dict[str, float]) -> None:
    col_w   = 17
    name_w  = 11
    # columns: Device | Size (MB) | Precision | Recall | mAP50 | mAP50:95 | Latency | Throughput
    all_cols = ["Device", "Size (MB)"] + METRIC_COLS
    total_w  = name_w + col_w * len(all_cols)
    sep      = "─" * total_w

    hdr = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in all_cols)

    print("\n\n" + "═" * total_w)
    print(f"  QUANTIZATION COMPARISON — Fire & Smoke Detection (YOLO26l) — NVIDIA L4 GPU")
    print("═" * total_w)
    print(hdr)
    print(sep)

    for name, m in metrics.items():
        dev = DEVICE_LABEL.get(name, "?")
        sz  = sizes.get(name, float("nan"))
        if not m:
            print(f"  {name:<{name_w - 2}}  (skipped — see warning above)")
            continue
        row = f"{name:<{name_w}}{dev:>{col_w}}{sz:>{col_w}.1f}"
        for col in METRIC_COLS:
            val = m.get(col, float("nan"))
            row += (
                f"{val:>{col_w}.2f}" if ("Latency" in col or "Throughput" in col)
                else f"{val:>{col_w}.4f}"
            )
        print(row)

    print(sep)

    # Delta table versus PyTorch FP32 baseline
    base   = metrics.get("PT-FP32", {})
    base_sz = sizes.get("PT-FP32", float("nan"))
    if base and any(k != "PT-FP32" and v for k, v in metrics.items()):
        print(f"\n  Δ relative to PT-FP32 baseline  (+) = improvement for accuracy / throughput")
        print(sep)
        for name, m in metrics.items():
            if name == "PT-FP32" or not m:
                continue
            sz   = sizes.get(name, float("nan"))
            sz_d = _pct(sz, base_sz)
            row  = f"{name:<{name_w}}{'':>{col_w}}{sz_d:>{col_w}}"
            for col in METRIC_COLS:
                row += f"{_pct(m.get(col, float('nan')), base.get(col, float('nan'))):>{col_w}}"
            print(row)
        print("═" * total_w)

    # Human-readable summary
    print(f"\n  Summary")
    print(sep)
    for name, m in metrics.items():
        if not m:
            continue
        print(
            f"  {name:<11}  "
            f"mAP50:95={m.get('mAP50:95', float('nan')):.4f}  "
            f"latency={m.get('Latency (ms)', float('nan')):.1f} ms  "
            f"throughput={m.get('Throughput (img/s)', float('nan')):.1f} img/s  "
            f"size={sizes.get(name, float('nan')):.1f} MB  "
            f"device={DEVICE_LABEL.get(name, '?')}"
        )
    print("═" * total_w + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    banner = "YOLO GPU Quantization Evaluation — Fire/Smoke (NVIDIA L4, CC 8.9)"
    print("╔" + "═" * (len(banner) + 4) + "╗")
    print(f"║  {banner}  ║")
    print("╚" + "═" * (len(banner) + 4) + "╝")
    print(f"  Model  : {MODEL_PATH}  ({_file_mb(MODEL_PATH):.1f} MB)")
    n = len(list(Path(TEST_DIR, "images").glob("*.jpg")))
    print(f"  Test   : {TEST_DIR}/images/  ({n} images)")
    print(f"  Output : {OUTPUT_DIR}/\n")

    setup()

    metrics: dict[str, dict] = {}
    sizes:   dict[str, float] = {}

    # ── PyTorch ───────────────────────────────────────────────────────────────
    metrics["PT-FP32"] = _safe_run(eval_pytorch_fp32,  "PyTorch FP32")
    sizes["PT-FP32"]   = _file_mb(MODEL_PATH)

    metrics["PT-FP16"] = _safe_run(eval_pytorch_fp16,  "PyTorch FP16")
    sizes["PT-FP16"]   = _file_mb(MODEL_PATH)   # same .pt file, different compute

    # ── TensorRT ──────────────────────────────────────────────────────────────
    metrics["TRT-FP16"] = _safe_run(eval_trt_fp16,  "TensorRT FP16")
    sizes["TRT-FP16"]   = _file_mb(_trt_path("fp16"))

    metrics["TRT-INT8"] = _safe_run(eval_trt_int8,  "TensorRT INT8")
    sizes["TRT-INT8"]   = _file_mb(_trt_path("int8"))

    # ── ONNX ──────────────────────────────────────────────────────────────────
    #metrics["ONNX-FP32"] = _safe_run(eval_onnx_fp32, "ONNX FP32")
    #sizes["ONNX-FP32"]   = _file_mb(_onnx_fp32_path())

    #metrics["ONNX-INT8"] = _safe_run(eval_onnx_int8, "ONNX INT8")
    #sizes["ONNX-INT8"]   = _file_mb(_onnx_int8_path())

    # ── TFLite ────────────────────────────────────────────────────────────────
    #metrics["TFL-FP16"] = _safe_run(eval_tflite_fp16, "TFLite FP16")
    #sizes["TFL-FP16"]   = _file_mb(_tflite_dest("fp16"))

    #metrics["TFL-INT8"] = _safe_run(eval_tflite_int8, "TFLite INT8")
    #sizes["TFL-INT8"]   = _file_mb(_tflite_dest("int8"))

    print_report(metrics, sizes)


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
#    pip install onnx                 # ONNX model format support
#    pip install onnxruntime-gpu      # ONNX Runtime with CUDA EP + quantisation tools
#    pip install tensorflow           # required for TFLite export (goes via SavedModel)
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
#    import torch, ultralytics, onnxruntime, tensorflow as tf, tensorrt
#    print('torch    :', torch.__version__, '| CUDA:', torch.cuda.is_available())
#    print('ultralytics:', ultralytics.__version__)
#    print('onnxruntime:', onnxruntime.__version__)
#    print('tensorflow :', tf.__version__)
#    print('tensorrt   :', tensorrt.__version__)
#    "
# =============================================================================
