#!/usr/bin/env python3
"""
build_engine.py
===============
Build a TensorRT INT8 engine from a PyTorch (.pt) YOLO model file.

The engine is saved to ENGINE_PATH and is idempotent: if the file already
exists the build is skipped, allowing volume-persisted caching across
container restarts.

INT8 PTQ calibration (optional but recommended)
------------------------------------------------
If calibration images are mounted at CALIB_DIR/images/ (JPEG or PNG), they
are used for post-training quantization, which yields better INT8 accuracy.
If the directory is absent or empty the engine is still built as INT8 using
TensorRT's implicit calibration — accuracy may be slightly lower.

Environment variable overrides
-------------------------------
  MODEL_PATH    path to .pt source model       default: /app/models/yolo26l_best.pt
  ENGINE_PATH   destination .engine path       default: /app/engines/yolo26l_int8_bs16.engine
  CALIB_DIR     calibration images root dir    default: /data/calibration
  IMG_SIZE      inference input resolution     default: 640
  BATCH_SIZE    engine fixed batch size        default: 16
  DEVICE        CUDA device index              default: 0
  WORKSPACE_GB  TRT workspace memory (GiB)    default: 4
"""

import os
import shutil
import sys
import yaml
from pathlib import Path

# ── Configuration (overridable via environment) ───────────────────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "/app/models/yolo26l_best.pt")
ENGINE_PATH  = os.getenv("ENGINE_PATH",  "/app/engines/yolo26l_int8_bs16.engine")
CALIB_DIR    = os.getenv("CALIB_DIR",    "/data/calibration")
IMG_SIZE     = int(os.getenv("IMG_SIZE",      "640"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE",    "16"))
DEVICE       = os.getenv("DEVICE",            "0")
WORKSPACE_GB = int(os.getenv("WORKSPACE_GB",  "4"))
# ─────────────────────────────────────────────────────────────────────────────


def _file_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1e6
    except FileNotFoundError:
        return float("nan")


def _find_calibration_images() -> list:
    calib_images_dir = Path(CALIB_DIR) / "images"
    if not calib_images_dir.exists():
        return []
    imgs = (
        list(calib_images_dir.glob("*.jpg"))
        + list(calib_images_dir.glob("*.jpeg"))
        + list(calib_images_dir.glob("*.png"))
    )
    return imgs


def _write_calibration_yaml() -> str:
    yaml_path = "/tmp/calib_dataset.yaml"
    cfg = {
        "path": CALIB_DIR,
        "val":  "images",
        "train": "images",
        "test":  "images",
        "nc":    2,
        "names": {0: "smoke", 1: "fire"},
    }
    with open(yaml_path, "w") as fh:
        yaml.dump(cfg, fh, sort_keys=False)
    return yaml_path


def build_engine() -> str:
    """Build the TRT INT8 engine and return its path."""

    print("=" * 64)
    print("  TensorRT INT8 Engine Builder")
    print("=" * 64)
    print(f"  Source model  : {MODEL_PATH}  ({_file_mb(MODEL_PATH):.1f} MB)")
    print(f"  Output engine : {ENGINE_PATH}")
    print(f"  Image size    : {IMG_SIZE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  CUDA device   : {DEVICE}")
    print(f"  TRT workspace : {WORKSPACE_GB} GB")
    print()

    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        sys.exit(1)

    Path(ENGINE_PATH).parent.mkdir(parents=True, exist_ok=True)

    if Path(ENGINE_PATH).exists():
        print(f"[✓] Engine already present ({_file_mb(ENGINE_PATH):.1f} MB) — skipping build.")
        print(f"    To force a rebuild, delete: {ENGINE_PATH}")
        return ENGINE_PATH

    from ultralytics import YOLO

    calib_imgs = _find_calibration_images()

    if calib_imgs:
        print(f"[✓] Found {len(calib_imgs)} calibration images in {CALIB_DIR}/images/")
        print(f"[→] Building TensorRT INT8 engine with PTQ calibration ...")
        calib_yaml = _write_calibration_yaml()
        export_kwargs = dict(
            format="engine",
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            half=False,
            int8=True,
            workspace=WORKSPACE_GB,
            data=calib_yaml,
            fraction=1.0,
            verbose=True,
        )
    else:
        print(f"[!] No calibration images found at {CALIB_DIR}/images/")
        print(f"    Mount images there for better INT8 accuracy (see docker-compose.trt-build.yml).")
        print(f"[→] Building TensorRT INT8 engine with TRT implicit calibration ...")
        export_kwargs = dict(
            format="engine",
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            half=False,
            int8=True,
            workspace=WORKSPACE_GB,
            verbose=True,
        )

    print()
    print("[→] Layer fusion + kernel tuning in progress — this may take several minutes ...")
    exported = Path(str(YOLO(MODEL_PATH).export(**export_kwargs)))

    shutil.move(str(exported), ENGINE_PATH)

    print()
    print(f"[✓] Engine saved to {ENGINE_PATH}  ({_file_mb(ENGINE_PATH):.1f} MB)")
    return ENGINE_PATH


if __name__ == "__main__":
    build_engine()
