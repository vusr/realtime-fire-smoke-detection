#!/usr/bin/env python3
"""
evaluate.py
===========
Comprehensive evaluation of yolo26m_best.pt on the fire-and-smoke test set.

Outputs (saved to evaluation_results/):
  pr_curves/per_iou/          – PR curve PNG for each IoU threshold (50–95)
  pr_curves/combined/         – PR curves with all IoU thresholds overlaid (per class)
  precision_recall_f1/        – Precision, Recall, F1 vs confidence curves
  map_vs_iou/                 – mAP vs IoU threshold curve
  confusion_matrix/           – Raw + normalised confusion matrix heatmaps
  confidence_dist/            – Prediction confidence score histograms
  box_size_dist/              – GT bounding-box size distribution
  visualizations/fire/        – 20 annotated images sampled with fire GT
  visualizations/smoke/       – 20 annotated images sampled with smoke GT
  visualizations/background/  – 20 annotated images with no GT (background)
  visualizations/grids/       – 4×5 montage grid for each category
  summary/                    – metrics_summary.txt + metrics_summary.csv

Usage:
  cd /path/to/YOLO_Prediction
  python evaluate.py --data data.yaml
"""

import sys, random, time, csv, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import yaml

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ERROR: ultralytics not installed.  Run:  pip install ultralytics")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Evaluate YOLO model on test dataset")
parser.add_argument("--data", type=str, default="/home/smoon23/data.yaml", 
                    help="Path to data.yaml file containing dataset configuration")
parser.add_argument("--model", type=str, default="yolo26m_best.pt",
                    help="Path to trained model weights")
args = parser.parse_args()

# Load data configuration from YAML
DATA_YAML = Path(args.data)
if not DATA_YAML.exists():
    sys.exit(f"ERROR: data.yaml file not found at {DATA_YAML}")

with open(DATA_YAML, 'r') as f:
    data_config = yaml.safe_load(f)

# Extract configuration from data.yaml
DATA_ROOT    = Path(data_config.get('path', '.'))
TEST_IMG_DIR = DATA_ROOT / data_config.get('test', 'test/images')
# Derive labels directory from images directory
TEST_LBL_DIR = Path(str(TEST_IMG_DIR).replace('/images', '/labels'))

MODEL_PATH   = args.model
OUT          = Path("evaluation_results")

N_CLS        = data_config.get('nc', 2)
# Parse class names - handle both dict and list formats
names_data = data_config.get('names', {0: "smoke", 1: "fire"})
if isinstance(names_data, dict):
    CLASS_NAMES = [names_data[i] for i in range(N_CLS)]
else:
    CLASS_NAMES = names_data

# Colour scheme  (hex → used for matplotlib;  BGR computed below for OpenCV)
# Use consistent colors for both GT and predictions
CLS_HEX  = ["#FF4500", "#4682B4"]   # 0: fire = OrangeRed,  1: smoke = SteelBlue
# Adjust colors based on actual class mapping
if CLASS_NAMES[0] == "smoke":
    CLS_HEX = ["#4682B4", "#FF4500"]  # 0: smoke = SteelBlue, 1: fire = OrangeRed

# Colour scheme  (hex → used for matplotlib;  BGR computed below for OpenCV)
# Use consistent colors for both GT and predictions
CLS_HEX  = ["#FF4500", "#4682B4"]   # 0: fire = OrangeRed,  1: smoke = SteelBlue
# Adjust colors based on actual class mapping
if CLASS_NAMES[0] == "smoke":
    CLS_HEX = ["#4682B4", "#FF4500"]  # 0: smoke = SteelBlue, 1: fire = OrangeRed

# IoU thresholds for metric sweeps
IOU_THRS  = np.round(np.arange(0.50, 1.00, 0.05), 2)   # [0.50, 0.55, …, 0.95]

# Confidence steps for P/R/F1 curves
CONF_STEPS = np.linspace(0.0, 1.0, 1001)

# Inference settings
INF_CONF    = 0.001   # very low → capture virtually all detections
INF_NMS_IOU = 0.65    # NMS IoU during inference
VIZ_CONF    = 0.25    # threshold used for visualisations & confusion matrix

BATCH = 16
SEED  = 42

DEVICE = ("mps"  if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available()          else "cpu")

# Grid panel size
PANEL_W, PANEL_H = 640, 480
GRID_COLS = 4

# ──────────────────────────────────────────────────────────────────────────────
# Create output directories & print configuration
# ──────────────────────────────────────────────────────────────────────────────
for sub in [
    "pr_curves/per_iou", "pr_curves/combined",
    "precision_recall_f1",
    "map_vs_iou",
    "confusion_matrix",
    "confidence_dist",
    "box_size_dist",
    "visualizations/fire", "visualizations/smoke",
    "visualizations/background", "visualizations/grids",
    "summary",
]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight", "grid.alpha": 0.3,
})

print(f"Configuration loaded from: {DATA_YAML}")
print(f"  Data root      : {DATA_ROOT}")
print(f"  Test images    : {TEST_IMG_DIR}")
print(f"  Test labels    : {TEST_LBL_DIR}")
print(f"  Model          : {MODEL_PATH}")
print(f"  Classes ({N_CLS})    : {CLASS_NAMES}")
print(f"  Device         : {DEVICE}")
print(f"  Output         : {OUT.resolve()}\n")
t_start = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def hex_to_bgr(h: str):
    """Convert '#RRGGBB' hex colour to OpenCV (B,G,R) tuple."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


CLS_BGR = [hex_to_bgr(c) for c in CLS_HEX]


def yolo_to_xyxy(cx, cy, w, h, W, H):
    """Normalised YOLO [cx,cy,w,h] → pixel [x1,y1,x2,y2]."""
    return [(cx - w / 2) * W, (cy - h / 2) * H,
            (cx + w / 2) * W, (cy + h / 2) * H]


def load_gt(lbl_path, W, H):
    """
    Parse a YOLO-format label file.
    Returns {class_id: [[x1,y1,x2,y2], …]} in pixel coordinates.
    """
    d = defaultdict(list)
    p = Path(lbl_path)
    if p.exists() and p.stat().st_size > 0:
        for line in p.read_text().splitlines():
            tokens = line.split()
            if len(tokens) < 5:
                continue
            c = int(tokens[0])
            d[c].append(yolo_to_xyxy(*map(float, tokens[1:5]), W, H))
    return d


def iou_mat(A, B):
    """
    Vectorised IoU between two sets of boxes.
    A: array-like (N, 4) in [x1,y1,x2,y2]
    B: array-like (M, 4) in [x1,y1,x2,y2]
    Returns np.ndarray of shape (N, M).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim == 1:
        A = A[np.newaxis]
    if B.ndim == 1:
        B = B[np.newaxis]
    N, M = A.shape[0], B.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M))

    # A[:,0:1] shape (N,1); B[:,0] shape (M,) → broadcast to (N,M)
    x1 = np.maximum(A[:, 0:1], B[:, 0])
    y1 = np.maximum(A[:, 1:2], B[:, 1])
    x2 = np.minimum(A[:, 2:3], B[:, 2])
    y2 = np.minimum(A[:, 3:4], B[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    aA = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])   # (N,)
    aB = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])   # (M,)
    union = aA[:, np.newaxis] + aB[np.newaxis, :] - inter
    return inter / np.maximum(union, 1e-8)


def match_preds(pred_xyxy, gt_xyxy, iou_thr):
    """
    Greedy matching (predictions assumed sorted by confidence descending).
    Returns (tp, fp) arrays of shape (N,).
    """
    N, M = len(pred_xyxy), len(gt_xyxy)
    tp = np.zeros(N)
    fp = np.zeros(N)
    if N == 0:
        return tp, fp
    if M == 0:
        fp[:] = 1
        return tp, fp
    ious = iou_mat(pred_xyxy, gt_xyxy)   # (N, M)
    used = np.zeros(M, dtype=bool)
    for i in range(N):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not used[j]:
            tp[i] = 1
            used[j] = True
        else:
            fp[i] = 1
    return tp, fp


def compute_ap(rec, pre):
    """
    COCO-style all-point interpolated Average Precision.
    rec, pre: 1-D arrays already sorted by the confidence ordering.
    """
    r = np.concatenate([[0.], rec, [1.]])
    p = np.concatenate([[1.], pre, [0.]])
    # Monotonically decreasing precision envelope
    for i in range(p.size - 1, 0, -1):
        p[i - 1] = max(p[i - 1], p[i])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def evaluate_iou(iou_thr, img_paths, pred_store, gt_store):
    """
    Compute per-class precision/recall curves and AP at one IoU threshold.
    Returns (cls_data dict, mAP float).
    cls_data[c] = {confs, precision, recall, ap, n_gt}
    """
    cls_data = {}
    for c in range(N_CLS):
        confs_all, tp_all, fp_all = [], [], []
        n_gt = 0
        for ip in img_paths:
            p_cls = sorted(
                [d for d in pred_store[ip] if d[5] == c],
                key=lambda x: x[4], reverse=True
            )
            g_cls = gt_store[ip].get(c, [])
            n_gt += len(g_cls)
            tp, fp = match_preds([d[:4] for d in p_cls], g_cls, iou_thr)
            confs_all.extend(d[4] for d in p_cls)
            tp_all.extend(tp.tolist())
            fp_all.extend(fp.tolist())

        if confs_all:
            idx  = np.argsort(confs_all)[::-1]
            conf = np.array(confs_all)[idx]
            ctp  = np.cumsum(np.array(tp_all)[idx])
            cfp  = np.cumsum(np.array(fp_all)[idx])
            rec  = ctp / max(n_gt, 1)
            pre  = ctp / np.maximum(ctp + cfp, 1)
            ap   = compute_ap(rec, pre)
        else:
            conf = pre = rec = np.array([])
            ap = 0.0

        cls_data[c] = dict(confs=conf, precision=pre, recall=rec,
                           ap=ap, n_gt=n_gt)

    mAP = float(np.mean([cls_data[c]['ap'] for c in range(N_CLS)]))
    return cls_data, mAP


def pr_at_conf(cls_d, conf_thr):
    """
    Return (precision, recall) when using only predictions with
    confidence ≥ conf_thr (confs array must be sorted descending).
    """
    confs, pre, rec = cls_d['confs'], cls_d['precision'], cls_d['recall']
    if len(confs) == 0:
        return 1.0, 0.0
    k = int(np.sum(confs >= conf_thr))
    if k == 0:
        return 1.0, 0.0
    return float(pre[k - 1]), float(rec[k - 1])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – SCAN & CATEGORISE TEST IMAGES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("STEP 1  Scanning test images")
print("=" * 64)

img_paths = sorted(TEST_IMG_DIR.glob("*.jpg"))
if not img_paths:
    img_paths = sorted(TEST_IMG_DIR.glob("*.png"))
print(f"  {len(img_paths):,} images found in {TEST_IMG_DIR}")

# Find class indices for fire and smoke
fire_cls_idx = CLASS_NAMES.index("fire") if "fire" in CLASS_NAMES else None
smoke_cls_idx = CLASS_NAMES.index("smoke") if "smoke" in CLASS_NAMES else None

fire_imgs, smoke_imgs, bg_imgs = [], [], []
for ip in tqdm(img_paths, desc="  Categorising", leave=False):
    lp  = TEST_LBL_DIR / (ip.stem + ".txt")
    gt  = load_gt(lp, 1, 1)          # dummy W/H – only need class presence
    if fire_cls_idx is not None and gt.get(fire_cls_idx):  
        fire_imgs.append(ip)
    if smoke_cls_idx is not None and gt.get(smoke_cls_idx): 
        smoke_imgs.append(ip)
    if not any(gt.values()): 
        bg_imgs.append(ip)

print(f"  fire={len(fire_imgs):,}  smoke={len(smoke_imgs):,}  background={len(bg_imgs):,}")

# Random samples for visualisation (20 per category)
sample_fire  = random.sample(fire_imgs,  min(20, len(fire_imgs)))
sample_smoke = random.sample(smoke_imgs, min(20, len(smoke_imgs)))
sample_bg    = random.sample(bg_imgs,    min(20, len(bg_imgs)))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – RUN INFERENCE ON ALL TEST IMAGES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 2  Running inference on all test images")
print("=" * 64)

model = YOLO(MODEL_PATH)

pred_store = {}   # img_path → [[x1,y1,x2,y2,conf,cls], …]
img_shapes  = {}  # img_path → (W, H) pixels

for i in tqdm(range(0, len(img_paths), BATCH), desc="  Inference"):
    batch = img_paths[i : i + BATCH]
    results = model.predict(
        [str(p) for p in batch],
        conf=INF_CONF,
        iou=INF_NMS_IOU,
        device=DEVICE,
        verbose=False,
        save=False,
    )
    for ip, r in zip(batch, results):
        H0, W0 = r.orig_shape
        img_shapes[ip] = (W0, H0)
        dets = []
        if r.boxes is not None and len(r.boxes):
            bxs = r.boxes.xyxy.cpu().numpy()
            cfs = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for b, cf, cl in zip(bxs, cfs, cls):
                dets.append([*b.tolist(), float(cf), int(cl)])
        pred_store[ip] = dets

n_dets = sum(len(v) for v in pred_store.values())
print(f"  {n_dets:,} total detections collected (conf ≥ {INF_CONF})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – LOAD GROUND-TRUTH LABELS IN PIXEL COORDS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 3  Loading ground-truth labels")
print("=" * 64)

gt_store = {}   # img_path → {cls: [[x1,y1,x2,y2], …]}
for ip in tqdm(img_paths, desc="  Loading GT", leave=False):
    W, H = img_shapes[ip]
    gt_store[ip] = load_gt(TEST_LBL_DIR / (ip.stem + ".txt"), W, H)

# Count GT boxes per class
gt_counts = {c: sum(len(gt_store[ip].get(c, [])) for ip in img_paths) for c in range(N_CLS)}
n_gt_total = sum(gt_counts.values())

print(f"  GT boxes — ", end="")
for c, name in enumerate(CLASS_NAMES):
    print(f"{name}: {gt_counts[c]:,}  ", end="")
print(f"total: {n_gt_total:,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 – METRIC COMPUTATION (all IoU thresholds)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 4  Computing metrics at each IoU threshold")
print("=" * 64)

iou_results = {}
for thr in tqdm(IOU_THRS, desc="  IoU sweep"):
    iou_results[thr] = evaluate_iou(thr, img_paths, pred_store, gt_store)

map50   = iou_results[0.50][1]
map5095 = float(np.mean([iou_results[t][1] for t in IOU_THRS]))
print(f"\n  mAP@0.50      : {map50:.4f}")
print(f"  mAP@0.50:0.95 : {map5095:.4f}")

# ── Pre-compute Precision / Recall / F1 arrays at IoU = 0.50 ─────────────────
cls50  = iou_results[0.50][0]
P_arr  = np.zeros((N_CLS, len(CONF_STEPS)))
R_arr  = np.zeros((N_CLS, len(CONF_STEPS)))
F1_arr = np.zeros((N_CLS, len(CONF_STEPS)))

for c in range(N_CLS):
    for j, ct in enumerate(CONF_STEPS):
        p, r = pr_at_conf(cls50[c], ct)
        P_arr[c, j]  = p
        R_arr[c, j]  = r
        F1_arr[c, j] = 2 * p * r / max(p + r, 1e-8)

mean_P  = P_arr.mean(0)
mean_R  = R_arr.mean(0)
mean_F1 = F1_arr.mean(0)

best_f1 = {}
for c in range(N_CLS):
    bi = int(np.argmax(F1_arr[c]))
    best_f1[c] = dict(
        conf=float(CONF_STEPS[bi]),
        f1=float(F1_arr[c, bi]),
        precision=float(P_arr[c, bi]),
        recall=float(R_arr[c, bi]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 – GENERATE PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 5  Generating metric plots")
print("=" * 64)


# ── 5a. PR curve – one plot per IoU threshold ─────────────────────────────────
print("  5a. PR curves per IoU threshold …")
for thr in IOU_THRS:
    cls_data, map_val = iou_results[thr]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(N_CLS):
        d = cls_data[c]
        ax.plot(d['recall'], d['precision'],
                color=CLS_HEX[c], lw=2,
                label=f"{CLASS_NAMES[c]}  AP={d['ap']:.3f}")
        ax.fill_between(d['recall'], d['precision'],
                        alpha=0.07, color=CLS_HEX[c])
    ax.set(
        xlabel="Recall", ylabel="Precision",
        title=f"Precision–Recall  |  IoU = {thr:.2f}  |  mAP = {map_val:.3f}",
        xlim=[0, 1], ylim=[0, 1.02],
    )
    ax.legend(loc="lower left"); ax.grid(True)
    fname = f"pr_iou{int(round(thr * 100)):02d}.png"
    fig.savefig(OUT / "pr_curves" / "per_iou" / fname)
    plt.close(fig)


# ── 5b. Combined PR curves – all IoU thresholds overlaid, per class ───────────
print("  5b. Combined PR curves (per class) …")
plasma = plt.cm.plasma
for c in range(N_CLS):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, thr in enumerate(IOU_THRS):
        d   = iou_results[thr][0][c]
        col = plasma(i / max(len(IOU_THRS) - 1, 1))
        ax.plot(d['recall'], d['precision'],
                color=col, lw=1.8, alpha=0.9,
                label=f"IoU={thr:.2f}  AP={d['ap']:.3f}")
    # Colour-bar legend proxy
    sm = plt.cm.ScalarMappable(cmap=plasma,
                               norm=plt.Normalize(IOU_THRS[0], IOU_THRS[-1]))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="IoU threshold")
    ax.set(
        xlabel="Recall", ylabel="Precision",
        title=f"PR at All IoU Thresholds — {CLASS_NAMES[c].title()}",
        xlim=[0, 1], ylim=[0, 1.02],
    )
    ax.grid(True)
    fig.savefig(OUT / "pr_curves" / "combined" / f"pr_combined_{CLASS_NAMES[c]}.png")
    plt.close(fig)


# ── 5c. Precision / Recall / F1 vs Confidence (IoU = 0.50) ───────────────────
print("  5c. P/R/F1 vs confidence …")

# Combined 3-panel figure
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
data_triples = [
    ("Precision vs Confidence", P_arr,  mean_P,  "Precision"),
    ("Recall vs Confidence",    R_arr,  mean_R,  "Recall"),
    ("F1 Score vs Confidence",  F1_arr, mean_F1, "F1 Score"),
]
for ax, (title, arr, marr, yl) in zip(axes, data_triples):
    for c in range(N_CLS):
        ax.plot(CONF_STEPS, arr[c], color=CLS_HEX[c], lw=2, label=CLASS_NAMES[c])
    ax.plot(CONF_STEPS, marr, 'k--', lw=2, label='mean')
    ax.set(title=title, xlabel="Confidence Threshold", ylabel=yl,
           xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="best", fontsize=9); ax.grid(True)

# Mark best-F1 points on the F1 panel
for c in range(N_CLS):
    bf = best_f1[c]
    axes[2].scatter([bf['conf']], [bf['f1']], color=CLS_HEX[c], s=90, zorder=6)
    offset_x = 0.04 if bf['conf'] < 0.8 else -0.18
    axes[2].annotate(
        f"F1={bf['f1']:.3f}\n@ {bf['conf']:.2f}",
        xy=(bf['conf'], bf['f1']),
        xytext=(bf['conf'] + offset_x, bf['f1'] - 0.07),
        color=CLS_HEX[c], fontsize=8,
        arrowprops=dict(arrowstyle='->', color=CLS_HEX[c], lw=0.9),
    )

fig.suptitle("Precision / Recall / F1 vs Confidence Threshold  (IoU = 0.50)",
             fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "precision_recall_f1" / "prf1_vs_confidence.png")
plt.close(fig)

# Individual plots for each metric
for metric_name, arr, marr in [
    ("precision", P_arr,  mean_P),
    ("recall",    R_arr,  mean_R),
    ("f1_score",  F1_arr, mean_F1),
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(N_CLS):
        ax.plot(CONF_STEPS, arr[c], color=CLS_HEX[c], lw=2, label=CLASS_NAMES[c])
    ax.plot(CONF_STEPS, marr, 'k--', lw=2, label='mean')
    pretty = metric_name.replace('_', ' ').title()
    ax.set(
        title=f"{pretty} vs Confidence  (IoU = 0.50)",
        xlabel="Confidence Threshold", ylabel=pretty,
        xlim=[0, 1], ylim=[0, 1.02],
    )
    ax.legend(); ax.grid(True)
    fig.savefig(OUT / "precision_recall_f1" / f"{metric_name}_vs_conf.png")
    plt.close(fig)


# ── 5d. mAP vs IoU curve ──────────────────────────────────────────────────────
print("  5d. mAP vs IoU curve …")
map_vals = np.array([iou_results[t][1] for t in IOU_THRS])
per_ap   = {c: np.array([iou_results[t][0][c]['ap'] for t in IOU_THRS])
            for c in range(N_CLS)}

fig, ax = plt.subplots(figsize=(8, 5))
for c in range(N_CLS):
    ax.plot(IOU_THRS, per_ap[c],
            color=CLS_HEX[c], lw=2, marker='o', ms=5,
            label=f"{CLASS_NAMES[c]}")
ax.plot(IOU_THRS, map_vals, 'k-', lw=2.5, marker='D', ms=6, label="mAP (mean)")
ax.fill_between(IOU_THRS, map_vals, alpha=0.12, color='gray')

ax.annotate(
    f"mAP@0.50 = {map50:.3f}",
    xy=(0.50, map_vals[0]),
    xytext=(0.53, map_vals[0] + 0.04),
    fontsize=9,
    arrowprops=dict(arrowstyle='->', lw=1),
)
ax.annotate(
    f"mAP@[.50:.95] = {map5095:.3f}",
    xy=(np.mean(IOU_THRS), np.interp(np.mean(IOU_THRS), IOU_THRS, map_vals)),
    xytext=(0.62, np.interp(np.mean(IOU_THRS), IOU_THRS, map_vals) + 0.06),
    fontsize=9, color='gray',
    arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'),
)
ax.set(
    xlabel="IoU Threshold", ylabel="Average Precision",
    title="mAP / AP vs IoU Threshold  (0.50 – 0.95)",
    xlim=[0.47, 0.98], ylim=[0, 1.05],
)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax.legend(); ax.grid(True)
fig.savefig(OUT / "map_vs_iou" / "map_vs_iou.png")
plt.close(fig)


# ── 5e. Confusion matrix (conf ≥ VIZ_CONF, IoU ≥ 0.50) ──────────────────────
print("  5e. Confusion matrix …")
N_CM     = N_CLS + 1   # fire | smoke | background
cm       = np.zeros((N_CM, N_CM), dtype=int)
CM_LBLS  = CLASS_NAMES + ["background"]

for ip in img_paths:
    # Filter predictions above visualisation threshold
    preds = sorted(
        [d for d in pred_store[ip] if d[4] >= VIZ_CONF],
        key=lambda x: x[4], reverse=True
    )
    # Gather all GT boxes with their class ids
    gt_all = []
    for c in range(N_CLS):
        for b in gt_store[ip].get(c, []):
            gt_all.append(b + [c])         # [x1,y1,x2,y2,cls]

    matched_gt = [False] * len(gt_all)
    matched_pd = [False] * len(preds)

    if gt_all and preds:
        ious = iou_mat([p[:4] for p in preds],
                       [g[:4] for g in gt_all])    # (N_pred, N_gt)
        for i in range(len(preds)):
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= 0.50 and not matched_gt[j]:
                matched_gt[j] = True
                matched_pd[i] = True
                cm[gt_all[j][4], preds[i][5]] += 1

    # Unmatched GT → FN (row=GT class, col=background)
    for j, g in enumerate(gt_all):
        if not matched_gt[j]:
            cm[g[4], N_CLS] += 1

    # Unmatched predictions → FP (row=background, col=pred class)
    for i, p in enumerate(preds):
        if not matched_pd[i]:
            cm[N_CLS, p[5]] += 1


def _plot_cm(mat, title, fname, fmt_fn, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap=cmap, aspect='auto')
    ax.set(
        xticks=range(N_CM), yticks=range(N_CM),
        xticklabels=CM_LBLS, yticklabels=CM_LBLS,
        xlabel="Predicted", ylabel="Ground Truth",
        title=title,
    )
    for i in range(N_CM):
        for j in range(N_CM):
            val = mat[i, j]
            text_col = "white" if val > mat.max() * 0.55 else "black"
            ax.text(j, i, fmt_fn(val),
                    ha="center", va="center", fontsize=12, color=text_col)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)

_plot_cm(cm,
         f"Confusion Matrix  (conf ≥ {VIZ_CONF}, IoU ≥ 0.50)",
         OUT / "confusion_matrix" / "confusion_matrix.png",
         fmt_fn=str)

cm_norm = cm.astype(float)
row_sums = cm_norm.sum(1, keepdims=True)
cm_norm  = np.where(row_sums > 0, cm_norm / row_sums, 0.0)
_plot_cm(cm_norm,
         f"Normalised Confusion Matrix  (conf ≥ {VIZ_CONF}, IoU ≥ 0.50)",
         OUT / "confusion_matrix" / "confusion_matrix_normalized.png",
         fmt_fn=lambda v: f"{v:.2f}")


# ── 5f. Confidence score distribution ────────────────────────────────────────
print("  5f. Confidence distribution …")
all_confs = defaultdict(list)
for ip in img_paths:
    for d in pred_store[ip]:
        all_confs[int(d[5])].append(d[4])

fig, axes = plt.subplots(1, N_CLS + 1, figsize=(14, 4))
for c in range(N_CLS):
    axes[c].hist(all_confs[c], bins=50, color=CLS_HEX[c],
                 alpha=0.75, edgecolor='k', linewidth=0.3)
    axes[c].axvline(VIZ_CONF, color='red', ls='--', lw=1.5,
                    label=f"viz thr={VIZ_CONF}")
    axes[c].set(title=f"{CLASS_NAMES[c].title()}  (n={len(all_confs[c]):,})",
                xlabel="Confidence", ylabel="Count")
    axes[c].legend(fontsize=8); axes[c].grid(True)

combined = [v for lst in all_confs.values() for v in lst]
axes[-1].hist(combined, bins=50, color="dimgray",
              alpha=0.75, edgecolor='k', linewidth=0.3)
axes[-1].axvline(VIZ_CONF, color='red', ls='--', lw=1.5, label=f"viz thr={VIZ_CONF}")
axes[-1].set(title=f"All classes  (n={len(combined):,})",
             xlabel="Confidence", ylabel="Count")
axes[-1].legend(fontsize=8); axes[-1].grid(True)

fig.suptitle("Prediction Confidence Score Distribution  (all detections, conf ≥ 0.001)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "confidence_dist" / "confidence_distribution.png")
plt.close(fig)


# ── 5g. GT bounding-box size distribution ────────────────────────────────────
print("  5g. GT box size distribution …")
box_areas = defaultdict(list)   # cls → [relative_area, …]
for ip in img_paths:
    W, H = img_shapes[ip]
    img_area = W * H
    for c in range(N_CLS):
        for box in gt_store[ip].get(c, []):
            x1, y1, x2, y2 = box
            rel = (x2 - x1) * (y2 - y1) / img_area
            box_areas[c].append(rel)

fig, axes = plt.subplots(1, N_CLS, figsize=(12, 4))
for c in range(N_CLS):
    areas = box_areas[c]
    axes[c].hist(areas, bins=50, color=CLS_HEX[c],
                 alpha=0.75, edgecolor='k', linewidth=0.3)
    # COCO-style relative-area bands (≈32² and 96² pixels on a 640-px image)
    axes[c].axvline(0.0025, color='dodgerblue', ls='--', lw=1.2,
                    label="small  (≤0.0025)")
    axes[c].axvline(0.0225, color='tomato',    ls='--', lw=1.2,
                    label="large  (≥0.0225)")
    axes[c].set(
        title=f"{CLASS_NAMES[c].title()} GT sizes  (n={len(areas):,})",
        xlabel="Relative area  (box / image)", ylabel="Count", xscale="log",
    )
    axes[c].legend(fontsize=8); axes[c].grid(True, which='both', alpha=0.3)

fig.suptitle("GT Bounding-Box Size Distribution  (relative to image area)",
             fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "box_size_dist" / "box_size_distribution.png")
plt.close(fig)


# ── 5h. F1-score heat-map: class × confidence threshold ──────────────────────
print("  5h. F1 heat-map …")
# rows = classes, columns = conf threshold bins
conf_bins = np.linspace(0, 1, 101)
f1_heat   = np.zeros((N_CLS, len(conf_bins)))
for c in range(N_CLS):
    for j, ct in enumerate(conf_bins):
        p, r = pr_at_conf(cls50[c], ct)
        f1_heat[c, j] = 2 * p * r / max(p + r, 1e-8)

fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(f1_heat, cmap='RdYlGn', aspect='auto',
               vmin=0, vmax=1, extent=[0, 1, N_CLS - 0.5, -0.5])
ax.set(yticks=range(N_CLS), yticklabels=CLASS_NAMES,
       xlabel="Confidence Threshold", title="F1 Score Heat-map  (IoU = 0.50)")
plt.colorbar(im, ax=ax, label="F1")
fig.tight_layout()
fig.savefig(OUT / "precision_recall_f1" / "f1_heatmap.png")
plt.close(fig)


# ── 5i. Precision–Recall operating point scatter at key conf thresholds ───────
print("  5i. PR operating-point scatter …")
key_confs = [0.10, 0.25, 0.40, 0.50, 0.75]
fig, ax = plt.subplots(figsize=(7, 5))
markers = ['o', 's', 'D', '^', 'P']
for c in range(N_CLS):
    d = cls50[c]
    ax.plot(d['recall'], d['precision'],
            color=CLS_HEX[c], lw=1.5, alpha=0.6, label=f"_nolegend_")
    for ct, mk in zip(key_confs, markers):
        p, r = pr_at_conf(d, ct)
        ax.scatter(r, p, marker=mk, s=70, color=CLS_HEX[c], zorder=5)
        ax.annotate(f"{ct}", xy=(r, p), xytext=(r + 0.01, p + 0.02),
                    fontsize=7, color=CLS_HEX[c])

# Legend
for c in range(N_CLS):
    ax.plot([], [], color=CLS_HEX[c], lw=2, label=CLASS_NAMES[c])
for ct, mk in zip(key_confs, markers):
    ax.scatter([], [], marker=mk, s=60, color='gray', label=f"conf={ct}")
ax.set(xlabel="Recall", ylabel="Precision",
       title="PR Operating Points at Key Confidence Thresholds  (IoU=0.50)",
       xlim=[0, 1], ylim=[0, 1.02])
ax.legend(fontsize=8, ncol=2); ax.grid(True)
fig.savefig(OUT / "precision_recall_f1" / "pr_operating_points.png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 – PREDICTION VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 6  Generating prediction visualisations")
print("=" * 64)


def render(img_path, conf_thr=VIZ_CONF, max_side=900):
    """
    Return a vertically stacked annotated BGR image with:
      - Top half    : GT annotations (ground truth boxes)
      - Bottom half : Prediction annotations (predicted boxes)
    Both use consistent class colors.
    """
    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        return None
    H0, W0 = img_orig.shape[:2]
    scale  = min(max_side / W0, max_side / H0, 1.0)
    W1     = int(W0 * scale)
    H1     = int(H0 * scale)
    if scale < 1.0:
        img_orig = cv2.resize(img_orig, (W1, H1), interpolation=cv2.INTER_AREA)
    sw, sh = W1 / W0, H1 / H0
    
    # Create two copies for GT and Predictions
    img_gt = img_orig.copy()
    img_pred = img_orig.copy()

    def _draw_box(img, x1, y1, x2, y2, col, label, thickness=2):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx1, ly1 = x1, max(y1 - th - 8, 0)
        lx2, ly2 = x1 + tw + 8, max(y1, th + 8)
        # Semi-transparent background for label
        overlay = img.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), col, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        txt_col = (255, 255, 255)
        cv2.putText(img, label, (lx1 + 4, ly2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 1, cv2.LINE_AA)

    # Draw ground-truth boxes on img_gt
    for c in range(N_CLS):
        for box in gt_store[img_path].get(c, []):
            bx = [v * s for v, s in zip(box, [sw, sh, sw, sh])]
            _draw_box(img_gt, *bx, CLS_BGR[c], f"{CLASS_NAMES[c]}", thickness=3)

    # Draw predicted boxes on img_pred (sorted ascending so highest conf drawn last)
    for det in sorted(pred_store[img_path], key=lambda x: x[4]):
        if det[4] < conf_thr:
            continue
        cl   = int(det[5])
        bx   = [det[k] * s for k, s in zip(range(4), [sw, sh, sw, sh])]
        _draw_box(img_pred, *bx, CLS_BGR[cl], f"{CLASS_NAMES[cl]} {det[4]:.2f}", thickness=2)

    # Add labels to each image indicating GT vs Prediction
    label_h = 40
    
    # GT label bar
    gt_bar = np.zeros((label_h, W1, 3), dtype=np.uint8)
    gt_bar[:] = (40, 40, 40)  # Dark gray background
    cv2.putText(gt_bar, "GROUND TRUTH", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Prediction label bar
    pred_bar = np.zeros((label_h, W1, 3), dtype=np.uint8)
    pred_bar[:] = (40, 40, 40)  # Dark gray background
    cv2.putText(pred_bar, "PREDICTIONS", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Stack: GT_bar, GT_image, separator, Pred_bar, Pred_image
    separator = np.ones((5, W1, 3), dtype=np.uint8) * 200  # Light gray separator
    
    stacked = np.vstack([gt_bar, img_gt, separator, pred_bar, img_pred])
    
    # Add legend at the bottom
    legend_h = 30 + len(CLASS_NAMES) * 25
    legend = np.zeros((legend_h, W1, 3), dtype=np.uint8)
    legend[:] = (30, 30, 30)
    
    ly = 10
    cv2.putText(legend, "Legend:", (10, ly + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    ly += 30
    
    for c in range(N_CLS):
        # Draw color box
        cv2.rectangle(legend, (15, ly), (35, ly + 15), CLS_BGR[c], -1)
        # Draw label
        cv2.putText(legend, f"{CLASS_NAMES[c]}", (45, ly + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        ly += 25
    
    final = np.vstack([stacked, legend])
    
    return final


def save_category(samples, category, out_sub):
    """Save individual annotated images and a 4×5 grid montage."""
    panels = []
    for idx, ip in enumerate(samples):
        img = render(ip)
        if img is None:
            continue
        out_img = OUT / "visualizations" / out_sub / f"{idx:02d}_{ip.name}"
        cv2.imwrite(str(out_img), img)
        # Resize for grid - maintain aspect ratio
        h, w = img.shape[:2]
        aspect = w / h
        if aspect > PANEL_W / PANEL_H:
            # Width is limiting factor
            new_w = PANEL_W
            new_h = int(PANEL_W / aspect)
        else:
            # Height is limiting factor
            new_h = PANEL_H
            new_w = int(PANEL_H * aspect)
        resized = cv2.resize(img, (new_w, new_h))
        # Create panel with black padding if needed
        panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        y_offset = (PANEL_H - new_h) // 2
        x_offset = (PANEL_W - new_w) // 2
        panel[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        panels.append(panel)

    if not panels:
        print(f"    {category}: no images rendered")
        return

    n_cols = GRID_COLS
    n_rows = (len(panels) + n_cols - 1) // n_cols
    grid   = np.zeros((n_rows * PANEL_H, n_cols * PANEL_W, 3), dtype=np.uint8)
    for k, panel in enumerate(panels):
        r, col = divmod(k, n_cols)
        grid[r * PANEL_H:(r + 1) * PANEL_H,
             col * PANEL_W:(col + 1) * PANEL_W] = panel

    # Category title bar
    bar_h = 35
    bar   = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    bar[:] = (50, 50, 50)
    cv2.putText(bar, f"{category.upper()} — {len(panels)} samples  "
                     f"(GT above, Predictions below for each image)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    grid = np.vstack([bar, grid])

    out_grid = OUT / "visualizations" / "grids" / f"grid_{category}.jpg"
    cv2.imwrite(str(out_grid), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"    [{category}] {len(panels)} images saved  →  {out_grid.name}")


for samples, cat, sub in [
    (sample_fire,  "fire",       "fire"),
    (sample_smoke, "smoke",      "smoke"),
    (sample_bg,    "background", "background"),
]:
    save_category(samples, cat, sub)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 – SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("STEP 7  Writing summary report")
print("=" * 64)

ap_header = (f"{'Class':<12}" +
             "".join(f"{'AP@'+str(int(t*100)):<9}" for t in IOU_THRS) +
             f"{'mean-AP':<9}")
sep = "-" * len(ap_header)

ap_rows = []
for c in range(N_CLS):
    aps = [iou_results[t][0][c]['ap'] for t in IOU_THRS]
    ap_rows.append(
        f"{CLASS_NAMES[c]:<12}" +
        "".join(f"{v:<9.4f}" for v in aps) +
        f"{float(np.mean(aps)):<9.4f}"
    )
map_row = (f"{'mAP':<12}" +
           "".join(f"{iou_results[t][1]:<9.4f}" for t in IOU_THRS) +
           f"{map5095:<9.4f}")

elapsed = time.time() - t_start

# Build GT counts string dynamically
gt_counts_str = "  GT boxes     : " + "  ".join(
    f"{CLASS_NAMES[c]}={gt_counts[c]:,}" for c in range(N_CLS)
) + f"  total={n_gt_total:,}"

report_lines = [
    "=" * 80,
    "YOLO EVALUATION REPORT",
    f"Model  : {MODEL_PATH}",
    f"Device : {DEVICE}",
    "=" * 80,
    "",
    "── Dataset Statistics ──",
    f"  Test images  : {len(img_paths):,}",
    f"    with fire  : {len(fire_imgs):,}",
    f"    with smoke : {len(smoke_imgs):,}",
    f"    background : {len(bg_imgs):,}  (no GT boxes)",
    gt_counts_str,
    f"  Detections   : {n_dets:,}  (conf ≥ {INF_CONF}, NMS IoU={INF_NMS_IOU})",
    "",
    "── Average Precision by Class and IoU Threshold ──",
    ap_header, sep,
    *ap_rows, sep, map_row,
    "",
    f"  mAP@0.50      : {map50:.4f}",
    f"  mAP@0.50:0.95 : {map5095:.4f}",
    "",
    "── Per-class AP@0.50 ──",
    *[f"  {CLASS_NAMES[c]:<10}: {iou_results[0.50][0][c]['ap']:.4f}"
      for c in range(N_CLS)],
    "",
    "── Per-class AP@0.75 ──",
    *[f"  {CLASS_NAMES[c]:<10}: {iou_results[0.75][0][c]['ap']:.4f}"
      for c in range(N_CLS)],
    "",
    "── Best F1 Score per Class (IoU = 0.50) ──",
    *[
        f"  {CLASS_NAMES[c]:<10} "
        f"F1={best_f1[c]['f1']:.4f}  conf={best_f1[c]['conf']:.3f}  "
        f"P={best_f1[c]['precision']:.4f}  R={best_f1[c]['recall']:.4f}"
        for c in range(N_CLS)
    ],
    "",
    "── Confusion Matrix Totals (conf ≥ 0.25, IoU ≥ 0.50) ──",
]

# Add confusion matrix details dynamically based on class names
if N_CLS == 2:
    # For 2-class case, show detailed confusion matrix
    report_lines.extend([
        f"  TP {CLASS_NAMES[0]:<6}: {cm[0,0]:,}   FP {CLASS_NAMES[0]}→{CLASS_NAMES[1]} : {cm[1,0]:,}   "
        f"FN {CLASS_NAMES[0]}→bg : {cm[0,2]:,}",
        f"  TP {CLASS_NAMES[1]:<6}: {cm[1,1]:,}   FP {CLASS_NAMES[1]}→{CLASS_NAMES[0]} : {cm[0,1]:,}   "
        f"FN {CLASS_NAMES[1]}→bg : {cm[1,2]:,}",
        f"  FP {CLASS_NAMES[0]}  (bg→{CLASS_NAMES[0]})  : {cm[2,0]:,}",
        f"  FP {CLASS_NAMES[1]}  (bg→{CLASS_NAMES[1]}) : {cm[2,1]:,}",
    ])
else:
    # For more classes, show simpler summary
    for c in range(N_CLS):
        tp = cm[c, c]
        fn = cm[c, N_CLS]
        fp = cm[N_CLS, c]
        report_lines.append(f"  {CLASS_NAMES[c]:<10}: TP={tp:,}  FN={fn:,}  FP={fp:,}")

report_lines.extend([
    "",
    f"── Elapsed : {elapsed:.1f}s ──",
    "",
])

report_text = "\n".join(report_lines)
print(report_text)
(OUT / "summary" / "metrics_summary.txt").write_text(report_text)

# CSV
with open(OUT / "summary" / "metrics_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class"] + [f"AP@{int(t*100)}" for t in IOU_THRS] + ["mean_AP"])
    for c in range(N_CLS):
        aps = [iou_results[t][0][c]['ap'] for t in IOU_THRS]
        w.writerow([CLASS_NAMES[c]] +
                   [f"{v:.4f}" for v in aps] +
                   [f"{float(np.mean(aps)):.4f}"])
    map_vals = [iou_results[t][1] for t in IOU_THRS]
    w.writerow(["mAP"] +
               [f"{v:.4f}" for v in map_vals] +
               [f"{map5095:.4f}"])

# Best-F1 CSV
with open(OUT / "summary" / "best_f1_per_class.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "best_F1", "conf_threshold", "precision", "recall"])
    for c in range(N_CLS):
        bf = best_f1[c]
        w.writerow([CLASS_NAMES[c], f"{bf['f1']:.4f}", f"{bf['conf']:.4f}",
                    f"{bf['precision']:.4f}", f"{bf['recall']:.4f}"])

print("\n" + "=" * 64)
print(f"All outputs saved to : {OUT.resolve()}")
print(f"Total elapsed        : {elapsed:.1f}s")
print("=" * 64)
