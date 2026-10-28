"""
Evaluate a trained pothole model on the held-out test split and pick the
deployment confidence threshold.

Reports overall precision / recall / mAP, recall broken down by pothole size
(small potholes are the hard case) and by condition (night, Indian dashcam,
other sources), and image-level metrics that match how the app uses the model
("did it fire on this frame?").

Usage:
    python training/evaluate.py training/runs/yolo11n/weights/best.pt
Writes <run>/eval_report.json and prints a summary.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(ROOT, "data", "potholes")
IOU_MATCH = 0.5
# Box area as a fraction of the image: <0.5% is "small" (a pothole a few metres
# away in a phone frame), 0.5–5% medium, >5% large.
SIZE_BUCKETS = [("small", 0, 0.005), ("medium", 0.005, 0.05), ("large", 0.05, 1.01)]


def load_labels(stem):
    path = os.path.join(DATASET, "test", "labels", stem + ".txt")
    boxes = []
    for line in open(path):
        _, cx, cy, w, h = map(float, line.split())
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def match(gts, preds, threshold):
    """Greedy match predictions (>= threshold) to ground truth at IoU 0.5.
    Returns (tp_flags_per_gt, n_false_positives)."""
    preds = sorted([p for p in preds if p[4] >= threshold], key=lambda p: -p[4])
    hit = [False] * len(gts)
    fp = 0
    for p in preds:
        best, best_iou = -1, IOU_MATCH
        for i, g in enumerate(gts):
            if not hit[i]:
                v = iou(p, g)
                if v >= best_iou:
                    best, best_iou = i, v
        if best >= 0:
            hit[best] = True
        else:
            fp += 1
    return hit, fp


def group_of(tag):
    if "night" in tag["tags"]:
        return "night"
    if tag["source"] == "rdd_india":
        return "india_dashcam"
    if tag["source"].startswith("rdd_"):
        return "other_dashcam"
    return "phone_web_photos"


def main(weights):
    model = YOLO(weights)
    run_dir = os.path.dirname(os.path.dirname(weights))
    tags = json.load(open(os.path.join(DATASET, "test_tags.json")))

    # Standard metrics (mAP etc.) via Ultralytics on the test split. The test split
    # only has box labels, so a segmentation model is scored on its boxes by the
    # threshold sweep below (same matching as for detectors) and mAP is skipped.
    if model.task == "segment":
        summary = {"note": "segmentation model: test split has box labels only; mAP not computed, "
                           "see threshold_sweep for box precision/recall"}
    else:
        metrics = model.val(data=os.path.join(DATASET, "data.yaml"), split="test", imgsz=640,
                            batch=8, device="mps", plots=False, verbose=False)
        summary = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }

    # Raw predictions at a low threshold so we can sweep the deployment threshold
    images = sorted(glob.glob(os.path.join(DATASET, "test", "images", "*.jpg")))
    preds = {}
    for i in range(0, len(images), 16):
        for r in model.predict(images[i:i + 16], conf=0.05, imgsz=640, device="mps", verbose=False):
            stem = os.path.splitext(os.path.basename(r.path))[0]
            h, w = r.orig_shape
            preds[stem] = [(x1 / w, y1 / h, x2 / w, y2 / h, float(c))
                           for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]

    sweep = []
    for t in np.arange(0.10, 0.71, 0.05):
        tp = fp = fn = 0
        img_tp = img_fp = img_fn = img_tn = 0
        for stem in preds:
            gts = load_labels(stem)
            hit, nfp = match(gts, preds[stem], t)
            tp += sum(hit); fn += len(hit) - sum(hit); fp += nfp
            fired = any(p[4] >= t for p in preds[stem])
            if gts and fired: img_tp += 1
            elif gts: img_fn += 1
            elif fired: img_fp += 1
            else: img_tn += 1
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        sweep.append({
            "threshold": round(float(t), 2),
            "box_precision": round(p, 3), "box_recall": round(r, 3),
            "box_f1": round(2 * p * r / (p + r), 3) if p + r else 0,
            "image_recall": round(img_tp / (img_tp + img_fn), 3) if img_tp + img_fn else 0,
            "image_false_alarm_rate": round(img_fp / (img_fp + img_tn), 3) if img_fp + img_tn else 0,
        })

    # Deployment threshold: best box F1, nudged towards recall (missing a real
    # pothole is worse than a false alarm the citizen can dismiss)
    best_f1 = max(s["box_f1"] for s in sweep)
    candidates = [s for s in sweep if s["box_f1"] >= best_f1 - 0.01]
    chosen = min(candidates, key=lambda s: s["threshold"])
    t = chosen["threshold"]

    # Breakdowns at the chosen threshold
    by_size = defaultdict(lambda: [0, 0])
    by_group = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "images": 0})
    for stem in preds:
        gts = load_labels(stem)
        hit, nfp = match(gts, preds[stem], t)
        g = by_group[group_of(tags.get(stem, {"source": "?", "tags": []}))]
        g["tp"] += sum(hit); g["fn"] += len(hit) - sum(hit); g["fp"] += nfp; g["images"] += 1
        for box, h in zip(gts, hit):
            area = (box[2] - box[0]) * (box[3] - box[1])
            for name, lo, hi in SIZE_BUCKETS:
                if lo <= area < hi:
                    by_size[name][0] += h
                    by_size[name][1] += 1

    report = {
        "weights": weights,
        "test_images": len(preds),
        "ultralytics_metrics": summary,
        "chosen_threshold": t,
        "at_threshold": chosen,
        "recall_by_size": {k: {"recall": round(v[0] / v[1], 3) if v[1] else None, "boxes": v[1]}
                           for k, v in by_size.items()},
        "by_condition": {k: {"recall": round(v["tp"] / (v["tp"] + v["fn"]), 3) if v["tp"] + v["fn"] else None,
                             "precision": round(v["tp"] / (v["tp"] + v["fp"]), 3) if v["tp"] + v["fp"] else None,
                             "images": v["images"], "boxes": v["tp"] + v["fn"]}
                         for k, v in by_group.items()},
        "threshold_sweep": sweep,
    }
    with open(os.path.join(run_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "threshold_sweep"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
