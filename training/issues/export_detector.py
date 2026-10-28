"""
Evaluate a trained multi-class issue detector, pick a confidence threshold per class, export
to ONNX and install it in the backend.

Thresholds are the best-F1 confidence on the VALIDATION split; the metrics reported (and
stored) come from the held-out TEST split.

Usage:
    python training/issues/export_detector.py training/runs/issues_yolo11n/weights/best.pt
Installs backend/ml_models/issue_detector.onnx + issue_detector.json
"""

import json
import os
import shutil
import sys

import numpy as np
from ultralytics import YOLO

from common import DATA_DET, PROJECT, detector_class_names, load_classes

INPUT_SIZE = 640
MIN_THRESHOLD = 0.15   # never deploy a threshold below this, even if the F1 curve peaks there


def per_class(metrics, names):
    box = metrics.box
    out = {}
    for k, ci in enumerate(box.ap_class_index):
        f1 = box.f1_curve[k]
        best = int(np.argmax(f1))
        out[names[int(ci)]] = {"threshold": float(box.px[best]), "f1": float(f1[best]), "ap50": float(box.ap50[k]),
                               "precision": float(box.p[k]), "recall": float(box.r[k])}
    return out


def main(weights):
    names = detector_class_names()
    data = os.path.join(DATA_DET, "data.yaml")
    model = YOLO(weights)
    val = per_class(model.val(data=data, split="val", conf=0.001, plots=False, verbose=False), names)
    test = per_class(model.val(data=data, split="test", conf=0.001, plots=False, verbose=False), names)

    thresholds = {c: round(max(MIN_THRESHOLD, v["threshold"]), 2) for c, v in val.items()}
    for c in names:
        thresholds.setdefault(c, 0.5)   # class absent from val: be conservative
    onnx_path = model.export(format="onnx", imgsz=INPUT_SIZE, opset=12, simplify=True, dynamic=False)
    dest = os.path.join(PROJECT, "backend", "ml_models")
    os.makedirs(dest, exist_ok=True)
    shutil.copy(onnx_path, os.path.join(dest, "issue_detector.onnx"))

    cfg = load_classes()["detector"]
    sidecar = {"input_size": INPUT_SIZE, "classes": names, "subcategory": cfg["classes"],
               "thresholds": thresholds, "test_metrics": test, "weights": os.path.relpath(weights, PROJECT)}
    with open(os.path.join(dest, "issue_detector.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"{'class':14s} {'thr':>5s} {'test AP50':>10s} {'test P':>7s} {'test R':>7s}")
    for c in names:
        t = test.get(c)
        print(f"{c:14s} {thresholds[c]:5.2f} " + (f"{t['ap50']:10.3f} {t['precision']:7.3f} {t['recall']:7.3f}"
                                                  if t else "   (no test boxes)"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
