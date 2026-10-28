"""
Export a trained pothole model to ONNX and install it in the app.

  - frontend/public/models/pothole.onnx + pothole.json  (on-device detection)
  - backend/ml_models/pothole.onnx                      (server-side verification)

The deployment threshold comes from the run's eval_report.json (written by
evaluate.py), so evaluate before exporting.

Usage:
    python training/export.py training/runs/yolo11n/weights/best.pt
"""

import json
import os
import shutil
import sys

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
INPUT_SIZE = 640


def main(weights):
    run_dir = os.path.dirname(os.path.dirname(weights))
    report = json.load(open(os.path.join(run_dir, "eval_report.json")))

    onnx_path = YOLO(weights).export(format="onnx", imgsz=INPUT_SIZE, opset=12, simplify=True, dynamic=False)

    frontend_dir = os.path.join(PROJECT, "frontend", "public", "models")
    backend_dir = os.path.join(PROJECT, "backend", "ml_models")
    os.makedirs(frontend_dir, exist_ok=True)
    shutil.copy(onnx_path, os.path.join(frontend_dir, "pothole.onnx"))
    shutil.copy(onnx_path, os.path.join(backend_dir, "pothole.onnx"))

    config = {
        "threshold": report["chosen_threshold"],
        "inputSize": INPUT_SIZE,
        "model": os.path.relpath(weights, PROJECT),
        "testMetrics": report["at_threshold"],
    }
    with open(os.path.join(frontend_dir, "pothole.json"), "w") as f:
        json.dump(config, f, indent=2)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"exported {onnx_path} ({size_mb:.1f} MB), threshold {config['threshold']}")


if __name__ == "__main__":
    main(sys.argv[1])
