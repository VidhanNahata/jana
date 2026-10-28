"""
Export the trained issue classifier to ONNX and install it in the backend.

  backend/ml_models/issue_classifier.onnx
  backend/ml_models/issue_classifier.json   (class order, taxonomy mapping, thresholds, test metrics)

Run evaluate_classifier.py first: the thresholds come from its eval_report.json.

Usage:
    python training/issues/export_classifier.py
"""

import json
import os
import shutil

import torch

from common import INPUT_CLS, PROJECT, RUNS, classifier_class_names, load_classes
from train_classifier import build_model

OUT = os.path.join(RUNS, "issue_classifier")


def main():
    names = classifier_class_names()
    ckpt = torch.load(os.path.join(OUT, "best.pt"), map_location="cpu")
    report = json.load(open(os.path.join(OUT, "eval_report.json")))
    if ckpt["classes"] != names:
        raise SystemExit("classes.json changed since training; retrain.")
    model = build_model(len(names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    onnx_path = os.path.join(OUT, "issue_classifier.onnx")
    torch.onnx.export(model, torch.zeros(1, 3, INPUT_CLS, INPUT_CLS), onnx_path, opset_version=17,
                      input_names=["image"], output_names=["logits"], dynamic_axes={"image": {0: "batch"}})
    dest = os.path.join(PROJECT, "backend", "ml_models")
    os.makedirs(dest, exist_ok=True)
    shutil.copy(onnx_path, os.path.join(dest, "issue_classifier.onnx"))

    cfg = load_classes()["classifier"]
    sidecar = {
        "input_size": INPUT_CLS, "classes": names, "background": cfg["background"],
        "subcategory": cfg["classes"],
        "thresholds": {k: v["threshold"] for k, v in report["classes"].items()},
        "reliable": {k: v["reliable"] for k, v in report["classes"].items()},
        "test_metrics": {k: v["test"] for k, v in report["classes"].items()},
        "background_false_issue_rate": report.get("background_false_issue_rate"),
        "checkpoint_epoch": ckpt["epoch"],
    }
    with open(os.path.join(dest, "issue_classifier.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"installed issue_classifier.onnx ({os.path.getsize(onnx_path) / 1e6:.1f} MB) + json in {dest}")
    unreliable = [k for k, v in sidecar["reliable"].items() if not v]
    if unreliable:
        print("WARNING: too few test images to trust the metrics for:", ", ".join(unreliable))


if __name__ == "__main__":
    main()
