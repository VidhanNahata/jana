"""
Evaluate the trained issue classifier and choose a confidence threshold per class.

Thresholds are chosen on the VALIDATION split (the lowest threshold whose precision
reaches --precision, else the best-F1 threshold) and then reported on the held-out
TEST split, so the quoted numbers were never used for tuning. A class with fewer
than --min-test test images is marked unreliable in the report.

Usage:
    python training/issues/evaluate_classifier.py [--precision 0.8]
Writes training/runs/issue_classifier/eval_report.json
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import RUNS, classifier_class_names, metrics_at, pick_threshold
from train_classifier import IssueDataset, build_model, pick_device, predict

OUT = os.path.join(RUNS, "issue_classifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", type=float, default=0.8)
    ap.add_argument("--min-test", type=int, default=30)
    ap.add_argument("--weights", default=os.path.join(OUT, "best.pt"))
    args = ap.parse_args()

    names = classifier_class_names()
    device = pick_device()
    ckpt = torch.load(args.weights, map_location="cpu")
    if ckpt["classes"] != names:
        raise SystemExit("classes.json changed since training; retrain.")
    model = build_model(len(names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    loaders = {s: DataLoader(IssueDataset(s, False, names), 32, num_workers=2) for s in ("val", "test")}
    vp, vy = predict(model, loaders["val"], device)
    tp_, ty = predict(model, loaders["test"], device)
    bg = len(names) - 1

    report = {"classes": {}, "target_precision": args.precision, "weights": args.weights}
    for c, name in enumerate(names[:-1]):
        t, met_target = pick_threshold(vp[:, c], vy == c, args.precision)
        n_test = int(np.sum(ty == c))
        report["classes"][name] = {
            "threshold": round(t, 2), "precision_target_met_on_val": met_target,
            "val": metrics_at(vp[:, c], vy == c, t), "test": metrics_at(tp_[:, c], ty == c, t),
            "test_images": n_test, "reliable": n_test >= args.min_test}
    pred = tp_.argmax(1)
    cm = np.zeros((len(names), len(names)), int)
    for a, b in zip(ty, pred):
        cm[a, b] += 1
    report["test_accuracy_argmax"] = round(float((pred == ty).mean()), 3)
    report["confusion_matrix"] = {"rows_true": names, "cols_pred": names, "values": cm.tolist()}
    # What fraction of 'background' photos would raise a false issue at the chosen thresholds?
    bgm = ty == bg
    if bgm.any():
        fired = np.zeros(bgm.sum(), bool)
        for c, name in enumerate(names[:-1]):
            fired |= tp_[bgm, c] >= report["classes"][name]["threshold"]
        report["background_false_issue_rate"] = round(float(fired.mean()), 3)
    with open(os.path.join(OUT, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'class':18s} {'thr':>5s} {'test P':>7s} {'test R':>7s} {'n':>5s}")
    for name, r in report["classes"].items():
        t = r["test"]
        print(f"{name:18s} {r['threshold']:5.2f} {str(t['precision']):>7s} {str(t['recall']):>7s} "
              f"{r['test_images']:5d}{'' if r['reliable'] else '  (too few test images)'}")
    print("background false-issue rate:", report.get("background_false_issue_rate"))


if __name__ == "__main__":
    main()
