"""Shared helpers for the issue classifier / detector training scripts."""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(ROOT)
PROJECT = os.path.dirname(TRAINING)
CLASSES_JSON = os.path.join(ROOT, "classes.json")
RAW_CLS = os.path.join(TRAINING, "data", "issues_raw")        # <class>/*.jpg dropped here by you
DATA_CLS = os.path.join(TRAINING, "data", "issues")           # {train,val,test}/<class>/*.jpg
DATA_DET = os.path.join(TRAINING, "data", "issues_det")       # YOLO format
RUNS = os.path.join(TRAINING, "runs")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
INPUT_CLS = 256   # classifier input side (squashed resize; same at train and inference)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_classes():
    with open(CLASSES_JSON) as f:
        return json.load(f)


def classifier_class_names():
    """Ordered class list; index = output logit. 'background' is always last."""
    cfg = load_classes()["classifier"]
    return list(cfg["classes"]) + [cfg["background"]]


def detector_class_names():
    return list(load_classes()["detector"]["classes"])


def dhash(img, size=8):
    """64-bit difference hash; near-duplicate photos get (almost) the same value."""
    g = img.convert("L").resize((size + 1, size))
    px = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def group_near_duplicates(hashes, max_dist=4):
    """Union-find over hashes; returns a group id per index. O(n^2), fine for <50k images."""
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i], hashes[j]) <= max_dist:
                parent[find(j)] = find(i)
    return [find(i) for i in range(len(hashes))]


def pick_threshold(scores, is_pos, target_precision):
    """Lowest threshold with precision >= target (max recall at that precision), else best-F1."""
    best_f1, best_t, ok_t = -1.0, 0.5, None
    for t in np.linspace(0.05, 0.95, 91):
        pred = scores >= t
        tp = int(np.sum(pred & is_pos)); fp = int(np.sum(pred & ~is_pos)); fn = int(np.sum(~pred & is_pos))
        if tp + fp == 0:
            continue
        p, f1 = tp / (tp + fp), 2 * tp / (2 * tp + fp + fn)
        if p >= target_precision and tp > 0 and ok_t is None:
            ok_t = float(t)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return (ok_t if ok_t is not None else best_t), ok_t is not None


def metrics_at(scores, is_pos, t):
    pred = scores >= t
    tp = int(np.sum(pred & is_pos)); fp = int(np.sum(pred & ~is_pos)); fn = int(np.sum(~pred & is_pos))
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
