"""
Build a merged single-class pothole detection dataset (YOLO format) from the
raw sources in training/data/raw/.

Sources (all labelled with bounding boxes or polygons):
  - RDD2022 (CC BY 4.0) — India, Czech, US, China motorbike, Japan dashcam
    images; class D40 = pothole. Images without D40 (cracks, patches) are kept
    in a limited number as hard negatives.
  - MWPD — mixed day / night / wet potholes, YOLO txt
  - rupesh1, rupesh2 — Roboflow YOLO exports
  - manot, keremberke — Roboflow COCO exports

Roboflow exports contain several augmented copies of the same photo and the
public sets overlap each other, so images are grouped by perceptual hash and
each group goes to exactly one split. Without this the test score is inflated
by near-identical copies of training images.

Usage:
    python training/prepare_dataset.py
Output:
    training/data/potholes/{train,val,test}/{images,labels}
    training/data/potholes/data.yaml
    training/data/potholes/test_tags.json   (per-image source/condition tags)
"""

import glob
import json
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "potholes")

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
# Hard negatives (road images with no pothole) as a fraction of positive images
NEGATIVE_RATIO = 0.15
# Longest side saved to disk; keeps the dataset small without hurting 640px training
MAX_SIDE = 1280
# Max images kept from one duplicate group in val/test
HELDOUT_PER_GROUP = 5
SEED = 42


# ── Sample collection ────────────────────────────────────────────────────
# Each sample: {"path", "boxes": [(cx, cy, w, h) normalised], "source", "tags"}

def _yolo_samples(image_glob, source, tags_fn=None):
    samples = []
    skipped = []
    for img in sorted(glob.glob(image_glob)):
        label = re.sub(r"/images/", "/labels/", img)
        label = os.path.splitext(label)[0] + ".txt"
        boxes = []
        if os.path.exists(label):
            lines = [l.split() for l in open(label, errors="replace") if l.strip()]
            try:
                [float(v) for parts in lines for v in parts]
            except ValueError:
                # Some public exports contain corrupt/unrelated label files; drop the image
                skipped.append(label)
                continue
            for parts in lines:
                if len(parts) == 5:
                    boxes.append(tuple(float(v) for v in parts[1:]))
                elif len(parts) > 5:
                    # Polygon row (segmentation export): convert to bounding box
                    xs = [float(v) for v in parts[1::2]]
                    ys = [float(v) for v in parts[2::2]]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    boxes.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        tags = tags_fn(img) if tags_fn else []
        samples.append({"path": img, "boxes": boxes, "source": source, "tags": tags})
    if skipped:
        print(f"{source}: skipped {len(skipped)} images with unparseable label files")
    return samples


def _coco_samples(json_path, source):
    data = json.load(open(json_path))
    folder = os.path.dirname(json_path)
    images = {im["id"]: im for im in data["images"]}
    boxes = defaultdict(list)
    for ann in data["annotations"]:
        im = images[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes[ann["image_id"]].append(
            ((x + w / 2) / im["width"], (y + h / 2) / im["height"], w / im["width"], h / im["height"])
        )
    return [
        {"path": os.path.join(folder, im["file_name"]), "boxes": boxes[im_id], "source": source, "tags": []}
        for im_id, im in images.items()
    ]


def _rdd_samples(country):
    samples = []
    base = os.path.join(RAW, "rdd2022", country, country, "train")
    if not os.path.isdir(base):
        base = os.path.join(RAW, "rdd2022", country, "train")
    for xml_path in sorted(glob.glob(os.path.join(base, "annotations", "xmls", "*.xml"))):
        root = ET.parse(xml_path).getroot()
        img = os.path.join(base, "images", root.findtext("filename"))
        if not os.path.exists(img):
            continue
        w = float(root.findtext("size/width"))
        h = float(root.findtext("size/height"))
        boxes = []
        for obj in root.findall("object"):
            if obj.findtext("name") != "D40":
                continue
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            if x2 > x1 and y2 > y1:
                boxes.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
        samples.append({"path": img, "boxes": boxes, "source": f"rdd_{country.lower()}", "tags": ["dashcam"]})
    return samples


def collect():
    def mwpd_tags(path):
        return ["night"] if "night" in os.path.basename(path).lower() else []

    samples = []
    samples += _yolo_samples(os.path.join(RAW, "mwpd", "MWPD", "*", "images", "*"), "mwpd", mwpd_tags)
    samples += _yolo_samples(os.path.join(RAW, "rupesh1", "*", "*", "images", "*"), "rupesh1")
    samples += _yolo_samples(os.path.join(RAW, "rupesh2", "*", "*", "images", "*"), "rupesh2")
    for d in ("manot", "keremberke"):
        for split in ("train", "valid", "test"):
            p = os.path.join(RAW, d, split, "_annotations.coco.json")
            if os.path.exists(p):
                samples += _coco_samples(p, d)
    for country in ("India", "Czech", "United_States", "China_MotorBike", "Japan"):
        samples += _rdd_samples(country)
    return samples


# ── Near-duplicate grouping ──────────────────────────────────────────────

def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = list(img.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (px[row * (size + 1) + col] > px[row * (size + 1) + col + 1])
    return bits


def group_duplicates(samples, max_distance=6):
    """Union-find over likely duplicates.

    Two signals: Roboflow's augmented copies share a base filename, and
    re-uploads across datasets have near-identical dHashes. Comparing only
    images that share one of four 16-bit hash bands keeps this near-linear;
    it is guaranteed to find pairs within 3 bits and finds most within 6."""
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1. Roboflow augmented copies share the original name before ".rf."
    by_name = defaultdict(list)
    for i, s in enumerate(samples):
        name = os.path.basename(s["path"])
        if ".rf." in name:
            by_name[(s["source"], name.split(".rf.")[0])].append(i)
    for idxs in by_name.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    # 2. Perceptual hash: link images sharing any 16-bit band and within max_distance
    hashes = []
    for s in samples:
        try:
            hashes.append(dhash(s["path"]))
        except Exception:
            hashes.append(None)
    bands = defaultdict(list)
    for i, h in enumerate(hashes):
        if h is None:
            continue
        for b in range(4):
            bands[(b, (h >> (16 * b)) & 0xFFFF)].append(i)
    for idxs in bands.values():
        if len(idxs) > 200:  # uninformative band (e.g. all-black frames); skip
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if bin(hashes[i] ^ hashes[j]).count("1") <= max_distance:
                    union(i, j)

    groups = defaultdict(list)
    for i in range(len(samples)):
        groups[find(i)].append(i)
    return list(groups.values()), hashes


# ── Write YOLO dataset ───────────────────────────────────────────────────

def save_image(src, dst):
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    img.save(dst, quality=92)


def main():
    random.seed(SEED)
    samples = [s for s in collect() if os.path.exists(s["path"])]
    print(f"collected {len(samples)} images")

    groups, _ = group_duplicates(samples)
    print(f"{len(groups)} unique groups after near-duplicate grouping")

    # Whole groups go to one split, so nothing leaks into train. Val/test keep
    # up to HELDOUT_PER_GROUP members (e.g. distinct frames of one night video)
    # so rare conditions are still represented; train keeps all copies.
    random.shuffle(groups)
    positives = [g for g in groups if any(samples[i]["boxes"] for i in g)]
    negatives = [g for g in groups if not any(samples[i]["boxes"] for i in g)]
    negatives = negatives[: int(len(positives) * NEGATIVE_RATIO)]
    chosen = positives + negatives
    random.shuffle(chosen)

    n = len(chosen)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    split_groups = {
        "train": chosen[:n_train],
        "val": chosen[n_train:n_train + n_val],
        "test": chosen[n_train + n_val:],
    }

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    tags = {}
    stats = {}
    for split, sgroups in split_groups.items():
        os.makedirs(os.path.join(OUT, split, "images"))
        os.makedirs(os.path.join(OUT, split, "labels"))
        count = boxes = 0
        for g in sgroups:
            members = g if split == "train" else g[:HELDOUT_PER_GROUP]
            for i in members:
                s = samples[i]
                stem = f"{s['source']}_{i:06d}"
                save_image(s["path"], os.path.join(OUT, split, "images", stem + ".jpg"))
                with open(os.path.join(OUT, split, "labels", stem + ".txt"), "w") as f:
                    for cx, cy, w, h in s["boxes"]:
                        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
                        w, h = min(w, 1), min(h, 1)
                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                if split == "test":
                    tags[stem] = {"source": s["source"], "tags": s["tags"]}
                count += 1
                boxes += len(s["boxes"])
        stats[split] = (count, boxes)
        print(f"{split}: {count} images, {boxes} pothole boxes")

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                "nc: 1\nnames: ['pothole']\n")
    with open(os.path.join(OUT, "test_tags.json"), "w") as f:
        json.dump(tags, f)


if __name__ == "__main__":
    main()
