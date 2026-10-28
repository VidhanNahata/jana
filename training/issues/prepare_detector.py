"""
Merge several YOLO-format datasets into one multi-class issue-detection dataset.

For every source listed in detector_sources.json:
  - class ids are remapped BY NAME to ours (unmapped classes are dropped),
  - polygon labels are turned into boxes,
  - images with no mapped object are kept as hard negatives (up to negative_ratio).
All images from all sources are then grouped by perceptual hash (Roboflow exports contain
augmented copies of one photo and public sets overlap) and each group is assigned to exactly
one of train/val/test (80/10/10), so the test score isn't inflated by near-identical copies.

Usage:
    python training/issues/prepare_detector.py [--sources training/issues/detector_sources.json]
Output:
    training/data/issues_det/{train,val,test}/{images,labels}, data.yaml, summary.json
"""

import argparse
import glob
import json
import os
import random
import shutil
from collections import Counter

import yaml
from PIL import Image, ImageOps

from common import DATA_DET, IMG_EXT, PROJECT, ROOT, dhash, detector_class_names, group_near_duplicates

MAX_SIDE = 1280
SEED = 42


def source_names(src_dir):
    for cand in ("data.yaml", "dataset.yaml", "data.yml"):
        p = os.path.join(src_dir, cand)
        if os.path.exists(p):
            names = yaml.safe_load(open(p))["names"]
            return names if isinstance(names, list) else [names[k] for k in sorted(names)]
    raise SystemExit(f"{src_dir}: no data.yaml")


def to_box(parts):
    """YOLO label line -> (cls, cx, cy, w, h); polygon lines become their bounding box."""
    cls, vals = int(float(parts[0])), [float(v) for v in parts[1:]]
    if len(vals) == 4:
        return cls, *vals
    xs, ys = vals[0::2], vals[1::2]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    return cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def collect(source, ours):
    """Yield (image_path, [boxes in our class ids]) for one source."""
    src_dir = os.path.join(PROJECT, source["path"])
    names = source_names(src_dir)
    cmap = {k.strip().lower(): v for k, v in source["class_map"].items()}
    remap = {i: ours.index(cmap[n.strip().lower()]) for i, n in enumerate(names)
             if n.strip().lower() in cmap and cmap[n.strip().lower()] in ours}
    if not remap:
        print(f"WARNING {source['name']}: none of {names} match class_map")
    for img in sorted(p for p in glob.glob(os.path.join(src_dir, "**", "*"), recursive=True)
                      if os.path.splitext(p)[1].lower() in IMG_EXT):
        label = img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        label = os.path.splitext(label)[0] + ".txt"
        boxes = []
        if os.path.exists(label):
            for line in open(label).read().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls, cx, cy, w, h = to_box(parts)
                except ValueError:
                    continue
                if cls in remap and w > 0 and h > 0:
                    boxes.append((remap[cls], cx, cy, w, h))
        yield img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=os.path.join(ROOT, "detector_sources.json"))
    args = ap.parse_args()
    random.seed(SEED)
    cfg = json.load(open(args.sources))
    ours = detector_class_names()

    pos, neg = [], []
    for source in cfg["sources"]:
        n0 = len(pos)
        for img, boxes in collect(source, ours):
            (pos if boxes else neg).append((source["name"], img, boxes))
        print(f"{source['name']}: {len(pos) - n0} images with objects")
    if not pos:
        raise SystemExit("No labelled images found.")
    random.shuffle(neg)
    items = pos + neg[: int(len(pos) * cfg.get("negative_ratio", 0.15))]

    opened, hashes = [], []
    for source, path, boxes in items:
        try:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        except Exception:
            continue
        opened.append((source, path, boxes, img))
        hashes.append(dhash(img))
    groups = {}
    for gid, item in zip(group_near_duplicates(hashes), opened):
        groups.setdefault(gid, []).append(item)
    gids = list(groups)
    random.shuffle(gids)
    total = len(opened)
    want = {"test": 0.1 * total, "val": 0.1 * total}
    counts, assign = Counter(), {}
    for g in gids:
        for s in ("test", "val"):
            if counts[s] < want[s]:
                assign[g] = s; counts[s] += len(groups[g]); break
        else:
            assign[g] = "train"; counts["train"] += len(groups[g])

    if os.path.isdir(DATA_DET):
        shutil.rmtree(DATA_DET)
    per_class = {s: Counter() for s in ("train", "val", "test")}
    n = 0
    for g, items_g in groups.items():
        split = assign[g]
        for source, _, boxes, img in items_g:
            os.makedirs(os.path.join(DATA_DET, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(DATA_DET, split, "labels"), exist_ok=True)
            if max(img.size) > MAX_SIDE:
                s = MAX_SIDE / max(img.size)
                img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
            stem = f"{source}_{n:06d}"
            img.save(os.path.join(DATA_DET, split, "images", stem + ".jpg"), quality=92)
            with open(os.path.join(DATA_DET, split, "labels", stem + ".txt"), "w") as f:
                for cls, cx, cy, w, h in boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    per_class[split][ours[cls]] += 1
            n += 1

    with open(os.path.join(DATA_DET, "data.yaml"), "w") as f:
        yaml.safe_dump({"path": os.path.abspath(DATA_DET), "train": "train/images", "val": "val/images",
                        "test": "test/images", "names": {i: c for i, c in enumerate(ours)}}, f)
    summary = {"images": dict(counts), "boxes_per_class": {s: dict(c) for s, c in per_class.items()}}
    json.dump(summary, open(os.path.join(DATA_DET, "summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    thin = [c for c in ours if per_class["test"][c] < 30]
    if thin:
        print("WARNING: fewer than 30 test boxes for", ", ".join(thin), "- metrics for these will be unreliable")
    print("Next:  python training/train.py --model yolo11n.pt --data training/data/issues_det/data.yaml "
          "--name issues_yolo11n --epochs 80")


if __name__ == "__main__":
    main()
