"""
Build the train/val/test folders for the issue classifier.

You put photos in  training/data/issues_raw/<class>/  (class names are the keys in
classes.json, plus a 'background' folder of normal streets / non-civic photos).
This script:
  - drops unreadable files and tiny images,
  - groups near-duplicate photos by perceptual hash (Roboflow exports contain
    several augmented copies of one photo; public sets also overlap) so a group
    never straddles train and test, which would inflate the test score,
  - splits ~80/10/10 by group, per class (but at least 30 test and 15 val photos per class when the
    class is small, capped at 40% held out), and writes resized copies (<=1024 px).

Usage:
    python training/issues/prepare_classifier.py
"""

import os
import random
import shutil
from collections import defaultdict

from PIL import Image, ImageOps

from common import (DATA_CLS, IMG_EXT, RAW_CLS, classifier_class_names, dhash, group_near_duplicates)

RATIOS = (0.8, 0.1, 0.1)
# With ~100 photos per class a 10% test split is ~10 photos: too few to trust any score. Reserve at least
# this many per class (never more than 40% of the class) so the test numbers mean something.
MIN_TEST = 30
MIN_VAL = 15
MAX_HELDOUT_FRACTION = 0.4
MIN_SIDE = 96
MAX_SIDE = 1024
SEED = 42


def main():
    random.seed(SEED)
    names = classifier_class_names()
    missing = [c for c in names if not os.path.isdir(os.path.join(RAW_CLS, c))]
    if missing:
        raise SystemExit(f"No folder in {RAW_CLS} for: {', '.join(missing)}")
    if os.path.isdir(DATA_CLS):
        shutil.rmtree(DATA_CLS)

    summary = {}
    for cls in names:
        folder = os.path.join(RAW_CLS, cls)
        items = []   # (path, hash)
        for fn in sorted(os.listdir(folder)):
            if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                continue
            path = os.path.join(folder, fn)
            try:
                img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception:
                continue
            if min(img.size) < MIN_SIDE:
                continue
            items.append((path, dhash(img), img))
        groups = defaultdict(list)
        for gid, it in zip(group_near_duplicates([h for _, h, _ in items]), items):
            groups[gid].append(it)
        gids = list(groups)
        random.shuffle(gids)
        # Fill test, then val, then train, by image count, so tiny classes still get a test set
        total = sum(len(groups[g]) for g in gids)
        want = {"test": max(RATIOS[2] * total, MIN_TEST), "val": max(RATIOS[1] * total, MIN_VAL)}
        scale = min(1.0, MAX_HELDOUT_FRACTION * total / (want["test"] + want["val"]))
        want = {k: v * scale for k, v in want.items()}
        splits = {"train": [], "val": [], "test": []}
        counts = {"train": 0, "val": 0, "test": 0}
        for g in gids:
            for s in ("test", "val"):
                if counts[s] < want[s]:
                    splits[s].append(g); counts[s] += len(groups[g]); break
            else:
                splits["train"].append(g); counts["train"] += len(groups[g])
        for split, glist in splits.items():
            out = os.path.join(DATA_CLS, split, cls)
            os.makedirs(out, exist_ok=True)
            n = 0
            for g in glist:
                for path, _, img in groups[g]:
                    if max(img.size) > MAX_SIDE:
                        s = MAX_SIDE / max(img.size)
                        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
                    img.save(os.path.join(out, f"{cls}_{split}_{n:05d}.jpg"), quality=92)
                    n += 1
        summary[cls] = {s: counts[s] for s in counts} | {"duplicate_groups": len(groups), "images": len(items)}

    print(f"{'class':18s} {'images':>7s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for cls, s in summary.items():
        flag = "  <-- too few for a reliable test set (<30 test)" if s["test"] < 30 else ""
        print(f"{cls:18s} {s['images']:7d} {s['train']:6d} {s['val']:5d} {s['test']:5d}{flag}")


if __name__ == "__main__":
    main()
