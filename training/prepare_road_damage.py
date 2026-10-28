"""
Build the multi-class road-damage dataset for dashcam mode (YOLO format) from
RDD2022 (CC BY 4.0, Arya et al.) India and Czech dashcam/smartphone images.

Classes and the RDD2022 codes merged into them:
  0 pothole           D40
  1 alligator_crack   D20
  2 crack             D00 (longitudinal), D10 (transverse)
  3 faded_marking     D43 (crosswalk blur), D44 (white line blur) - India only

D01/D11 (construction joints) and D50 (manhole cover present) are not defects
and are dropped. Their images are kept, labelled only with the classes above.

RDD2022 images are frames from drives, so neighbouring file numbers often show
the same stretch of road. Images are grouped by blocks of BLOCK consecutive
file numbers and by perceptual hash, and every group goes to one split, so the
test score is not inflated by near-copies of training frames.

Usage:
    python training/prepare_road_damage.py
Output:
    training/data/road_damage/{train,val,test}/{images,labels}, data.yaml, manifest.json
"""

import json
import os
import random
import re
import shutil
import glob
import xml.etree.ElementTree as ET
from collections import Counter

from prepare_dataset import RAW, ROOT, SEED, group_duplicates, save_image

OUT = os.path.join(ROOT, "data", "road_damage")
COUNTRIES = ("India", "Czech")
CLASSES = ["pothole", "alligator_crack", "crack", "faded_marking"]
CODE_TO_CLASS = {"D40": 0, "D20": 1, "D00": 2, "D10": 2, "D43": 3, "D44": 3}
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
BLOCK = 10
# dHash distance for near-duplicates. Dashcam frames share a sky/road layout, so the
# default 6 chains thousands of different roads into one group; 1 keeps true copies.
HASH_DISTANCE = 1
# Images with none of the classes, as a fraction of labelled images
NEGATIVE_RATIO = 0.1


def collect():
    samples = []
    for country in COUNTRIES:
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
                cls = CODE_TO_CLASS.get(obj.findtext("name"))
                if cls is None:
                    continue
                b = obj.find("bndbox")
                x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                if x2 > x1 and y2 > y1:
                    boxes.append((cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
            num = int(re.search(r"(\d+)", os.path.basename(img)).group(1))
            samples.append({"path": img, "boxes": boxes, "source": f"rdd_{country.lower()}",
                            "block": (country, num // BLOCK)})
    return samples


def main():
    random.seed(SEED)
    samples = collect()
    print(f"collected {len(samples)} images")

    hash_groups, _ = group_duplicates(samples, max_distance=HASH_DISTANCE)
    # Merge perceptual-hash groups with file-number blocks (union-find over group ids)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for gi, g in enumerate(hash_groups):
        for i in g:
            parent[find(("img", i))] = find(("hash", gi))
            parent[find(("img", i))] = find(("block",) + samples[i]["block"])
    groups = {}
    for i in range(len(samples)):
        groups.setdefault(find(("img", i)), []).append(i)
    groups = list(groups.values())
    print(f"{len(groups)} leakage groups, largest {max(map(len, groups))} images")

    random.shuffle(groups)
    positives = [g for g in groups if any(samples[i]["boxes"] for i in g)]
    negatives = [g for g in groups if not any(samples[i]["boxes"] for i in g)]
    neg_images = 0
    kept_neg = []
    pos_images = sum(len(g) for g in positives)
    for g in negatives:
        if neg_images >= pos_images * NEGATIVE_RATIO:
            break
        kept_neg.append(g)
        neg_images += len(g)
    chosen = positives + kept_neg
    random.shuffle(chosen)

    n = len(chosen)
    n_train, n_val = int(n * SPLIT_RATIOS["train"]), int(n * SPLIT_RATIOS["val"])
    split_groups = {"train": chosen[:n_train], "val": chosen[n_train:n_train + n_val],
                    "test": chosen[n_train + n_val:]}

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    manifest = {"source": "RDD2022 (CC BY 4.0), countries " + ", ".join(COUNTRIES), "classes": CLASSES,
                "code_to_class": CODE_TO_CLASS, "block": BLOCK, "seed": SEED, "splits": {}}
    for split, sgroups in split_groups.items():
        os.makedirs(os.path.join(OUT, split, "images"))
        os.makedirs(os.path.join(OUT, split, "labels"))
        counts, images = Counter(), 0
        for g in sgroups:
            for i in g:
                s = samples[i]
                stem = f"{s['source']}_{i:06d}"
                save_image(s["path"], os.path.join(OUT, split, "images", stem + ".jpg"))
                with open(os.path.join(OUT, split, "labels", stem + ".txt"), "w") as f:
                    for cls, cx, cy, w, h in s["boxes"]:
                        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {min(w, 1):.6f} {min(h, 1):.6f}\n")
                        counts[CLASSES[cls]] += 1
                images += 1
        manifest["splits"][split] = {"images": images, "boxes": dict(counts)}
        print(f"{split}: {images} images, {dict(counts)}")

    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
