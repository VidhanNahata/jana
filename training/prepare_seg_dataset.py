"""
Build the pothole instance-segmentation dataset (YOLO polygon format) from the
detection dataset in training/data/potholes/ (same images and splits).

Masks:
  - manot, keremberke: their hand-drawn COCO polygons
  - every other source only has boxes, so each box is turned into a mask by
    SAM 2.1-tiny prompted with that box (mean IoU 0.80 against the hand-drawn
    manot polygons, vs 0.69 for using the box itself as the mask)

Resumable: label files that already exist are kept, and --limit caps the number
of images per process (PyTorch MPS memory grows in long runs).

Usage:
    python training/prepare_seg_dataset.py [--limit 1500]
Output:
    training/data/potholes_seg/{train,val,test}/{images (hard links to potholes/<split>/images), labels}
    training/data/potholes_seg/data.yaml
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from prepare_dataset import OUT as DET, RAW, collect  # noqa: E402

OUT = os.path.join(ROOT, "data", "potholes_seg")
SAM_MODEL = "sam2.1_t.pt"
# Polygon simplification tolerance, as a fraction of the contour perimeter
SIMPLIFY = 0.005


def coco_polygons():
    """Raw image path -> list of normalised polygons, for the COCO sources."""
    polys = defaultdict(list)
    for d in ("manot", "keremberke"):
        for p in glob.glob(os.path.join(RAW, d, "*", "_annotations.coco.json")):
            data = json.load(open(p))
            images = {im["id"]: im for im in data["images"]}
            for ann in data["annotations"]:
                im = images[ann["image_id"]]
                path = os.path.join(os.path.dirname(p), im["file_name"])
                for seg in ann.get("segmentation") or []:
                    if len(seg) >= 6:
                        xy = np.array(seg, np.float64).reshape(-1, 2) / (im["width"], im["height"])
                        polys[path].append(np.clip(xy, 0, 1))
    return polys


def mask_to_polygon(mask):
    """Largest outer contour of a binary mask, simplified; None if empty."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    c = cv2.approxPolyDP(c, SIMPLIFY * cv2.arcLength(c, True), True)
    return c.reshape(-1, 2) if len(c) >= 3 else None


def link_images(split):
    """Hard-link the detection images (no extra disk space). A symlinked folder
    doesn't work: Ultralytics resolves it and then reads the detection boxes
    from potholes/<split>/labels instead of these polygon labels."""
    d = os.path.join(OUT, split, "images")
    if os.path.islink(d):
        os.remove(d)
    os.makedirs(d, exist_ok=True)
    for f in glob.glob(os.path.join(DET, split, "images", "*.jpg")):
        dst = os.path.join(d, os.path.basename(f))
        if not os.path.exists(dst):
            os.link(f, dst)


def sam_one(sam, img_path, box):
    """SAM mask for a single box, or None when SAM returns nothing (degenerate boxes)."""
    m = sam(img_path, bboxes=[box], verbose=False)[0].masks
    return m.data.cpu().numpy()[0] > 0.5 if m is not None and len(m.data) else None


def write_label(path, polys):
    with open(path, "w") as f:
        for xy in polys:
            f.write("0 " + " ".join(f"{v:.5f}" for v in xy.reshape(-1)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max images to label in this process (0 = all)")
    parser.add_argument("--splits", default="train,val,test", help="comma-separated splits to label")
    parser.add_argument("--stems", default=None, help="file listing image stems to label (others skipped)")
    args = parser.parse_args()

    samples = [s for s in collect() if os.path.exists(s["path"])]  # same indexing as prepare_dataset.py
    gt = coco_polygons()
    only = set(open(args.stems).read().split()) if args.stems else None
    sam = None
    done = 0
    stats = defaultdict(int)
    for split in args.splits.split(","):
        os.makedirs(os.path.join(OUT, split, "labels"), exist_ok=True)
        link_images(split)
        for label in sorted(glob.glob(os.path.join(DET, split, "labels", "*.txt"))):
            stem = os.path.splitext(os.path.basename(label))[0]
            out = os.path.join(OUT, split, "labels", stem + ".txt")
            if os.path.exists(out) or (only is not None and stem not in only):
                continue
            if args.limit and done >= args.limit:
                print(f"limit reached; {dict(stats)}")
                return
            boxes = [list(map(float, l.split()[1:])) for l in open(label) if l.strip()]
            src = samples[int(stem.rsplit("_", 1)[1])]
            assert src["source"] == stem.rsplit("_", 1)[0], f"index mismatch for {stem}"
            if not boxes:
                write_label(out, [])
                stats["negative"] += 1
            elif src["path"] in gt:
                write_label(out, gt[src["path"]])
                stats["hand_drawn"] += 1
            else:
                if sam is None:
                    from ultralytics import SAM
                    sam = SAM(SAM_MODEL)
                img_path = os.path.join(DET, split, "images", stem + ".jpg")
                h, w = cv2.imread(img_path).shape[:2]
                xyxy = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                        for cx, cy, bw, bh in boxes]
                try:
                    result = sam(img_path, bboxes=xyxy, verbose=False)[0].masks
                    masks = result.data.cpu().numpy() > 0.5 if result is not None else []
                    if len(masks) != len(xyxy):
                        # SAM merged/dropped a prompt, so masks no longer line up with boxes: one call per box
                        masks = [sam_one(sam, img_path, b) for b in xyxy]
                except Exception as exc:  # never lose an image to a SAM failure; boxes are used instead
                    print(f"SAM failed on {stem}: {exc}", flush=True)
                    masks = []
                polys = []
                for k, (x0, y0, x1, y1) in enumerate(xyxy):
                    poly = None
                    # SAM can return fewer masks than boxes (e.g. duplicate boxes)
                    if k < len(masks) and masks[k] is not None:
                        m = cv2.resize(masks[k].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        # Keep the mask inside its box; SAM sometimes spills onto the road
                        clip = np.zeros_like(m)
                        clip[int(max(y0, 0)):int(np.ceil(y1)), int(max(x0, 0)):int(np.ceil(x1))] = 1
                        poly = mask_to_polygon(m & clip)
                    if poly is None:
                        # Fall back to the box itself so no pothole is lost
                        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                        stats["box_fallback"] += 1
                    polys.append(np.clip(poly / (w, h), 0, 1))
                write_label(out, polys)
                stats["sam"] += 1
            done += 1
            if done % 250 == 0:
                print(split, done, dict(stats), flush=True)

    if args.splits != "train,val,test":
        print(f"done with {args.splits}; {dict(stats)}")
        return
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                "nc: 1\nnames: ['pothole']\n")
    print(f"all labels written; {dict(stats)}")
    open(os.path.join(OUT, "DONE"), "w").close()


if __name__ == "__main__":
    main()
