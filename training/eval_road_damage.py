"""
Measure the pretrained road-damage model on our RDD2022 India+Czech test split
(training/data/road_damage/test). CAVEAT: the model was trained on RDD2022's
public train set, which contains these images, so the numbers are optimistic
(train-set contamination); they show the model works, not field accuracy.
Per class, the threshold maximising F1 (IoU 0.5) is chosen for deployment.
"""
import glob, json, os
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(ROOT, "data", "road_damage", "test")
WEIGHTS = os.path.join(os.path.dirname(ROOT), "models", "rdd2022-yolov8n", "best.pt")
# our label ids: 0 pothole, 1 alligator, 2 crack (long+trans), 3 faded marking (not in model)
MODEL_TO_OURS = {"longitudinal_crack": 2, "transverse_crack": 2, "alligator_crack": 1, "pothole": 0}
OURS = {0: "pothole", 1: "alligator_crack", 2: "crack"}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)


def main():
    model = YOLO(WEIGHTS)
    imgs = sorted(glob.glob(os.path.join(TEST, "images", "*.jpg")))
    preds, gts = [], []
    for r in (model.predict(im, conf=0.05, imgsz=640, device="mps", verbose=False)[0] for im in imgs):
        p = [(MODEL_TO_OURS[model.names[int(c)]], float(s), b.tolist())
             for b, s, c in zip(r.boxes.xyxyn, r.boxes.conf, r.boxes.cls)]
        lab = os.path.join(TEST, "labels", os.path.splitext(os.path.basename(r.path))[0] + ".txt")
        g = []
        for line in open(lab):
            c, cx, cy, w, h = map(float, line.split())
            g.append((int(c), [cx - w/2, cy - h/2, cx + w/2, cy + h/2]))
        preds.append(p); gts.append(g)
    per_class, thresholds = {}, {}
    for c, name in OURS.items():
        best = None
        for t in np.arange(0.1, 0.8, 0.05):
            tp = fp = 0; n_gt = sum(1 for g in gts for gc, _ in g if gc == c)
            for p, g in zip(preds, gts):
                gb = [b for gc, b in g if gc == c]; used = set()
                for pc, s, b in sorted(p, key=lambda x: -x[1]):
                    if pc != c or s < t: continue
                    m = max(((iou(b, x), j) for j, x in enumerate(gb) if j not in used), default=(0, None))
                    if m[0] >= 0.5: tp += 1; used.add(m[1])
                    else: fp += 1
            prec = tp / (tp + fp) if tp + fp else 0; rec = tp / n_gt if n_gt else 0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            if best is None or f1 > best["f1"]:
                best = {"threshold": round(float(t), 2), "precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3), "gt_boxes": n_gt}
        per_class[name] = best
        for mn, oc in MODEL_TO_OURS.items():
            if oc == c: thresholds[mn] = best["threshold"]
        print(name, best)
    json.dump({"images": len(imgs), "per_class": per_class, "thresholds": thresholds,
               "caveat": "Model trained on RDD2022 public train set, which includes these test images: optimistic."},
              open(os.path.join(ROOT, "runs", "road_damage_pretrained_eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
