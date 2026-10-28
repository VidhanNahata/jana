"""
Fit the issue classifier head on top of a pretrained CLIP image encoder.

Only a tiny logistic-regression head is trained (seconds on a CPU), so ~100 photos per class are
enough. The CLIP weights are downloaded once by scripts/install_clip.py and never change.

Steps: embed every photo in training/data/issues/{train,val,test} (cached) -> pick the head's
regularisation and each class's confidence threshold on VAL -> report on TEST, next to a
zero-shot baseline (CLIP matching the photo against a text prompt per class, no training) so you
can see how much your photos actually add.

Usage:
    python training/issues/train_clip_head.py [--model models/clip-vit-base-patch32] [--install]
Writes training/runs/issue_clip/{head.npz, head.json, eval_report.json}; --install also copies the
head into backend/ml_models/issue_clip_head.{npz,json}.
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageOps

from clip_head import choose_C, fit_head, head_probs, l2norm, macro_f1, zero_shot_probs
from common import (DATA_CLS, IMG_EXT, PROJECT, RUNS, classifier_class_names, load_classes, metrics_at,
                    pick_threshold)

OUT = os.path.join(RUNS, "issue_clip")
DEFAULT_MODEL = os.path.join(PROJECT, "models", "clip-vit-base-patch32")


def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Clip:
    """Image / text embeddings via the projection layers of transformers' CLIPModel
    (the backend does exactly the same, see backend/services/issue_models.py)."""

    def __init__(self, model_dir):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = pick_device(torch)
        self.processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()

    def images(self, paths, batch=32):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), batch):
                imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths[i:i + batch]]
                px = self.processor(images=imgs, return_tensors="pt")["pixel_values"].to(self.device)
                feats = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
                out.append(feats.float().cpu().numpy())
                print(f"  embedded {min(i + batch, len(paths))}/{len(paths)}", end="\r")
        print()
        return l2norm(np.concatenate(out))

    def texts(self, prompts):
        with self.torch.no_grad():
            tok = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            feats = self.model.text_projection(
                self.model.text_model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]).pooler_output)
        return feats.float().cpu().numpy()


def list_split(split, names):
    paths, labels = [], []
    for label, name in enumerate(names):
        folder = os.path.join(DATA_CLS, split, name)
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    paths.append(os.path.join(folder, f)); labels.append(label)
    return paths, np.array(labels)


def embed_split(clip, split, names):
    paths, y = list_split(split, names)
    cache = os.path.join(OUT, f"emb_{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if list(z["paths"]) == paths:
            return z["X"], y, paths
    print(f"{split}: {len(paths)} photos")
    X = clip.images(paths)
    np.savez(cache, X=X, paths=np.array(paths))
    return X, y, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--precision", type=float, default=0.8)
    ap.add_argument("--min-test", type=int, default=30)
    ap.add_argument("--install", action="store_true", help="copy the head into backend/ml_models")
    args = ap.parse_args()

    if not os.path.isdir(args.model):
        raise SystemExit(f"CLIP model not found at {args.model}. Run: python scripts/install_clip.py")
    os.makedirs(OUT, exist_ok=True)
    names = classifier_class_names()
    cfg = load_classes()["classifier"]
    clip = Clip(args.model)

    Xtr, ytr, _ = embed_split(clip, "train", names)
    Xva, yva, _ = embed_split(clip, "val", names)
    Xte, yte, _ = embed_split(clip, "test", names)
    missing = [n for i, n in enumerate(names) if not np.any(ytr == i)]
    if missing:
        raise SystemExit(f"No training photos for: {', '.join(missing)}")

    C, val_f1 = choose_C(Xtr, ytr, Xva, yva)
    W, b = fit_head(Xtr, ytr, C)
    pva, pte = head_probs(Xva, W, b), head_probs(Xte, W, b)
    n = len(names)
    print(f"regularisation C={C}  val macro-F1 {val_f1:.3f}")

    report = {"C": C, "val_macro_f1": round(val_f1, 3), "target_precision": args.precision,
              "train_photos": {nm: int(np.sum(ytr == i)) for i, nm in enumerate(names)}, "classes": {}}
    for c, name in enumerate(names[:-1]):
        t, met = pick_threshold(pva[:, c], yva == c, args.precision)
        n_test = int(np.sum(yte == c))
        report["classes"][name] = {"threshold": round(t, 2), "precision_target_met_on_val": met,
                                   "val": metrics_at(pva[:, c], yva == c, t),
                                   "test": metrics_at(pte[:, c], yte == c, t),
                                   "test_photos": n_test, "reliable": n_test >= args.min_test}
    bg = n - 1
    bgm = yte == bg
    if bgm.any():
        fired = np.zeros(int(bgm.sum()), bool)
        for c, name in enumerate(names[:-1]):
            fired |= pte[bgm, c] >= report["classes"][name]["threshold"]
        report["background_false_issue_rate"] = round(float(fired.mean()), 3)
    f1_head, _ = macro_f1(yte, pte.argmax(1), n)
    zs = zero_shot_probs(Xte, clip.texts([cfg["prompts"][nm] for nm in names]))
    f1_zs, _ = macro_f1(yte, zs.argmax(1), n)
    report["test_macro_f1"] = {"fitted_head": round(f1_head, 3), "zero_shot_baseline": round(f1_zs, 3)}
    json.dump(report, open(os.path.join(OUT, "eval_report.json"), "w"), indent=2)

    np.savez(os.path.join(OUT, "head.npz"), W=W.astype(np.float32), b=b.astype(np.float32))
    sidecar = {"kind": "clip_head", "clip_model": os.path.basename(args.model.rstrip("/")), "embedding_dim": int(W.shape[1]),
               "classes": names, "background": cfg["background"], "subcategory": cfg["classes"],
               "thresholds": {k: v["threshold"] for k, v in report["classes"].items()},
               "reliable": {k: v["reliable"] for k, v in report["classes"].items()},
               "test_metrics": {k: v["test"] for k, v in report["classes"].items()},
               "background_false_issue_rate": report.get("background_false_issue_rate")}
    json.dump(sidecar, open(os.path.join(OUT, "head.json"), "w"), indent=2)

    print(f"\n{'class':18s} {'thr':>5s} {'test P':>7s} {'test R':>7s} {'n':>4s}")
    for name, r in report["classes"].items():
        t = r["test"]
        print(f"{name:18s} {r['threshold']:5.2f} {str(t['precision']):>7s} {str(t['recall']):>7s} {r['test_photos']:4d}"
              + ("" if r["reliable"] else "  (too few test photos to trust)"))
    print("background false-issue rate:", report.get("background_false_issue_rate"))
    print("test macro-F1:", report["test_macro_f1"], "(zero-shot = no training, your photos add the difference)")

    if args.install:
        dest = os.path.join(PROJECT, "backend", "ml_models")
        os.makedirs(dest, exist_ok=True)
        shutil.copy(os.path.join(OUT, "head.npz"), os.path.join(dest, "issue_clip_head.npz"))
        shutil.copy(os.path.join(OUT, "head.json"), os.path.join(dest, "issue_clip_head.json"))
        print("installed backend/ml_models/issue_clip_head.{npz,json}")
    else:
        print("Review the numbers, then re-run with --install to enable it in the backend.")


if __name__ == "__main__":
    main()
