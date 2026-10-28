"""
Train the photo authenticity classifier: real / ai_generated / screen_photo.

EfficientNet-B0 (ImageNet-pretrained) on 256 px crops taken near native
resolution, because the evidence (generator upsampling traces, screen pixel
grids, moire) is high-frequency and is destroyed by squashing the whole image
down to 224 px.

Every image, whatever its class, goes through the same random downscale and
JPEG re-encode, so the model can't separate classes by file format or
compression history (GenImage stores real photos as JPEG and fakes as PNG).

Usage:
    python training/authenticity/train_authenticity.py --epochs 8
Writes training/runs/authenticity/{best.pt, history.json}
"""

import argparse
import io
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data", "authenticity")
OUT = os.path.join(ROOT, "..", "runs", "authenticity")
CLASSES = ["real", "ai_generated", "screen_photo"]
CROP = 256
# Inference resizes the shorter side to this before taking crops
INFER_SHORT_SIDE = 512
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def resize_short(img, short):
    w, h = img.size
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC) if s < 1 else img


class AuthenticityDataset(Dataset):
    def __init__(self, split, train):
        self.items = []
        for label, name in enumerate(CLASSES):
            folder = os.path.join(DATA, split, name)
            for f in sorted(os.listdir(folder)):
                self.items.append((os.path.join(folder, f), label))
        self.train = train
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.train:
            img = resize_short(img, random.randint(320, 640))
            img = jpeg(img, random.randint(55, 95))
            img = transforms.RandomCrop(CROP, pad_if_needed=True, padding_mode="reflect")(img)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = transforms.ColorJitter(0.15, 0.15, 0.1, 0.02)(img)
        else:
            # Deterministic, mirrors the app: shorter side 512, JPEG 85, center crop
            img = jpeg(resize_short(img, INFER_SHORT_SIDE), 85)
            img = transforms.CenterCrop(CROP)(img)
        return self.norm(img), label


def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    return m


def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            for t, p in zip(y.tolist(), pred.tolist()):
                confusion[t, p] += 1
    recall = confusion.diagonal() / confusion.sum(1).clip(min=1)
    return float(confusion.diagonal().sum() / confusion.sum()), recall.tolist(), confusion.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    device = torch.device(args.device)

    train_ds, val_ds = AuthenticityDataset("train", True), AuthenticityDataset("val", False)
    counts = np.bincount([l for _, l in train_ds.items], minlength=len(CLASSES))
    print("train counts", dict(zip(CLASSES, counts.tolist())), "val", len(val_ds))
    # Balance classes with a weighted loss rather than resampling
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=device)

    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    history, best = [], -1
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for step, (x, y) in enumerate(train_dl):
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_dl)} loss {loss.item():.3f}", flush=True)
        acc, recall, confusion = evaluate(model, val_dl, device)
        # Select on the worst class, not overall accuracy, so no class is neglected
        score = min(recall)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_dl), "val_acc": acc,
                        "val_recall": dict(zip(CLASSES, recall)), "confusion": confusion})
        print(f"EPOCH {epoch + 1}: loss {total / len(train_dl):.3f} val_acc {acc:.3f} "
              f"recall {dict(zip(CLASSES, [round(r, 3) for r in recall]))}", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(OUT, "best.pt"))
        json.dump(history, open(os.path.join(OUT, "history.json"), "w"), indent=2)
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
