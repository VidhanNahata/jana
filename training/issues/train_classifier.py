"""
Train the all-category issue classifier (EfficientNet-B0, ImageNet-pretrained).

One photo -> one of: the 8 issue classes in classes.json, or 'background'.
Class imbalance is handled with a (square-root) inverse-frequency loss weight
and label smoothing. The best epoch is chosen by validation macro-F1, not
accuracy, so a big 'background' class can't hide a weak small class.

Usage:
    python training/issues/train_classifier.py --epochs 15
Writes training/runs/issue_classifier/{best.pt, history.json}
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from common import DATA_CLS, IMG_EXT, INPUT_CLS, MEAN, RUNS, STD, classifier_class_names

OUT = os.path.join(RUNS, "issue_classifier")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def eval_transform():
    return transforms.Compose([transforms.Resize((INPUT_CLS, INPUT_CLS)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_CLS, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        # phone photos: night, glare, wet roads, blur
        transforms.ColorJitter(0.35, 0.35, 0.3, 0.04),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])


class IssueDataset(Dataset):
    def __init__(self, split, train, names):
        self.items = []
        for label, name in enumerate(names):
            folder = os.path.join(DATA_CLS, split, name)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    self.items.append((os.path.join(folder, f), label))
        self.tf = train_transform() if train else eval_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def macro_f1(y_true, y_pred, n):
    f1s = []
    for c in range(n):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)), f1s


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    names = classifier_class_names()
    device = pick_device()
    train_ds, val_ds = IssueDataset("train", True, names), IssueDataset("val", False, names)
    if not len(train_ds) or not len(val_ds):
        raise SystemExit("No data. Run prepare_classifier.py first.")
    counts = np.bincount([y for _, y in train_ds.items], minlength=len(names)).astype(float)
    empty = [n for n, c in zip(names, counts) if c == 0]
    if empty:
        raise SystemExit(f"No training images for: {', '.join(empty)}")
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}")
    print({n: int(c) for n, c in zip(names, counts)})

    # Mild rebalancing: sample ~sqrt-inverse-frequency AND weight the loss the same way
    cls_w = (counts.max() / counts) ** 0.5
    sampler = WeightedRandomSampler([cls_w[y] for _, y in train_ds.items], num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, args.batch, num_workers=args.workers)

    model = build_model(len(names)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cls_w / cls_w.mean(), dtype=torch.float32).to(device),
                                  label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))

    os.makedirs(OUT, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
        probs, labels = predict(model, val_dl, device)
        f1, per = macro_f1(labels, probs.argmax(1), len(names))
        acc = float((probs.argmax(1) == labels).mean())
        history.append({"epoch": epoch, "loss": run_loss / len(train_dl), "val_acc": acc, "val_macro_f1": f1,
                        "val_f1_per_class": dict(zip(names, per))})
        print(f"epoch {epoch:2d}  loss {history[-1]['loss']:.3f}  val acc {acc:.3f}  macro-F1 {f1:.3f}  "
              f"({time.time() - t0:.0f}s)")
        if f1 > best:
            best = f1
            torch.save({"state_dict": model.state_dict(), "classes": names, "epoch": epoch, "val_macro_f1": f1},
                       os.path.join(OUT, "best.pt"))
        with open(os.path.join(OUT, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    print(f"best val macro-F1 {best:.3f} -> {OUT}/best.pt")


if __name__ == "__main__":
    main()
