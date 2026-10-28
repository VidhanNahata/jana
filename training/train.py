"""
Train the JanaSutra pothole detector (single class) with Ultralytics YOLO.

Starts from COCO-pretrained weights and fine-tunes on the merged dataset built
by prepare_dataset.py. Augmentation is tuned for the conditions the live
camera sees: night/low light (wide brightness jitter), wet roads and
reflections (saturation/hue jitter), small distant potholes (scale jitter +
mosaic), and arbitrary phone tilt (small rotation).

Usage:
    python training/train.py --model yolo11n.pt --epochs 80
    python training/train.py --model yolo11s.pt --epochs 80 --batch 8
    python training/train.py --resume training/runs/yolo11n/weights/last.pt --batch 6
    # Instance segmentation, starting from the trained detector's weights
    python training/train.py --model yolo11n-seg.yaml --init runs/pothole_yolo11n/weights/deployed_epoch11.pt \
        --data data/potholes_seg/data.yaml --name pothole_yolo11n_seg --epochs 12 --batch 6
"""

import argparse
import os

from ultralytics import YOLO
from ultralytics.utils import nms

# Validation runs NMS at conf 0.001; on MPS an early-training model produces so
# many candidates that the default 2-3 s limit cuts NMS off and silently drops
# predictions, making val recall (and best.pt selection) wrong. Allow more time.
_nms = nms.non_max_suppression
nms.non_max_suppression = lambda *a, **k: _nms(*a, **{"max_time_img": 1.0, **k})

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "potholes", "data.yaml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--init", default=None, help="weights to transfer into --model (e.g. a trained detector into a -seg.yaml)")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default=None)
    parser.add_argument("--fraction", type=float, default=1.0, help="train on a subset (for timing runs)")
    parser.add_argument("--resume", default=None, help="last.pt of an interrupted run (only --batch/--device/--imgsz apply)")
    parser.add_argument("--one-epoch", action="store_true",
                        help="exit after each saved epoch (resume in a loop); works around PyTorch MPS memory growth")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        if args.one_epoch:
            model.add_callback("on_fit_epoch_end", _exit_after_epoch)
        model.train(resume=True, batch=args.batch, device=args.device)
        open(os.path.join(os.path.dirname(args.resume), "DONE"), "w").close()
        return

    model = YOLO(args.model)
    if args.init:
        model.load(args.init)
    if args.one_epoch:
        model.add_callback("on_fit_epoch_end", _exit_after_epoch)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        project=os.path.join(ROOT, "runs"),
        # A short --epochs run (fine-tuning) still needs the last epochs without mosaic
        close_mosaic=min(10, max(1, args.epochs // 4)),
        name=args.name or os.path.splitext(os.path.basename(args.model))[0],
        exist_ok=True,
        patience=20,
        cos_lr=True,
        # Photometric: night, glare, wet/reflective surfaces, dusty roads
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        # Geometric: phone tilt, distance, partial views
        degrees=5.0,
        translate=0.1,
        scale=0.6,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        plots=True,
        seed=42,
    )


def _exit_after_epoch(trainer):
    # Called after last.pt is saved; on the final epoch let training finish normally
    if not trainer.stop:
        os._exit(75)


if __name__ == "__main__":
    main()
