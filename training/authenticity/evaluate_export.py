"""
Evaluate the authenticity classifier the way the app runs it, choose the flag
threshold, and export to ONNX for the browser and the backend.

Inference = shorter side 512, five 256 px crops (center + corners), softmax
averaged; a photo is flagged when P(ai_generated) or P(screen_photo) >= threshold.
The threshold is the lowest value that keeps the false-flag rate on real val
photos at or below MAX_FALSE_FLAG, because wrongly flagging a genuine citizen
report is worse than missing some fakes (the server, metadata checks and
officials are further lines of defence).

Usage:
    python training/authenticity/evaluate_export.py
Writes training/runs/authenticity/eval_report.json and installs
frontend/public/models/authenticity.{onnx,json}, backend/ml_models/authenticity.onnx
"""

import glob
import io
import json
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_authenticity import CLASSES, CROP, DATA, INFER_SHORT_SIDE, MEAN, STD, OUT, build_model  # noqa: E402

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MAX_FALSE_FLAG = 0.03
MEAN, STD = np.array(MEAN, np.float32), np.array(STD, np.float32)
# Uploads arrive as phone JPEGs; evaluate at a typical quality
EVAL_JPEG_QUALITY = 90


def crops(img):
    w, h = img.size
    s = INFER_SHORT_SIDE / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    if min(w, h) < CROP:
        img = img.resize((max(CROP, w), max(CROP, h)), Image.BICUBIC)
        w, h = img.size
    arr = np.asarray(img, np.float32) / 255.0
    boxes = [((w - CROP) // 2, (h - CROP) // 2), (0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP)]
    out = [(arr[y:y + CROP, x:x + CROP] - MEAN) / STD for x, y in boxes]
    return np.stack(out).transpose(0, 3, 1, 2)



def predict_split(model, split, device):
    rows = []
    for label, name in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(DATA, split, name, "*.jpg"))):
            img = Image.open(f).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=EVAL_JPEG_QUALITY)
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            with torch.no_grad():
                logits = model(torch.from_numpy(crops(img)).to(device))
            p = torch.softmax(logits, 1).mean(0).cpu().numpy()
            rows.append((label, p))
    return rows


def flag_stats(rows, t):
    real = [p for l, p in rows if l == 0]
    fake = [(l, p) for l, p in rows if l != 0]
    flagged = lambda p: max(p[1], p[2]) >= t
    out = {"real_photos": len(real),
           "false_flag_rate": round(float(np.mean([flagged(p) for p in real])), 4) if real else None}
    for li, name in ((1, "ai_generated"), (2, "screen_photo")):
        sel = [p for l, p in fake if l == li]
        if sel:
            out[f"{name}_caught"] = round(float(np.mean([flagged(p) for p in sel])), 4)
            # Caught *and* given the right reason
            out[f"{name}_correct_reason"] = round(float(np.mean([flagged(p) and int(np.argmax(p[1:])) + 1 == li for p in sel])), 4)
            out[f"{name}_images"] = len(sel)
    return out


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(os.path.join(OUT, "best.pt"), map_location="cpu"))
    model.eval().to(device)

    val = predict_split(model, "val", device)
    real_scores = sorted(max(p[1], p[2]) for l, p in val if l == 0)
    # Lowest threshold whose false-flag rate on real val photos <= MAX_FALSE_FLAG
    t = float(real_scores[int(np.ceil(len(real_scores) * (1 - MAX_FALSE_FLAG))) - 1]) + 1e-4
    t = round(min(max(t, 0.5), 0.95), 3)

    report = {"threshold": t, "max_false_flag_target": MAX_FALSE_FLAG, "val": flag_stats(val, t), "test_sets": {}}
    for split in sorted(os.listdir(DATA)):
        if split.startswith("test"):
            report["test_sets"][split] = flag_stats(predict_split(model, split, device), t)
    json.dump(report, open(os.path.join(OUT, "eval_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))

    # Export: dynamic batch so the five crops run in one call
    onnx_path = os.path.join(OUT, "authenticity.onnx")
    model.cpu()
    torch.onnx.export(model, torch.zeros(5, 3, CROP, CROP), onnx_path, input_names=["images"],
                      output_names=["logits"], dynamic_axes={"images": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    frontend = os.path.join(PROJECT, "frontend", "public", "models")
    shutil.copy(onnx_path, os.path.join(frontend, "authenticity.onnx"))
    shutil.copy(onnx_path, os.path.join(PROJECT, "backend", "ml_models", "authenticity.onnx"))
    json.dump({"threshold": t, "crop": CROP, "shortSide": INFER_SHORT_SIDE, "classes": CLASSES,
               "testMetrics": report["test_sets"]}, open(os.path.join(frontend, "authenticity.json"), "w"), indent=2)
    print(f"exported {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB), threshold {t}")


if __name__ == "__main__":
    main()
