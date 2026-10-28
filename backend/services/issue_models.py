"""
All-category issue models (photo -> civic problem).

Two optional ONNX models, installed by training/issues/export_classifier.py and
export_detector.py. Either one, both or neither may be present; a missing model is
simply skipped, so the pipeline runs unchanged until the files are installed.

  issue_classifier.onnx  EfficientNet-B0, one label per photo (garbage, drain_sewage,
                         waterlogging, streetlight, fallen_tree, footpath_damage,
                         damaged_sign, pothole) or 'background'.
  issue_clip_head.npz    pretrained CLIP image encoder (models/clip-vit-base-patch32) + a small trained
                         head: same labels as the classifier, needs far fewer photos to train.
  issue_detector.onnx    YOLO11, boxes for garbage / open_manhole / pothole.

Each model ships with a JSON sidecar holding its class order, the taxonomy mapping and
the per-class confidence thresholds chosen on validation data. ROAD.POTHOLE is owned by
the dedicated pothole detector (services/pothole_detector.py) and is dropped here.

Output of evidence(): a list of
    {"subcategory_id", "confidence", "source": "issue_classifier" | "issue_detector", "evidence"}
"""

import io
import json
import os
import threading

import numpy as np
from PIL import Image, ImageOps

from backend.config import settings

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
POTHOLE = "ROAD.POTHOLE"
NMS_IOU = 0.5

_lock = threading.Lock()
_loaded: dict[str, tuple | None] = {}
_embedder: dict = {}          # CLIP encoder (loaded lazily on first photo); {"error": ...} after a failure


def _load(kind: str, path: str):
    """(session, sidecar) or None when the model isn't installed. Cached per process."""
    if kind in _loaded:
        return _loaded[kind]
    with _lock:
        if kind not in _loaded:
            meta_path = os.path.splitext(path)[0] + ".json"
            if not (settings.ISSUE_MODELS_ENABLED and os.path.exists(path) and os.path.exists(meta_path)):
                _loaded[kind] = None
            else:
                import onnxruntime as ort
                with open(meta_path) as f:
                    meta = json.load(f)
                _loaded[kind] = (ort.InferenceSession(path, providers=["CPUExecutionProvider"]), meta)
    return _loaded[kind]


def reset_cache() -> None:
    """For tests and after installing new model files."""
    with _lock:
        _loaded.clear()
        _embedder.clear()


def status() -> dict:
    out = {"clip": clip_status()}
    for kind, path in (("classifier", settings.ISSUE_CLASSIFIER_PATH), ("detector", settings.ISSUE_DETECTOR_PATH)):
        try:
            m = _load(kind, path)
        except Exception as exc:
            out[kind] = {"installed": True, "error": f"{type(exc).__name__}: {exc}"}
            continue
        out[kind] = {"installed": m is not None, **({"classes": m[1]["classes"]} if m else {})}
    return out


# ── classifier ───────────────────────────────────────────────────────────────────────

def preprocess_classifier(image: Image.Image, size: int) -> np.ndarray:
    """Squash-resize to size x size + ImageNet normalisation (identical to training)."""
    arr = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def classifier_evidence(probs: np.ndarray, meta: dict) -> list[dict]:
    """Classes whose probability passes their own threshold; 'background' never produces evidence."""
    out = []
    for name, p in zip(meta["classes"], probs):
        sid = meta["subcategory"].get(name)
        if not sid or name == meta.get("background"):
            continue
        if p >= meta["thresholds"].get(name, 0.5):
            out.append({"subcategory_id": sid, "confidence": float(p), "source": "issue_classifier",
                        "evidence": f"Photo classifier: {name.replace('_', ' ')} ({p:.0%})"})
    return out


def classify(image: Image.Image) -> list[dict] | None:
    m = _load("classifier", settings.ISSUE_CLASSIFIER_PATH)
    if m is None:
        return None
    session, meta = m
    x = preprocess_classifier(image, meta["input_size"])
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    return classifier_evidence(softmax(logits), meta)


# ── CLIP encoder + trained head ─────────────────────────────────────────────────────

def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), path)


def _clip_model_dir() -> str:
    return _abs(settings.ISSUE_CLIP_MODEL_PATH)


def _load_clip_head():
    """({"W", "b"}, sidecar) when the head AND the CLIP weights are installed, else None."""
    if "clip" in _loaded:
        return _loaded["clip"]
    with _lock:
        if "clip" not in _loaded:
            head_path = _abs(settings.ISSUE_CLIP_HEAD_PATH)
            meta_path = os.path.splitext(head_path)[0] + ".json"
            if not (settings.ISSUE_MODELS_ENABLED and os.path.exists(head_path) and os.path.exists(meta_path)
                    and os.path.isfile(os.path.join(_clip_model_dir(), "config.json"))):
                _loaded["clip"] = None
            else:
                z = np.load(head_path, allow_pickle=False)
                with open(meta_path) as f:
                    meta = json.load(f)
                _loaded["clip"] = ({"W": z["W"].astype(np.float32), "b": z["b"].astype(np.float32)}, meta)
    return _loaded["clip"]


def clip_status() -> dict:
    try:
        m = _load_clip_head()
    except Exception as exc:
        return {"installed": True, "error": f"{type(exc).__name__}: {exc}"}
    out = {"installed": m is not None, "model_dir_present": os.path.isdir(_clip_model_dir())}
    if m:
        out["classes"] = m[1]["classes"]
    if "error" in _embedder:
        out["encoder_error"] = _embedder["error"]
    return out


def _embed_image(image: Image.Image) -> np.ndarray:
    """L2-normalised CLIP image embedding (same computation as training/issues/train_clip_head.py)."""
    if "error" in _embedder:
        raise RuntimeError(_embedder["error"])
    if "model" not in _embedder:
        with _lock:
            if "model" not in _embedder:
                try:
                    from backend.ai import vision
                    vision.enforce_offline()
                    import torch
                    from transformers import CLIPModel, CLIPProcessor
                    device = vision._pick_device(torch)
                    d = _clip_model_dir()
                    model = CLIPModel.from_pretrained(d, local_files_only=True).to(device).eval()
                    _embedder.update(model=model, processor=CLIPProcessor.from_pretrained(d, local_files_only=True),
                                     device=device, torch=torch)
                except Exception as exc:
                    _embedder["error"] = f"CLIP encoder failed to load: {type(exc).__name__}: {exc}"
                    raise RuntimeError(_embedder["error"]) from exc
    torch, model = _embedder["torch"], _embedder["model"]
    with torch.no_grad():
        px = _embedder["processor"](images=[image], return_tensors="pt")["pixel_values"].to(_embedder["device"])
        feats = model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
    v = feats.float().cpu().numpy()[0]
    return v / max(float(np.linalg.norm(v)), 1e-9)


def classify_clip(image: Image.Image) -> list[dict] | None:
    m = _load_clip_head()
    if m is None:
        return None
    head, meta = m
    probs = softmax(_embed_image(image) @ head["W"].T + head["b"])
    return classifier_evidence(probs, meta)


# ── detector ─────────────────────────────────────────────────────────────────────────

def _iou(box, others):
    xx1, yy1 = np.maximum(box[0], others[:, 0]), np.maximum(box[1], others[:, 1])
    xx2, yy2 = np.minimum(box[2], others[:, 2]), np.minimum(box[3], others[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(others) - inter + 1e-9)


def decode_detections(preds: np.ndarray, meta: dict, scale: float, pad_x: int, pad_y: int,
                      width: int, height: int) -> list[dict]:
    """YOLO output (4 + n_classes, anchors) -> boxes per class, with per-class thresholds and
    class-wise NMS. Coordinates are normalised 0-1 of the original image."""
    classes = meta["classes"]
    cls_scores = preds[4:4 + len(classes)]                 # (nc, anchors)
    cls_idx, scores = cls_scores.argmax(0), cls_scores.max(0)
    thr = np.array([meta["thresholds"].get(c, 0.5) for c in classes])
    keep = scores >= thr[cls_idx]
    if not keep.any():
        return []
    cx, cy, w, h = preds[:4, keep]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    cls_idx, scores = cls_idx[keep], scores[keep]
    out = []
    for c in np.unique(cls_idx):
        sel = np.where(cls_idx == c)[0]
        order = sel[scores[sel].argsort()[::-1]]
        while order.size:
            i = order[0]
            x1, y1, x2, y2 = xyxy[i]
            out.append({"class": classes[c], "confidence": float(scores[i]),
                        "x1": float(min(max((x1 - pad_x) / scale, 0), width)) / width,
                        "x2": float(min(max((x2 - pad_x) / scale, 0), width)) / width,
                        "y1": float(min(max((y1 - pad_y) / scale, 0), height)) / height,
                        "y2": float(min(max((y2 - pad_y) / scale, 0), height)) / height})
            rest = order[1:]
            order = rest[_iou(xyxy[i], xyxy[rest]) <= NMS_IOU] if rest.size else rest
    return sorted(out, key=lambda b: -b["confidence"])


def detector_evidence(boxes: list[dict], meta: dict) -> list[dict]:
    best: dict[str, dict] = {}
    count: dict[str, int] = {}
    for b in boxes:
        count[b["class"]] = count.get(b["class"], 0) + 1
        best.setdefault(b["class"], b)          # boxes are sorted: first = most confident
    out = []
    for name, b in best.items():
        sid = meta["subcategory"].get(name)
        if sid:
            out.append({"subcategory_id": sid, "confidence": b["confidence"], "source": "issue_detector",
                        "evidence": f"Detector found {count[name]} x {name.replace('_', ' ')}, "
                                    f"max confidence {b['confidence']:.2f}"})
    return out


def detect(image: Image.Image) -> list[dict] | None:
    m = _load("detector", settings.ISSUE_DETECTOR_PATH)
    if m is None:
        return None
    from backend.services.pothole_detector import _letterbox   # same Ultralytics-style letterbox
    session, meta = m
    width, height = image.size
    tensor, scale, pad_x, pad_y = _letterbox(image)
    preds = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    return detector_evidence(decode_detections(preds, meta, scale, pad_x, pad_y, width, height), meta)


# ── public entry point ───────────────────────────────────────────────────────────────

def evidence(images: list[bytes]) -> list[dict]:
    """Run whichever issue models are installed on every photo; one entry per subcategory
    (the strongest), capped so a model alone can't reach certainty. Never raises."""
    best: dict[tuple[str, str], dict] = {}
    for data in images:
        try:
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        except Exception:
            continue
        for run in (classify, classify_clip, detect):
            try:
                found = run(img) or []
            except Exception as exc:
                print(f"[issue_models] {run.__name__} failed: {type(exc).__name__}: {exc}")
                continue
            for ev in found:
                if ev["subcategory_id"] == POTHOLE:
                    continue
                key = (ev["subcategory_id"], ev["source"])
                if key not in best or ev["confidence"] > best[key]["confidence"]:
                    best[key] = dict(ev)
    cap = settings.ISSUE_MODEL_MAX_CONFIDENCE
    for ev in best.values():
        ev["confidence"] = round(min(ev["confidence"], cap), 3)
    return sorted(best.values(), key=lambda e: -e["confidence"])
