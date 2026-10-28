"""
Pothole Detector — YOLOv8 inference for live camera frames.

Model: keremberke/yolov8s-pothole-segmentation (single class "pothole",
box mAP@0.5 = 0.93 on its validation split), exported to ONNX so it runs
on onnxruntime without PyTorch (~200 MB RAM instead of ~1.5 GB).

To use custom-trained weights, export them with
    YOLO("best.pt").export(format="onnx", imgsz=640, opset=12, simplify=True)
and point settings.POTHOLE_MODEL_PATH at the .onnx file.
"""

import io
import threading

import numpy as np
from PIL import Image

from backend.config import settings

INPUT_SIZE = 640
NMS_IOU_THRESHOLD = 0.7  # Ultralytics default

_session = None
_session_lock = threading.Lock()


def get_session():
    """Lazily create the ONNX Runtime session once per process."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import onnxruntime as ort
                _session = ort.InferenceSession(
                    settings.POTHOLE_MODEL_PATH, providers=["CPUExecutionProvider"]
                )
    return _session


def warm_up_model() -> None:
    """Load the model at startup; failures are logged, not fatal."""
    try:
        get_session()
    except Exception as exc:
        print(f"[pothole_detector] Model warm-up failed: {exc}")


def _letterbox(image: Image.Image) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to a square, matching Ultralytics'
    LetterBox (cv2 INTER_LINEAR — PIL's antialiased resize shifts confidences)."""
    import cv2

    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_w, new_h = round(width * scale), round(height * scale)
    pad_x = round((INPUT_SIZE - new_w) / 2 - 0.1)
    pad_y = round((INPUT_SIZE - new_h) / 2 - 0.1)

    resized = cv2.resize(np.asarray(image), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, scale, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def detect_potholes(image_bytes: bytes) -> dict:
    """
    Run pothole detection on one image.

    Returns:
        {
            "detected": bool,          # any box >= confidence threshold
            "count": int,              # number of boxes >= threshold
            "max_confidence": float,   # highest box confidence (0 if none)
            "boxes": [                 # normalised 0-1 coords, for overlay drawing
                {"x1", "y1", "x2", "y2", "confidence"}
            ],
            "threshold": float
        }
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    threshold = settings.POTHOLE_CONF_THRESHOLD

    tensor, scale, pad_x, pad_y = _letterbox(image)
    session = get_session()
    # output0: (1, 4 box + 1 class + 32 mask coeffs, 8400 anchors); masks are unused
    preds = session.run(["output0"], {session.get_inputs()[0].name: tensor})[0][0]

    scores = preds[4]
    candidates = scores >= threshold
    cx, cy, w, h = preds[:4, candidates]
    scores = scores[candidates]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    boxes = []
    for i in _nms(xyxy, scores, NMS_IOU_THRESHOLD):
        x1, y1, x2, y2 = xyxy[i]
        # Undo letterbox padding/scale, then normalise to 0-1 of the original image
        x1 = min(max((x1 - pad_x) / scale, 0), width)
        x2 = min(max((x2 - pad_x) / scale, 0), width)
        y1 = min(max((y1 - pad_y) / scale, 0), height)
        y2 = min(max((y2 - pad_y) / scale, 0), height)
        boxes.append({
            "x1": round(float(x1) / width, 4),
            "y1": round(float(y1) / height, 4),
            "x2": round(float(x2) / width, 4),
            "y2": round(float(y2) / height, 4),
            "confidence": round(float(scores[i]), 3),
        })
    boxes.sort(key=lambda b: b["confidence"], reverse=True)

    return {
        "detected": len(boxes) > 0,
        "count": len(boxes),
        "max_confidence": boxes[0]["confidence"] if boxes else 0.0,
        "boxes": boxes,
        "threshold": threshold,
    }
