"""
Photo authenticity checks for complaint images.

Two independent signals:

1. Metadata: AI provenance markers (IPTC "trainedAlgorithmicMedia", C2PA,
   Stable Diffusion "parameters" chunks, generator names), screenshot markers,
   and whether the file carries camera EXIF. A gallery upload with no camera
   EXIF at all is most likely downloaded or a screenshot.
2. Classifier (when backend/ml_models/authenticity.onnx exists): real /
   ai_generated / screen_photo from image content, averaged over several
   crops at near-native resolution (see training/authenticity/).

Nothing here blocks a report; the result is a flag for admins and a warning
for the citizen.
"""

import io
import json
import os
import threading

import numpy as np
from PIL import Image, ExifTags

from backend.config import settings

CLASSES = ["real", "ai_generated", "screen_photo"]
CROP = 256
SHORT_SIDE = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

AI_MARKERS = [
    (b"trainedAlgorithmicMedia", "IPTC metadata declares the image AI-generated"),
    (b"compositeWithTrainedAlgorithmicMedia", "IPTC metadata declares AI-generated content"),
    (b"c2pa", "C2PA content credentials present (typical of AI tools such as DALL-E / Firefly)"),
    (b"Midjourney", "Midjourney metadata"),
    (b"DALL-E", "DALL-E metadata"),
    (b"DALL\xc2\xb7E", "DALL-E metadata"),
    (b"Stable Diffusion", "Stable Diffusion metadata"),
    (b"ComfyUI", "ComfyUI workflow metadata"),
    (b"NovelAI", "NovelAI metadata"),
    (b"Adobe Firefly", "Adobe Firefly metadata"),
    (b"Gemini", "Google Gemini / Imagen metadata"),
]
SCREENSHOT_MARKERS = [b"Screenshot", b"screenshot"]
EDITOR_SOFTWARE = ("photoshop", "gimp", "canva", "picsart", "lightroom", "snapseed", "pixlr", "facetune")

_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_session = None
_lock = threading.Lock()


def inspect_metadata(image_bytes: bytes) -> dict:
    """Return metadata findings: {"ai_markers", "screenshot", "camera", "software", "size", "format"}."""
    head = image_bytes[:2_000_000]
    findings = {"ai_markers": [], "screenshot": False, "camera": None, "software": None,
                "captured": None, "size": None, "format": None}
    for marker, reason in AI_MARKERS:
        if marker in head and reason not in findings["ai_markers"]:
            findings["ai_markers"].append(reason)
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return findings
    findings["size"] = list(img.size)
    findings["format"] = img.format
    # PNG text chunks: Stable Diffusion WebUI writes "parameters", ComfyUI "prompt"/"workflow"
    for key in ("parameters", "prompt", "workflow", "Dream", "sd-metadata"):
        if key in img.info:
            findings["ai_markers"].append(f"PNG '{key}' chunk written by an AI image generator")
    exif = img.getexif()
    make, model = exif.get(_TAGS["Make"]), exif.get(_TAGS["Model"])
    if make or model:
        findings["camera"] = " ".join(str(v).strip("\x00 ") for v in (make, model) if v)
    software = exif.get(_TAGS["Software"])
    if software:
        findings["software"] = str(software).strip("\x00 ")
    sub = exif.get_ifd(0x8769)  # Exif sub-IFD
    captured = sub.get(_TAGS["DateTimeOriginal"]) if sub else None
    if captured:
        findings["captured"] = str(captured)
    comment = (sub.get(_TAGS["UserComment"]) if sub else None) or b""
    if any(m in head for m in SCREENSHOT_MARKERS) or b"Screenshot" in (comment if isinstance(comment, bytes) else str(comment).encode()):
        findings["screenshot"] = True
    return findings


def _get_session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                path = settings.AUTHENTICITY_MODEL_PATH
                if not os.path.exists(path):
                    return None
                import onnxruntime as ort
                _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _session


def _crops(img: Image.Image) -> np.ndarray:
    """Shorter side to 512, then center + four corner 256 px crops (matches training)."""
    w, h = img.size
    s = SHORT_SIDE / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    if min(w, h) < CROP:
        img = img.resize((max(CROP, w), max(CROP, h)), Image.BICUBIC)
        w, h = img.size
    boxes = [((w - CROP) // 2, (h - CROP) // 2), (0, 0), (w - CROP, 0), (0, h - CROP), (w - CROP, h - CROP)]
    arr = np.asarray(img, np.float32) / 255.0
    out = [(arr[y:y + CROP, x:x + CROP] - MEAN) / STD for x, y in boxes]
    return np.stack(out).transpose(0, 3, 1, 2).astype(np.float32)


def classify(image_bytes: bytes) -> dict | None:
    session = _get_session()
    if session is None:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logits = session.run(None, {session.get_inputs()[0].name: _crops(img)})[0]
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs = (e / e.sum(1, keepdims=True)).mean(0)
    return {name: round(float(p), 4) for name, p in zip(CLASSES, probs)}


def assess(image_bytes: bytes, live_capture: bool = False) -> dict:
    """Combine metadata and classifier into a verdict.

    verdict: real | ai_generated | screen_photo | downloaded
    is_authentic is False only for strong evidence (AI markers, classifier,
    screenshot). A missing camera EXIF alone is a soft "downloaded" hint:
    WhatsApp-forwarded photos and some mobile browsers strip EXIF too.
    """
    meta = inspect_metadata(image_bytes)
    probs = classify(image_bytes)
    reasons = []
    verdict = "real"
    strong = True
    threshold = settings.AUTHENTICITY_THRESHOLD

    if meta["ai_markers"]:
        verdict = "ai_generated"
        reasons += meta["ai_markers"]
    elif probs and probs["ai_generated"] >= threshold:
        verdict = "ai_generated"
        reasons.append(f"Image content looks AI-generated ({probs['ai_generated']:.0%})")
    elif probs and probs["screen_photo"] >= threshold:
        verdict = "screen_photo"
        reasons.append(f"Looks like a photo of a screen: pixel grid / moire pattern ({probs['screen_photo']:.0%})")
    elif meta["screenshot"]:
        verdict = "downloaded"
        reasons.append("Image is a screenshot")
    elif not live_capture and not meta["camera"]:
        # Gallery uploads straight from a phone camera keep Make/Model EXIF;
        # images saved from the web or messaging apps usually don't.
        verdict = "downloaded"
        strong = False
        reasons.append("No camera information in the photo (possibly downloaded or forwarded)")

    if meta["software"] and any(s in meta["software"].lower() for s in EDITOR_SOFTWARE):
        reasons.append(f"Edited with {meta['software']}")

    return {
        "verdict": verdict,
        "is_authentic": verdict == "real" or not strong,
        "reasons": reasons,
        "probabilities": probs,
        "metadata": {k: meta[k] for k in ("camera", "software", "captured", "size", "format")},
        "live_capture": live_capture,
    }


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p, json.dumps(assess(open(p, "rb").read()), indent=2))
