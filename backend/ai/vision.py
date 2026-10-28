"""
Stage 1: image understanding with a local open-weight vision-language model.

Model: Qwen3-VL-2B-Instruct (Apache-2.0), loaded from a local directory with
local_files_only=True. Chosen over Qwen2.5-VL-3B (Qwen Research licence,
non-commercial) and the 7B/8B models (too large for 8 GB machines and the
free API host). See backend/ai/README.md for the evaluation notes.

The model is asked for a strict JSON object describing the scene, image
quality and visible civic problems (taxonomy subcategory IDs only); the
output is parsed and validated, unknown IDs are dropped with a warning, and a
malformed answer is retried once before the stage reports
invalid_model_output. Nothing here downloads anything.
"""

import io
import json
import os
import re
import threading

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.ai import knowledge
from backend.ai.schema import EvidenceSource, ImageAnalysis, ImageQuality, StageStatus, VisualProblem
from backend.config import settings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "Qwen3-VL-2B-Instruct"
MODEL_LICENSE = "Apache-2.0"

_lock = threading.Lock()
_model = None
_processor = None
_device = None
_load_error = None


class ModelUnavailable(RuntimeError):
    pass


def enforce_offline():
    """Make every Hugging Face library call local-only for this process."""
    if settings.AI_STRICT_OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def model_path() -> str:
    p = settings.AI_VLM_PATH
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def model_installed() -> bool:
    p = model_path()
    return os.path.isfile(os.path.join(p, "config.json")) and any(
        f.endswith(".safetensors") for f in os.listdir(p)) if os.path.isdir(p) else False


def _pick_device(torch):
    want = settings.AI_VLM_DEVICE
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(torch, device):
    want = settings.AI_VLM_DTYPE
    if want != "auto":
        return getattr(torch, want)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32


def load():
    """Load the VLM once (thread-safe). Raises ModelUnavailable with a clear reason."""
    global _model, _processor, _device, _load_error
    if _model is not None:
        return _model, _processor
    if not settings.AI_VLM_ENABLED:
        raise ModelUnavailable("Vision-language model disabled (AI_VLM_ENABLED=false)")
    if not model_installed():
        raise ModelUnavailable(f"Vision-language model not installed at {model_path()} "
                               "(run: python scripts/install_models.py)")
    with _lock:
        if _model is not None:
            return _model, _processor
        enforce_offline()
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ModelUnavailable(f"torch/transformers not installed: {exc}") from exc
        device = _pick_device(torch)
        dtype = _pick_dtype(torch, device)
        kwargs = {"dtype": dtype, "local_files_only": True}
        quant = settings.AI_VLM_QUANTIZATION
        if quant in ("bnb-8bit", "bnb-4bit"):
            if device != "cuda":
                raise ModelUnavailable(f"{quant} quantization needs a CUDA GPU (device is {device})")
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=quant == "bnb-8bit",
                                                               load_in_4bit=quant == "bnb-4bit")
            kwargs["device_map"] = "auto"
        try:
            processor = AutoProcessor.from_pretrained(model_path(), local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(model_path(), **kwargs)
            if "device_map" not in kwargs:
                model = model.to(device)
            if quant == "dynamic-int8":
                if device != "cpu":
                    raise ModelUnavailable("dynamic-int8 quantization is CPU-only")
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.eval()
        except ModelUnavailable:
            raise
        except Exception as exc:
            _load_error = str(exc)
            raise ModelUnavailable(f"Failed to load {model_path()}: {exc}") from exc
        _model, _processor, _device = model, processor, device
        return _model, _processor


def metadata() -> dict:
    return {
        "name": MODEL_NAME,
        "license": MODEL_LICENSE,
        "path": settings.AI_VLM_PATH,
        "enabled": settings.AI_VLM_ENABLED,
        "installed": model_installed(),
        "loaded": _model is not None,
        "device": _device,
        "dtype": settings.AI_VLM_DTYPE,
        "quantization": settings.AI_VLM_QUANTIZATION,
        "load_error": _load_error,
    }


def open_image(data: bytes) -> Image.Image:
    """Decode an upload; raises ValueError for corrupted or unsupported files."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unreadable or corrupted image: {exc}") from exc
    img = ImageOps.exif_transpose(img).convert("RGB")
    side = settings.AI_VLM_MAX_IMAGE_SIDE
    if max(img.size) > side:
        img.thumbnail((side, side), Image.LANCZOS)
    return img


def _generate(messages: list, max_new_tokens: int) -> str:
    import torch
    model, processor = load()
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def _prompt() -> str:
    lines = []
    for sid in knowledge.civic_subcategory_ids():
        sub = knowledge.subcategory(sid)
        lines.append(f"- {sid}: {sub['visual_cues']}")
    return (
        "You inspect photos submitted to a civic complaint system in Karnataka, India. "
        "Describe only what is actually visible. Do not guess causes, locations, dates or people.\n"
        "Civic problem IDs (use only these IDs):\n" + "\n".join(lines) + "\n\n"
        "Answer with ONE JSON object and nothing else, exactly with these keys:\n"
        '{"scene": "<one factual sentence describing the photo>", '
        '"is_civic_scene": <true if the photo shows a street, road, public place or public infrastructure, else false>, '
        '"image_quality": "<good|blurry|dark|obstructed>", '
        '"problems": [{"id": "<problem ID from the list>", "evidence": "<what you see>", "confidence": <0.0-1.0>}]}\n'
        "Use an empty problems list if no listed problem is clearly visible. "
        "A selfie, food, document, screenshot, indoor private room, animal or product photo is not a civic scene."
    )


def _parse(text: str) -> dict:
    """Extract and validate the JSON object from the model's answer."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    data = json.loads(m.group(0))
    if not isinstance(data, dict) or "problems" not in data or not isinstance(data["problems"], list):
        raise ValueError("JSON missing 'problems' list")
    return data


def road_damage_problems(road: dict | None) -> list[VisualProblem]:
    """One visual problem per taxonomy subcategory the road-damage detector found."""
    by_sub = {}
    for d in (road or {}).get("detections", []):
        by_sub.setdefault(d["subcategory_id"], []).append(d)
    problems = []
    for sid, ds in sorted(by_sub.items()):
        if not knowledge.subcategory(sid):
            continue
        counts = {}
        for d in ds:
            counts[d["label_en"]] = counts.get(d["label_en"], 0) + 1
        problems.append(VisualProblem(
            subcategory_id=sid, source=EvidenceSource.road_damage_detector,
            confidence=max(d["confidence"] for d in ds),
            evidence="Road-damage detector found " + ", ".join(f"{n} {l.lower()}" for l, n in counts.items())))
    return problems


def analyze_images(images: list[bytes], detector_results: list[dict | None] | None = None,
                   use_vlm: bool = True) -> ImageAnalysis:
    """Analyze the first image with the VLM (plus pothole-detector evidence for all images).

    use_vlm=False skips the language model (e.g. a cheap re-check at submission)
    but still reports the specialist detector's evidence."""
    if not images:
        return ImageAnalysis(status=StageStatus.no_input)
    detector_results = detector_results or [None] * len(images)
    detector = next((d for d in detector_results if d), None)
    warnings = []
    det_problems = []
    if detector and detector.get("detected"):
        det_problems.append(VisualProblem(
            subcategory_id="ROAD.POTHOLE", source=EvidenceSource.pothole_detector,
            confidence=round(float(detector["max_confidence"]), 3),
            evidence=f"Pothole detector found {detector['count']} pothole(s), max confidence "
                     f"{detector['max_confidence']:.2f}"))
    det_problems += road_damage_problems((detector or {}).get("road_damage"))

    try:
        img = open_image(images[0])
    except ValueError as exc:
        return ImageAnalysis(status=StageStatus.invalid_input, images_analyzed=0, error=str(exc), detector=detector)

    try:
        if not use_vlm:
            raise ModelUnavailable("Image understanding skipped for this request")
        load()
    except ModelUnavailable as exc:
        return ImageAnalysis(status=StageStatus.model_unavailable, images_analyzed=0, detector=detector,
                             problems=det_problems, error=str(exc),
                             warnings=["Image understanding unavailable; only the road detectors ran."])

    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": _prompt()}]}]
    data, last_err, raw = None, None, ""
    for attempt in range(2):
        try:
            raw = _generate(messages, settings.AI_VLM_MAX_NEW_TOKENS)
            data = _parse(raw)
            break
        except Exception as exc:  # malformed output: retry once with a reminder
            last_err = f"{type(exc).__name__}: {exc}"
            messages = messages[:1] + [{"role": "assistant", "content": [{"type": "text", "text": raw or ""}]},
                                       {"role": "user", "content": [{"type": "text", "text":
                                        "Reply again with only the JSON object in the required format."}]}]
    if data is None:
        return ImageAnalysis(status=StageStatus.invalid_model_output, images_analyzed=1, detector=detector,
                             problems=det_problems, error=last_err, warnings=[f"raw output: {raw[:300]}"])

    problems = []
    for p in data.get("problems", []):
        sid = str(p.get("id", "")).strip()
        if not knowledge.subcategory(sid) or sid.startswith("OUT_OF_SCOPE."):
            warnings.append(f"model returned unknown problem id {sid!r}; ignored")
            continue
        try:
            conf = float(p.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        problems.append(VisualProblem(subcategory_id=sid, evidence=str(p.get("evidence", ""))[:300],
                                      confidence=min(max(conf, 0.0), 1.0)))
    quality = str(data.get("image_quality", "unknown")).lower()
    civic = data.get("is_civic_scene")
    return ImageAnalysis(
        status=StageStatus.ok,
        images_analyzed=1,
        scene_description=str(data.get("scene", ""))[:400] or None,
        is_civic_scene=civic if isinstance(civic, bool) else None,
        image_quality=ImageQuality(quality) if quality in ImageQuality._value2member_map_ else ImageQuality.unknown,
        problems=problems + det_problems,
        detector=detector,
        warnings=warnings,
    )


def translate_to_english(text: str) -> str | None:
    """Translate Kannada / code-mixed text with the local VLM's language model.

    Returns None when the model is unavailable; text analysis then relies on
    the multilingual lexicon only."""
    if not settings.AI_VLM_TRANSLATE_KANNADA:
        return None
    try:
        load()
    except ModelUnavailable:
        return None
    messages = [{"role": "user", "content": [{"type": "text", "text":
                 "Translate this citizen complaint from Kannada (it may mix English or be written in Latin "
                 "letters) into plain English. Keep place names. Output only the translation.\n\n" + text}]}]
    try:
        return _generate(messages, 200) or None
    except Exception:
        return None


def _text_prompt(text: str, translation: str | None) -> str:
    lines = [f"- {sid}: {knowledge.subcategory(sid)['name_en']}" for sid in knowledge.civic_subcategory_ids()
             if sid != "OTHER.CIVIC_INFRA"]
    shown = text + (f"\n(English translation: {translation})" if translation else "")
    return (
        "A citizen in Karnataka, India wrote this complaint (it may be English, Kannada, or Kannada typed in "
        "English letters):\n\n" + shown + "\n\n"
        "Which public civic infrastructure problems does the citizen report? Problem IDs (use only these):\n"
        + "\n".join(lines) + "\n\n"
        "If the complaint is not about public infrastructure, set out_of_scope to PERSONAL_DISPUTE (a dispute "
        "between private people: family, neighbours, landlord, employer) or NON_CIVIC_GRIEVANCE (a government "
        "service or private company matter: pension, certificates, police, hospitals, refunds, salary). "
        "Otherwise set it to null.\n"
        "Answer with ONE JSON object and nothing else:\n"
        '{"problems": [{"id": "<problem ID>", "confidence": <0.0-1.0>}], "out_of_scope": null}\n'
        "List only problems the citizen actually states; use an empty list if none."
    )


def read_text(text: str, translation: str | None = None) -> dict | None:
    """Zero-shot reading of the complaint text by the local LLM.

    Returns {"problems": [{"id", "confidence"}], "out_of_scope": None|kind,
    "warnings": [...]} with unknown IDs removed, or None when the model is
    unavailable or its answer could not be parsed."""
    if not settings.AI_LLM_TEXT_CLASSIFY or not (text or "").strip():
        return None
    try:
        load()
    except ModelUnavailable:
        return None
    messages = [{"role": "user", "content": [{"type": "text", "text": _text_prompt(text, translation)}]}]
    for _ in range(2):
        try:
            data = _parse(_generate(messages, 160))
            break
        except Exception:
            data = None
    if data is None:
        return None
    out = {"problems": [], "out_of_scope": None, "warnings": []}
    for p in data.get("problems", []):
        sid = str(p.get("id", "")).strip() if isinstance(p, dict) else ""
        if not knowledge.subcategory(sid) or sid.startswith("OUT_OF_SCOPE.") or sid == "OTHER.CIVIC_INFRA":
            out["warnings"].append(f"unknown problem id {sid!r} ignored")
            continue
        try:
            conf = min(max(float(p.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5
        out["problems"].append({"id": sid, "confidence": conf})
    oos = data.get("out_of_scope")
    if oos in ("PERSONAL_DISPUTE", "NON_CIVIC_GRIEVANCE"):
        out["out_of_scope"] = oos
    return out
