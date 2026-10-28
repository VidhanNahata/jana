"""
Civic complaint AI pipeline (orchestration).

  1 image understanding    vision.analyze_images (local VLM) + pothole detector evidence
  2 text understanding     text_analysis.analyze_text (+ local translation for Kannada)
  3 consistency            fusion.assess_consistency
  4 classification         fusion.classify (versioned taxonomy)
  5 routing                routing_engine.route (versioned authority registry)
  6 drafting               drafting.draft_complaint (English / Kannada)
  7 confidence & review    status, scope and review reasons below

Photo authenticity is NOT decided here: the existing fake-image pipeline
(services.image_authenticity.assess) runs unchanged and its output is
attached as-is. A flagged photo adds a review reason; it never changes what
the photo is judged to show, and a genuine photo never verifies the text.
"""

import time
import uuid
from collections import OrderedDict

from backend.ai import knowledge, vision
from backend.ai.drafting import draft_complaint
from backend.ai.fusion import assess_consistency, classify
from backend.ai.routing_engine import CitizenLocation, route
from backend.ai.schema import (AnalysisResult, Language, ModelMetadata, RoutingStatus, Scope, StageStatus, Status)
from backend.ai.text_analysis import analyze_text, detect_language, merge_llm_reading, normalize
from backend.config import settings

PIPELINE_VERSION = "1.0.0"
_CACHE_SIZE = 256
_cache: "OrderedDict[str, AnalysisResult]" = OrderedDict()


def _authenticity(images: list[bytes], live_flags: list[bool]) -> dict | None:
    """Run the existing fake-image detector and return its output unmodified (plus a summary)."""
    if not images:
        return None
    from backend.services.image_authenticity import assess
    results, errors = [], []
    for data, live in zip(images, live_flags):
        try:
            results.append(assess(data, live_capture=live))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if not results:
        return {"status": "error", "errors": errors}
    flagged = [r for r in results if not r["is_authentic"]]
    return {"status": "ok", "verdict": flagged[0]["verdict"] if flagged else "real", "flagged": bool(flagged),
            "reasons": sorted({reason for r in results for reason in r["reasons"]}), "images": results,
            "errors": errors}


def _detector(images: list[bytes]) -> list[dict | None]:
    """Pothole detector per image, with the road-damage detector's output (when
    that model is installed) under "road_damage"."""
    from backend.services.pothole_detector import detect_potholes
    from backend.services.road_damage_detector import detect_road_damage
    out = []
    for data in images:
        try:
            d = detect_potholes(data)
        except Exception:
            d = None
        try:
            road = detect_road_damage(data)
        except Exception as exc:
            road = {"error": f"{type(exc).__name__}: {exc}"}
        if road is not None:
            d = {**(d or {"detected": False, "count": 0, "boxes": []}), "road_damage": road}
        out.append(d)
    return out


def analyze(images: list[bytes], text: str, citizen: CitizenLocation | None = None, draft_language: str = "en",
            confirmed_subcategory: str | None = None, live_flags: list[bool] | None = None,
            image_refs: list[str] | None = None, run_vlm: bool = True, run_detectors: bool = True,
            prior: AnalysisResult | None = None) -> AnalysisResult:
    """Run the pipeline.

    run_detectors=False skips the pothole and fake-image detectors (the
    submission path runs them as background tasks instead). prior reuses the
    image, text and authenticity stages of an earlier analysis and re-runs
    stages 3-7 only, e.g. after the citizen confirms the primary issue or the
    city corporation."""
    citizen = citizen or CitizenLocation()
    live_flags = live_flags or [False] * len(images)
    timings, errors = {}, []

    def timed(name, fn, *a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            timings[name] = round((time.perf_counter() - t0) * 1000, 1)

    if settings.AI_STRICT_OFFLINE:
        vision.enforce_offline()

    if prior is not None:
        authenticity, image, text_result = prior.authenticity, prior.image_analysis, prior.text_analysis
        timings.update({k: v for k, v in prior.model_metadata.timings_ms.items()
                        if k not in ("routing",)})
        if (text or "").strip() != prior.text_analysis.original_text:
            # The citizen edited the description after the analysis: re-read the text
            text_result = timed("text_understanding", analyze_text, text)
    else:
        authenticity = timed("authenticity", _authenticity, images, live_flags) if run_detectors else None
        detections = timed("detectors", _detector, images) if images and run_detectors else []
        image = timed("image_understanding", vision.analyze_images, images, detections, use_vlm=run_vlm)

        translation = None
        lang = detect_language(normalize(text or ""))
        if run_vlm and lang in (Language.kn, Language.kn_en_mixed, Language.kn_transliterated):
            translation = timed("translation", vision.translate_to_english, text)
        text_result = timed("text_understanding", analyze_text, text, translation,
                            "local_vlm" if translation else None)
        if run_vlm and text_result.status == StageStatus.ok:
            reading = timed("text_llm", vision.read_text, text, translation)
            text_result = merge_llm_reading(text_result, reading)
    if image.error and image.status not in (StageStatus.no_input,):
        errors.append(f"image: {image.error}")

    fusion = assess_consistency(image, text_result)
    try:
        classification = classify(image, text_result, fusion, confirmed_subcategory)
    except ValueError as exc:
        errors.append(str(exc))
        classification = classify(image, text_result, fusion, None)
    routing = timed("routing", route, classification, text_result.location, citizen)

    if authenticity and authenticity.get("flagged"):
        classification.human_review_required = True
        classification.review_reasons.append("Photo authenticity flagged by the fake-image detector: "
                                             + "; ".join(authenticity["reasons"]))

    draft = None
    if not classification.out_of_scope:
        draft = draft_complaint(draft_language, classification, image, text_result, fusion, routing,
                                image_refs or (prior.complaint_draft.attachment_refs if prior and prior.complaint_draft
                                               else [f"photo_{i + 1}" for i in range(len(images))]),
                                lat=citizen.lat, lon=citizen.lon, locality=citizen.locality)

    # Stage 7: overall status and scope
    if classification.out_of_scope:
        scope, status = Scope.out_of_scope, Status.out_of_scope
    else:
        scope = Scope.uncertain if classification.subcategory_id == "OTHER.CIVIC_INFRA" and \
            classification.confidence < settings.AI_REVIEW_CONFIDENCE else Scope.in_scope
        if classification.primary_selection_required or routing.status in (
                RoutingStatus.location_required,) or "city_corporation" in routing.missing_location_fields:
            status = Status.needs_citizen_input
        elif classification.human_review_required or routing.status in (
                RoutingStatus.authority_not_configured, RoutingStatus.confirmation_required) or scope == Scope.uncertain:
            status = Status.needs_review
        else:
            status = Status.ok

    vlm_meta = vision.metadata()
    result = AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        status=status,
        scope=scope,
        image_analysis=image,
        text_analysis=text_result,
        fusion=fusion,
        classification=classification,
        routing=routing,
        complaint_draft=draft,
        authenticity=authenticity,
        model_metadata=ModelMetadata(
            pipeline_version=PIPELINE_VERSION,
            offline_mode=settings.AI_STRICT_OFFLINE,
            taxonomy_version=knowledge.versions()["taxonomy"],
            registry_version=knowledge.versions()["registry"],
            lexicon_version=knowledge.versions()["lexicon"],
            vlm=vlm_meta,
            detector=settings.POTHOLE_MODEL_PATH,
            authenticity_model=settings.AUTHENTICITY_MODEL_PATH,
            timings_ms=timings,
        ),
        errors=errors,
    )
    _cache[result.analysis_id] = result
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return result


def cached(analysis_id: str | None) -> AnalysisResult | None:
    return _cache.get(analysis_id) if analysis_id else None
