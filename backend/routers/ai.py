"""
Civic AI endpoints: analyse a photo + description before submission, and
expose the taxonomy / authority registry the review screen needs.

The analysis never submits anything: the citizen reviews and edits the draft,
then submits through POST /api/complaints with the analysis_id.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.ai import knowledge, pipeline, vision
from backend.ai.routing_engine import CitizenLocation
from backend.config import settings
from backend.models.user import User
from backend.routers.auth import get_current_user
from backend.routers.complaints import parse_dashcam
from backend.services.priority import assess_priority

router = APIRouter()

MAX_IMAGES = 5
MAX_IMAGE_BYTES = 15 * 1024 * 1024


@router.post("/analyze")
def analyze_complaint_draft(
    description: str = Form(""),
    images: List[UploadFile] = File(default=[]),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    locality: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    district: Optional[str] = Form(None),
    pin_code: Optional[str] = Form(None),
    rural: Optional[bool] = Form(None),
    city_corporation_id: Optional[str] = Form(None),
    asset_owner_id: Optional[str] = Form(None),
    confirmed_subcategory_id: Optional[str] = Form(None),
    draft_language: str = Form("en"),
    live_detection: bool = Form(False),
    analysis_id: Optional[str] = Form(None),
    # Dashcam mode: JSON {speed_kmh, nearby_detections, track_frames, detector_class}
    dashcam_evidence: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    # Sync endpoint: FastAPI runs it in the threadpool, so slow local inference
    # doesn't block other requests.
    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_IMAGES} images allowed")
    if draft_language not in ("en", "kn"):
        raise HTTPException(status_code=400, detail="draft_language must be 'en' or 'kn'")
    if confirmed_subcategory_id and not knowledge.subcategory(confirmed_subcategory_id):
        raise HTTPException(status_code=400, detail=f"Unknown subcategory {confirmed_subcategory_id}")
    for aid in (city_corporation_id, asset_owner_id):
        if aid and not knowledge.authority(aid):
            raise HTTPException(status_code=400, detail=f"Unknown authority {aid}")
    # Re-analysis after a citizen choice (issue, corporation, language): reuse the
    # earlier image/text stages instead of running the models again.
    prior = pipeline.cached(analysis_id)
    if analysis_id and prior is None:
        raise HTTPException(status_code=410, detail="Analysis expired; please analyse the photo again")
    image_bytes = []
    for image in images:
        data = image.file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"{image.filename}: image larger than 15 MB")
        image_bytes.append(data)
    citizen = CitizenLocation(locality=locality, city=city, district=district, pin_code=pin_code, rural=rural,
                              city_corporation_id=city_corporation_id, asset_owner_id=asset_owner_id,
                              lat=latitude, lon=longitude)
    result = pipeline.analyze(image_bytes, description, citizen, draft_language=draft_language,
                     confirmed_subcategory=confirmed_subcategory_id,
                     live_flags=[live_detection and i == 0 for i in range(len(image_bytes))],
                     image_refs=[img.filename or f"photo_{i + 1}" for i, img in enumerate(images)] or None,
                     prior=prior)
    out = result.model_dump(mode="json")
    # Priority preview (the same rule the submission uses)
    sev = result.classification.severity or (knowledge.subcategory(result.classification.subcategory_id) or {}).get("severity", 2)
    out["priority_preview"] = assess_priority(sev, result.classification.subcategory_id,
                                              result.image_analysis.detector, parse_dashcam(dashcam_evidence))
    return out


@router.get("/taxonomy")
def get_taxonomy():
    return knowledge.taxonomy()


@router.get("/authorities")
def get_authorities():
    """Registry entries (id, names, type, verification) for the confirmation UI."""
    return {
        "version": knowledge.registry()["version"],
        "authorities": [
            {"id": a["id"], "official_name": a["official_name"], "name_kn": a.get("name_kn"),
             "authority_type": a.get("authority_type"), "official_source_url": a.get("official_source_url"),
             "verification": a.get("verification"), "last_verified": a.get("last_verified")}
            for a in knowledge.all_authorities().values()
        ],
    }


@router.get("/status")
def get_status():
    """Which local models are installed/loaded; no inference, no network."""
    return {
        "offline_mode": settings.AI_STRICT_OFFLINE,
        "vlm": vision.metadata(),
        "knowledge_versions": knowledge.versions(),
    }
