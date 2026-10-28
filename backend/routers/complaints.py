import os
import json
import uuid
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.complaint import Complaint
from backend.routers.auth import get_current_user
from backend.models.user import User
from backend.services.complaint_service import process_new_complaint, verify_live_detection, verify_authenticity
from backend.services.ai_service import OutOfScopeComplaint
from backend.ai import knowledge
from backend.ai.routing_engine import CitizenLocation

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_complaint(
    background_tasks: BackgroundTasks,
    description: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    location_source: str = Form(...),
    images: List[UploadFile] = File(...),
    captured_at: Optional[str] = Form(None),
    live_detection: bool = Form(False),
    # From the AI review step (POST /api/ai/analyze); all optional
    analysis_id: Optional[str] = Form(None),
    # Dashcam mode: JSON {speed_kmh, nearby_detections, track_frames, detector_class}
    dashcam_evidence: Optional[str] = Form(None),
    confirmed_subcategory_id: Optional[str] = Form(None),
    confirmed_authority_id: Optional[str] = Form(None),
    complaint_text: Optional[str] = Form(None),
    locality: Optional[str] = Form(None),
    city: Optional[str] = Form(None),
    district: Optional[str] = Form(None),
    pin_code: Optional[str] = Form(None),
    rural: Optional[bool] = Form(None),
    city_corporation_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if len(images) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed")
    if confirmed_subcategory_id and (not knowledge.subcategory(confirmed_subcategory_id)
                                     or confirmed_subcategory_id.startswith("OUT_OF_SCOPE.")):
        raise HTTPException(status_code=400, detail=f"Unknown subcategory {confirmed_subcategory_id}")
    for aid in (confirmed_authority_id, city_corporation_id):
        if aid and not knowledge.authority(aid):
            raise HTTPException(status_code=400, detail=f"Unknown authority {aid}")

    saved_images = []
    image_bytes = []
    first_image_bytes = None
    for image in images:
        ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
        filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            content = await image.read()
            f.write(content)
        if first_image_bytes is None:
            first_image_bytes = content
        image_bytes.append(content)
        saved_images.append(f"/uploads/{filename}")

    citizen = CitizenLocation(locality=locality, city=city, district=district, pin_code=pin_code, rural=rural,
                              city_corporation_id=city_corporation_id, lat=latitude, lon=longitude)
    try:
        complaint = process_new_complaint(
            db=db,
            user_id=current_user.id,
            description=description,
            images=saved_images,
            lat=latitude,
            lon=longitude,
            location_source=location_source,
            captured_at=captured_at,
            image_bytes=image_bytes,
            citizen=citizen,
            analysis_id=analysis_id,
            confirmed_subcategory=confirmed_subcategory_id,
            confirmed_authority_id=confirmed_authority_id,
            submitted_draft=complaint_text,
            dashcam=parse_dashcam(dashcam_evidence),
        )
    except OutOfScopeComplaint as exc:
        _remove_uploads(saved_images)
        raise HTTPException(status_code=422, detail={"code": "out_of_scope", "message": str(exc)})
    except ValueError as exc:
        _remove_uploads(saved_images)
        raise HTTPException(status_code=400, detail=str(exc))

    civic = json.loads(complaint.ai_analysis).get("civic_ai", {})
    # Re-verify the on-device detection server-side rather than trusting the
    # client. Runs after the response: inference is slow on small CPU hosts.
    if live_detection and first_image_bytes:
        background_tasks.add_task(verify_live_detection, complaint.id, first_image_bytes)
    # Real vs AI-generated / photo-of-a-screen / downloaded. Flags for admin
    # review; never blocks. The first image of a live scan came from the camera.
    # Skipped when the reviewed analysis already ran the same detector.
    if not (civic.get("authenticity") or {}).get("status") == "ok":
        background_tasks.add_task(
            verify_authenticity, complaint.id,
            [(content, live_detection and i == 0) for i, content in enumerate(image_bytes)],
        )

    return {
        "ticket_id": complaint.ticket_id,
        "category": complaint.category,
        "department": complaint.department,
        "priority": complaint.priority,
        "status": complaint.status,
        "master_ticket_id": complaint.master_ticket_id,
        "subcategory_id": civic.get("classification", {}).get("subcategory_id"),
        "authority_id": civic.get("submission", {}).get("authority_id"),
        "routing_status": civic.get("routing", {}).get("status"),
        "department_basis": civic.get("submission", {}).get("department_basis"),
        "message": "Complaint registered successfully"
    }


def parse_dashcam(raw: Optional[str]) -> Optional[dict]:
    """Validate the device-reported dashcam evidence (used only for urgency)."""
    if not raw:
        return None
    try:
        d = json.loads(raw)
        out = {
            "speed_kmh": None if d.get("speed_kmh") is None else min(max(float(d["speed_kmh"]), 0.0), 200.0),
            "nearby_detections": None if d.get("nearby_detections") is None else min(max(int(d["nearby_detections"]), 0), 100),
            "track_frames": None if d.get("track_frames") is None else min(max(int(d["track_frames"]), 0), 10000),
            "detector_class": str(d["detector_class"])[:40] if d.get("detector_class") else None,
        }
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid dashcam_evidence")
    return out


def _remove_uploads(paths: list[str]):
    for p in paths:
        try:
            os.remove(os.path.join(UPLOAD_DIR, os.path.basename(p)))
        except OSError:
            pass

@router.get("/track/{ticket_id}")
def track_complaint(ticket_id: str, db: Session = Depends(get_db)):
    complaint = db.query(Complaint).filter(Complaint.ticket_id == ticket_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    return {
        "ticket_id": complaint.ticket_id,
        "description": complaint.description,
        "category": complaint.category,
        "department": complaint.department,
        "priority": complaint.priority,
        "status": complaint.status,
        "created_at": complaint.created_at.isoformat(),
        "updated_at": complaint.updated_at.isoformat(),
        "images": json.loads(complaint.images),
        "location": {"lat": complaint.lat, "lon": complaint.lon},
        "master_ticket_id": complaint.master_ticket_id
    }

@router.get("/my")
def get_my_complaints(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    complaints = db.query(Complaint).filter(Complaint.user_id == current_user.id).all()
    return [
        {
            "ticket_id": c.ticket_id,
            "description": c.description,
            "status": c.status,
            "priority": c.priority,
            "category": c.category,
            "created_at": c.created_at.isoformat()
        } for c in complaints
    ]
