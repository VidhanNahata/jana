import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from backend.database import get_db
from backend.models.complaint import Complaint
from backend.models.master_ticket import MasterTicket
from backend.routers.auth import get_current_user
from backend.models.user import User

router = APIRouter()

def require_admin(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user

class ComplaintUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None

def _authenticity(c):
    try:
        a = json.loads(c.ai_analysis or "{}").get("authenticity")
    except ValueError:
        return None
    return {"verdict": a["verdict"], "flagged": a["flagged"], "reasons": a["reasons"]} if a else None


def _civic_ai(c):
    """What the reviewing official needs from the civic AI result."""
    try:
        civic = json.loads(c.ai_analysis or "{}").get("civic_ai")
    except ValueError:
        return None
    if not civic:
        return None
    cls, routing, sub = civic["classification"], civic["routing"], civic.get("submission", {})
    return {
        "status": civic["status"],
        "subcategory_id": cls["subcategory_id"],
        "subcategory_name": cls["subcategory_name_en"],
        "confidence": cls["confidence"],
        "consistency": civic["fusion"]["consistency"],
        "review_reasons": cls["review_reasons"],
        "routing_status": routing["status"],
        "authority_id": sub.get("authority_id"),
        "department_basis": sub.get("department_basis"),
        "citizen_draft": sub.get("submitted_draft"),
    }


@router.get("/complaints")
def get_complaints(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    query = db.query(Complaint)
    if status:
        query = query.filter(Complaint.status == status)
    if priority:
        query = query.filter(Complaint.priority == priority)
    if department:
        query = query.filter(Complaint.department == department)
        
    total = query.count()
    pages = (total + limit - 1) // limit
    
    complaints = query.order_by(Complaint.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    
    result = []
    for c in complaints:
        result.append({
            "id": c.id,
            "ticket_id": c.ticket_id,
            "description": c.description,
            "category": c.category,
            "department": c.department,
            "priority": c.priority,
            "status": c.status,
            "created_at": c.created_at.isoformat(),
            "location": {"lat": c.lat, "lon": c.lon},
            "master_ticket_id": c.master_ticket_id,
            "images": json.loads(c.images or "[]"),
            "is_authentic": c.is_authentic,
            "authenticity": _authenticity(c),
            "civic_ai": _civic_ai(c),
        })
        
    return {
        "complaints": result,
        "total": total,
        "page": page,
        "pages": pages
    }

@router.get("/stats")
def get_stats(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    total_complaints = db.query(Complaint).count()
    total_open = db.query(Complaint).filter(Complaint.status != "resolved").count()
    total_resolved = db.query(Complaint).filter(Complaint.status == "resolved").count()
    total_master_tickets = db.query(MasterTicket).count()
    
    # Group by department
    dept_counts = db.query(Complaint.department, func.count(Complaint.id)).group_by(Complaint.department).all()
    by_department = {d: c for d, c in dept_counts}
    
    # Group by priority
    pri_counts = db.query(Complaint.priority, func.count(Complaint.id)).group_by(Complaint.priority).all()
    by_priority = {p: c for p, c in pri_counts}
    
    # Group by status
    stat_counts = db.query(Complaint.status, func.count(Complaint.id)).group_by(Complaint.status).all()
    by_status = {s: c for s, c in stat_counts}
    
    return {
        "total_complaints": total_complaints,
        "total_open": total_open,
        "total_resolved": total_resolved,
        "total_master_tickets": total_master_tickets,
        "by_department": by_department,
        "by_priority": by_priority,
        "by_status": by_status
    }

@router.get("/hotspots")
def get_hotspots(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    # Simply returning master tickets as hotspots
    masters = db.query(MasterTicket).filter(MasterTicket.status != "resolved").all()
    return [
        {
            "lat": m.centroid_lat,
            "lon": m.centroid_lon,
            "count": m.child_count,
            "category": m.category,
            "priority": m.priority
        } for m in masters
    ]

@router.get("/master-tickets")
def get_master_tickets(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    masters = db.query(MasterTicket).order_by(MasterTicket.created_at.desc()).all()
    return [
        {
            "id": m.id,
            "canonical_description": m.canonical_description,
            "category": m.category,
            "department": m.department,
            "priority": m.priority,
            "child_count": m.child_count,
            "centroid_lat": m.centroid_lat,
            "centroid_lon": m.centroid_lon,
            "status": m.status,
            "created_at": m.created_at.isoformat()
        } for m in masters
    ]

@router.patch("/complaints/{complaint_id}")
def update_complaint(complaint_id: str, update: ComplaintUpdate, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
        
    if update.status:
        complaint.status = update.status
    if update.priority:
        complaint.priority = update.priority
        
    db.commit()
    db.refresh(complaint)
    
    return {
        "id": complaint.id,
        "ticket_id": complaint.ticket_id,
        "status": complaint.status,
        "priority": complaint.priority
    }

@router.delete("/complaints/{complaint_id}", status_code=204)
def delete_complaint(complaint_id: str, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    # Try by UUID first, then by ticket_id
    complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
    if not complaint:
        complaint = db.query(Complaint).filter(Complaint.ticket_id == complaint_id).first()
    if not complaint:
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    db.delete(complaint)
    db.commit()
    return None
