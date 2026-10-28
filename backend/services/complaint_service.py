import json
import uuid
from backend.models.complaint import Complaint
from backend.models.master_ticket import MasterTicket
from backend.services.ai_service import analyze_complaint
from backend.services.priority import assess_priority, boost_priority
from backend.services.deduplication import find_master_ticket
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.services.pothole_detector import detect_potholes
from backend.services.image_authenticity import assess
from backend.ai import knowledge, pipeline
from backend.ai.routing_engine import CitizenLocation
from backend.ai.schema import AnalysisResult, EvidenceSource, VisualProblem

def process_new_complaint(db: Session, user_id: str, description: str, images: list[str], lat: float, lon: float,
                          location_source: str, captured_at: str | None = None, image_bytes: list[bytes] | None = None,
                          citizen: CitizenLocation | None = None, analysis_id: str | None = None,
                          confirmed_subcategory: str | None = None, confirmed_authority_id: str | None = None,
                          submitted_draft: str | None = None, dashcam: dict | None = None) -> Complaint:
    # 1-2. Civic AI: classification (taxonomy) and routing (authority registry).
    # Raises OutOfScopeComplaint / ValueError before anything is stored.
    citizen = citizen or CitizenLocation(lat=lat, lon=lon)
    ai_result = analyze_complaint(description, image_bytes, citizen, analysis_id=analysis_id,
                                  confirmed_subcategory=confirmed_subcategory,
                                  confirmed_authority_id=confirmed_authority_id, submitted_draft=submitted_draft)

    if captured_at:
        ai_result["captured_at"] = captured_at

    category = ai_result["category"]
    severity = ai_result["severity"]
    is_auth = ai_result["is_authentic"]
    department = ai_result["department"]
    if ai_result["civic_ai"].get("authenticity"):
        ai_result["authenticity"] = ai_result["civic_ai"]["authenticity"]
    
    # 3. Priority: ITIL matrix of taxonomy severity and evidence-based urgency
    civic = ai_result["civic_ai"]
    basis = assess_priority(severity, civic.get("classification", {}).get("subcategory_id"),
                            civic.get("image_analysis", {}).get("detector"), dashcam)
    if dashcam:
        ai_result["dashcam"] = dashcam
    ai_result["priority_basis"] = basis
    urgency, priority = basis["urgency"], basis["priority"]
    
    # 4. Deduplication
    master_ticket = find_master_ticket(db, description, lat, lon, category)
    
    if master_ticket:
        # Attach as child
        master_ticket.child_count += 1
        
        # Recalculate centroid (simple average)
        n = master_ticket.child_count
        master_ticket.centroid_lat = ((master_ticket.centroid_lat * (n - 1)) + lat) / n
        master_ticket.centroid_lon = ((master_ticket.centroid_lon * (n - 1)) + lon) / n
        
        # Boost priority
        master_ticket.priority = boost_priority(master_ticket.priority, master_ticket.child_count)
        master_id = master_ticket.id
    else:
        # Create new master ticket
        new_master = MasterTicket(
            canonical_description=description,
            category=category,
            department=department,
            priority=priority,
            centroid_lat=lat,
            centroid_lon=lon
        )
        db.add(new_master)
        db.flush() # get ID
        master_id = new_master.id

    # Generate ticket ID JS-{YEAR}-{SEQUENTIAL_5_DIGIT} (safe against deleted tickets)
    year = 2026
    all_tickets = db.query(Complaint.ticket_id).filter(Complaint.ticket_id.like(f"JS-{year}-%")).all()
    max_num = 0
    for (t_id,) in all_tickets:
        try:
            num = int(t_id.split("-")[-1])
            if num > max_num:
                max_num = num
        except (ValueError, IndexError):
            pass
    ticket_id = f"JS-{year}-{str(max_num + 1).zfill(5)}"
    
    complaint = Complaint(
        ticket_id=ticket_id,
        user_id=user_id,
        description=description,
        images=json.dumps(images),
        lat=lat,
        lon=lon,
        location_source=location_source,
        category=category,
        department=department,
        priority=priority,
        urgency=urgency,
        severity=severity,
        ai_analysis=json.dumps(ai_result),
        is_authentic=is_auth,
        master_ticket_id=master_id
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)
    return complaint


def verify_live_detection(complaint_id: str, image_bytes: bytes) -> None:
    """Background task: confirm a live-camera pothole report with the server model.

    A confirmed detection is added as visual evidence and classification and
    routing are re-run (unless the citizen chose the category themselves); the
    result is stored in ai_analysis either way so admins can see whether the
    on-device detection held up.
    """
    try:
        detection = detect_potholes(image_bytes)
    except Exception as exc:
        print(f"[verify_live_detection] {complaint_id}: detection failed: {exc}")
        return

    db = SessionLocal()
    try:
        complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
        if not complaint:
            return
        ai_result = json.loads(complaint.ai_analysis)
        ai_result["camera_detection"] = detection
        ai_result["camera_detection_verified"] = detection["detected"]
        civic = ai_result.get("civic_ai")
        submission = (civic or {}).get("submission", {})
        if detection["detected"] and civic and not submission.get("confirmed_subcategory_id"):
            # Add the detector as visual evidence and re-run classification and
            # routing; the citizen's own choices (if any) still take precedence.
            prior = AnalysisResult.model_validate({k: v for k, v in civic.items() if k != "submission"})
            if not any(p.source == EvidenceSource.pothole_detector for p in prior.image_analysis.problems):
                prior.image_analysis.problems.append(VisualProblem(
                    subcategory_id="ROAD.POTHOLE", source=EvidenceSource.pothole_detector,
                    confidence=round(float(detection["max_confidence"]), 3),
                    evidence=f"Pothole detector (live camera) found {detection['count']} pothole(s)"))
                prior.image_analysis.detector = detection
            result = pipeline.analyze([], complaint.description, CitizenLocation(**submission.get("citizen_location", {})),
                                      run_vlm=False, run_detectors=False, prior=prior)
            if result.classification.subcategory_id == "ROAD.POTHOLE":
                sub = knowledge.subcategory("ROAD.POTHOLE")
                ai_result["category"] = complaint.category = sub["legacy_code"]
                ai_result["severity"] = complaint.severity = result.classification.severity or sub["severity"]
                ai_result["description"] = "Detected: a road surface pothole (live camera)."
                if not submission.get("confirmed_authority_id") and result.routing.authority_name:
                    ai_result["department"] = complaint.department = result.routing.authority_name
                ai_result["civic_ai"] = {**result.model_dump(mode="json"), "submission": submission}
        complaint.ai_analysis = json.dumps(ai_result)
        db.commit()
    finally:
        db.close()


def verify_authenticity(complaint_id: str, images: list[tuple[bytes, bool]]) -> None:
    """Background task: check each photo is a real capture.

    images: (bytes, is_live_capture) per uploaded photo. The complaint is marked
    is_authentic=False if any photo shows strong evidence of being AI-generated,
    a photo of a screen, or a screenshot; reasons go to ai_analysis for admins.
    """
    results = []
    for content, live in images:
        try:
            results.append(assess(content, live_capture=live))
        except Exception as exc:
            print(f"[verify_authenticity] {complaint_id}: assessment failed: {exc}")
    if not results:
        return

    db = SessionLocal()
    try:
        complaint = db.query(Complaint).filter(Complaint.id == complaint_id).first()
        if not complaint:
            return
        flagged = [r for r in results if not r["is_authentic"]]
        ai_result = json.loads(complaint.ai_analysis)
        ai_result["authenticity"] = {
            "verdict": flagged[0]["verdict"] if flagged else "real",
            "flagged": bool(flagged),
            "reasons": sorted({reason for r in results for reason in r["reasons"]}),
            "images": results,
        }
        complaint.ai_analysis = json.dumps(ai_result)
        complaint.is_authentic = not flagged
        db.commit()
    finally:
        db.close()
