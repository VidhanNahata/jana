"""HTTP flow: analyse -> citizen review -> submit with the analysis_id."""

import json

from backend.tests.conftest import jpeg

BLR_FORM = {"latitude": "12.93", "longitude": "77.58", "location_source": "gps", "city": "Bengaluru"}


def _files(n=1):
    return [("images", (f"p{i}.jpg", jpeg(), "image/jpeg")) for i in range(n)]


def test_analyze_endpoint_returns_schema(client, auth):
    r = client.post("/api/ai/analyze", headers=auth, files=_files(),
                    data={"description": "Big pothole on the road", "city": "Bengaluru"})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("status", "scope", "image_analysis", "text_analysis", "fusion", "classification", "routing",
                "complaint_draft", "authenticity", "model_metadata"):
        assert key in body
    assert body["classification"]["subcategory_id"] == "ROAD.POTHOLE"
    assert body["authenticity"]["status"] == "ok"          # existing fake-image detector ran
    assert body["model_metadata"]["offline_mode"] is True


def test_analyze_validates_ids(client, auth):
    r = client.post("/api/ai/analyze", headers=auth, data={"description": "x", "confirmed_subcategory_id": "NOPE"})
    assert r.status_code == 400
    r = client.post("/api/ai/analyze", headers=auth, data={"description": "x", "city_corporation_id": "KA.FAKE"})
    assert r.status_code == 400


def test_submit_with_reviewed_analysis(client, auth):
    a = client.post("/api/ai/analyze", headers=auth, files=_files(),
                    data={"description": "Garbage dumped and a pothole on the road", "city": "Bengaluru"}).json()
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Garbage dumped and a pothole on the road", "analysis_id": a["analysis_id"],
        "confirmed_subcategory_id": "ROAD.POTHOLE", "city_corporation_id": "KA.BLR.SOUTH",
        "complaint_text": "Edited draft text"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["category"] == "pothole"
    assert body["subcategory_id"] == "ROAD.POTHOLE"
    assert body["authority_id"] == "KA.BLR.SOUTH"
    assert "South" in body["department"]


def test_submit_without_analysis_still_classifies(client, auth):
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Water pipe burst and water is leaking", "pin_code": "560041"})
    assert r.status_code == 201, r.text
    assert r.json()["category"] == "water_leak"
    assert r.json()["authority_id"] == "KA.BWSSB"


def test_submit_out_of_scope_is_rejected(client, auth):
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "I have not received my pension for three months"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "out_of_scope"


def test_submit_rejects_authority_that_does_not_handle_problem(client, auth):
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Big pothole on the road", "confirmed_authority_id": "KA.BWSSB"})
    assert r.status_code == 400


def test_bengaluru_without_corporation_is_pending(client, auth):
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Streetlight not working"})
    assert r.status_code == 201
    assert r.json()["routing_status"] == "confirmation_required"


def test_full_result_is_stored_for_admins(client, auth):
    from backend.database import SessionLocal
    from backend.models.complaint import Complaint
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Tree fallen on the road", "city_corporation_id": "KA.BLR.EAST"})
    db = SessionLocal()
    try:
        c = db.query(Complaint).filter(Complaint.ticket_id == r.json()["ticket_id"]).first()
        civic = json.loads(c.ai_analysis)["civic_ai"]
        assert civic["classification"]["subcategory_id"] == "TREE.FALLEN"
        assert civic["submission"]["citizen_location"]["city_corporation_id"] == "KA.BLR.EAST"
    finally:
        db.close()


def test_reference_endpoints(client):
    assert client.get("/api/ai/taxonomy").json()["version"]
    auths = client.get("/api/ai/authorities").json()["authorities"]
    assert any(a["id"] == "KA.BWSSB" for a in auths)
    status = client.get("/api/ai/status").json()
    assert status["offline_mode"] is True and "vlm" in status


def test_reanalysis_reuses_prior_and_applies_choices(client, auth):
    a = client.post("/api/ai/analyze", headers=auth, files=_files(),
                    data={"description": "Pothole on the road", "city": "Bengaluru"}).json()
    assert a["status"] == "needs_citizen_input"
    b = client.post("/api/ai/analyze", headers=auth, data={
        "description": "Pothole on the road", "city": "Bengaluru", "analysis_id": a["analysis_id"],
        "city_corporation_id": "KA.BLR.WEST", "draft_language": "kn"}).json()
    assert b["routing"]["authority_id"] == "KA.BLR.WEST"
    assert b["complaint_draft"]["language"] == "kn"
    assert b["complaint_draft"]["attachment_refs"] == ["p0.jpg"]
    assert b["authenticity"] == a["authenticity"]


def test_expired_analysis(client, auth):
    r = client.post("/api/ai/analyze", headers=auth, data={"description": "x", "analysis_id": "missing"})
    assert r.status_code == 410


def test_citizen_chosen_corporation_counts_as_confirmed(client, auth):
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Pothole on the road", "city_corporation_id": "KA.BLR.NORTH"})
    assert r.json()["department_basis"] == "citizen_confirmed"


def test_admin_summary_of_civic_ai(client, auth):
    from backend.database import SessionLocal
    from backend.models.complaint import Complaint
    from backend.routers.admin import _civic_ai
    r = client.post("/api/complaints", headers=auth, files=_files(), data={
        **BLR_FORM, "description": "Streetlight not working"})
    db = SessionLocal()
    try:
        s = _civic_ai(db.query(Complaint).filter(Complaint.ticket_id == r.json()["ticket_id"]).first())
    finally:
        db.close()
    assert s["subcategory_id"] == "LIGHT.STREETLIGHT_OUT"
    assert s["department_basis"] in ("registry_unconfirmed", "unresolved")
