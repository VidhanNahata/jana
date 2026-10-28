from backend.routers.complaints import parse_dashcam
from backend.services.priority import assess_priority


def test_no_evidence_keeps_standard_urgency():
    p = assess_priority(3, "ROAD.POTHOLE")
    assert (p["urgency"], p["priority"]) == (2, "P2")


def test_fast_road_raises_urgency():
    p = assess_priority(3, "ROAD.POTHOLE", dashcam={"speed_kmh": 50})
    assert (p["urgency"], p["priority"]) == (3, "P1")
    assert any("50 km/h" in f for f in p["factors"])


def test_large_detection_counts_only_for_the_classified_problem():
    det = {"boxes": [], "road_damage": {"detections": [
        {"subcategory_id": "ROAD.SURFACE_DAMAGE", "area": 0.2}]}}
    assert assess_priority(2, "ROAD.POTHOLE", det)["urgency"] == 2
    assert assess_priority(2, "ROAD.SURFACE_DAMAGE", det)["urgency"] == 3


def test_dashcam_evidence_is_clamped():
    d = parse_dashcam('{"speed_kmh": 999, "nearby_detections": -4, "detector_class": "pothole"}')
    assert d["speed_kmh"] == 200.0 and d["nearby_detections"] == 0
