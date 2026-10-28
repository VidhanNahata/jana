def calculate_priority(urgency: int, severity: int) -> str:
    """
    ITIL Priority Matrix
    Urgency(1-3) + Severity(1-3) - 1
    3=High, 2=Medium, 1=Low
    Formula yields 1-5, mapped to P5-P1
    Value 5 -> P1
    Value 4 -> P2
    Value 3 -> P3
    Value 2 -> P4
    Value 1 -> P5
    """
    # Clip values to 1-3 range
    u = max(1, min(3, urgency))
    s = max(1, min(3, severity))
    
    score = u + s - 1
    # 5->P1, 4->P2, 3->P3, 2->P4, 1->P5
    # Priority = 6 - score
    p_level = 6 - score
    return f"P{p_level}"

def boost_priority(current_priority: str, child_count: int) -> str:
    """Every 5 children -> priority escalates by 1 level"""
    if not current_priority.startswith("P"):
        return current_priority
    
    level = int(current_priority[1:])
    boosts = child_count // 5
    
    new_level = max(1, level - boosts)
    return f"P{new_level}"


# ── Evidence-based urgency (dashcam and photo detections) ────────────────
# Urgency starts at 2 (the old fixed value) and rises to 3 when the evidence
# shows the hazard is more dangerous than average. Every factor is listed so
# officials can see why a ticket got its priority. Thresholds are heuristics,
# not learned from outcome data.
LARGE_AREA = 0.04        # detection box covers >= 4% of the photo (close and/or big)
FAST_ROAD_KMH = 30       # device GPS speed when the dashcam saw it
CLUSTER_COUNT = 3        # defects within ~100 m on the same drive


def assess_priority(severity: int, subcategory_id: str | None = None, detector: dict | None = None,
                    dashcam: dict | None = None) -> dict:
    factors = []
    areas = []
    for d in ((detector or {}).get("road_damage") or {}).get("detections", []):
        if d.get("subcategory_id") == subcategory_id:
            areas.append(d["area"])
    if subcategory_id == "ROAD.POTHOLE":
        areas += [(b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) for b in (detector or {}).get("boxes", [])]
    if areas and max(areas) >= LARGE_AREA:
        factors.append(f"Large in the photo ({max(areas) * 100:.0f}% of the frame, server detector)")
    dashcam = dashcam or {}
    speed = dashcam.get("speed_kmh")
    if speed is not None and speed >= FAST_ROAD_KMH:
        factors.append(f"On a road driven at about {speed:.0f} km/h (device GPS)")
    nearby = dashcam.get("nearby_detections")
    if nearby is not None and nearby >= CLUSTER_COUNT:
        factors.append(f"{nearby} road defects within about 100 m on this drive (device)")
    urgency = 3 if factors else 2
    return {"priority": calculate_priority(urgency, severity), "urgency": urgency, "severity": severity,
            "factors": factors or ["No extra risk factors; standard urgency"]}
