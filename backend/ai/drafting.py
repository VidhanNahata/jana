"""
Stage 6: factual complaint draft in English or Kannada.

Template-based on purpose: every sentence is filled from validated fields
(classification, visual evidence, the citizen's own words, location fields),
so the draft cannot invent dates, locations, causes, legal violations,
responsible persons or damage estimates. Visual findings and citizen-reported
details are kept in separate, labelled sections. The citizen edits the draft
before submitting; nothing here claims the authority received anything.
"""

from backend.ai import knowledge
from backend.ai.schema import (Classification, ComplaintDraft, Consistency, Fusion, ImageAnalysis, Routing,
                               RoutingStatus, StageStatus, TextAnalysis)

ACTIONS = {
    "ROAD": ("inspect and repair the damaged road / footpath surface", "ಹಾಳಾದ ರಸ್ತೆ / ಪಾದಚಾರಿ ಮಾರ್ಗವನ್ನು ಪರಿಶೀಲಿಸಿ ದುರಸ್ತಿ ಮಾಡಬೇಕು"),
    "DRAIN": ("clear and repair the drain and make any open manhole safe", "ಚರಂಡಿಯನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ ದುರಸ್ತಿ ಮಾಡಬೇಕು ಮತ್ತು ತೆರೆದ ಮ್ಯಾನ್‌ಹೋಲ್ ಇದ್ದರೆ ಸುರಕ್ಷಿತಗೊಳಿಸಬೇಕು"),
    "WATER": ("stop the leak and repair the water-supply infrastructure", "ನೀರಿನ ಸೋರಿಕೆಯನ್ನು ತಡೆದು ನೀರು ಸರಬರಾಜು ಮೂಲಸೌಕರ್ಯವನ್ನು ದುರಸ್ತಿ ಮಾಡಬೇಕು"),
    "WASTE": ("clear the waste and keep the spot clean", "ಕಸವನ್ನು ತೆರವುಗೊಳಿಸಿ ಸ್ಥಳವನ್ನು ಸ್ವಚ್ಛವಾಗಿಡಬೇಕು"),
    "LIGHT": ("inspect, make safe and repair the streetlight / electrical infrastructure", "ಬೀದಿದೀಪ / ವಿದ್ಯುತ್ ಮೂಲಸೌಕರ್ಯವನ್ನು ಪರಿಶೀಲಿಸಿ ಸುರಕ್ಷಿತಗೊಳಿಸಿ ದುರಸ್ತಿ ಮಾಡಬೇಕು"),
    "TRANSPORT": ("repair or replace the damaged road / transport infrastructure", "ಹಾಳಾದ ರಸ್ತೆ / ಸಾರಿಗೆ ಮೂಲಸೌಕರ್ಯವನ್ನು ದುರಸ್ತಿ ಅಥವಾ ಬದಲಾಯಿಸಬೇಕು"),
    "PARK": ("repair the public amenity", "ಸಾರ್ವಜನಿಕ ಸೌಲಭ್ಯವನ್ನು ದುರಸ್ತಿ ಮಾಡಬೇಕು"),
    "TREE": ("clear the obstruction and make the tree safe", "ಅಡಚಣೆಯನ್ನು ತೆರವುಗೊಳಿಸಿ ಮರವನ್ನು ಸುರಕ್ಷಿತಗೊಳಿಸಬೇಕು"),
    "BUILDING": ("inspect and repair the public structure", "ಸಾರ್ವಜನಿಕ ರಚನೆಯನ್ನು ಪರಿಶೀಲಿಸಿ ದುರಸ್ತಿ ಮಾಡಬೇಕು"),
    "STORMWATER": ("clear the stormwater drainage and resolve the waterlogging", "ಮಳೆನೀರು ಚರಂಡಿಯನ್ನು ತೆರವುಗೊಳಿಸಿ ನೀರು ನಿಲ್ಲುವುದನ್ನು ತಡೆಯಬೇಕು"),
    "ACCESS": ("remove the obstruction and restore safe pedestrian access", "ಅಡಚಣೆಯನ್ನು ತೆರವುಗೊಳಿಸಿ ಸುರಕ್ಷಿತ ಪಾದಚಾರಿ ಪ್ರವೇಶ ಒದಗಿಸಬೇಕು"),
    "OTHER": ("inspect the location and take the necessary corrective action", "ಸ್ಥಳವನ್ನು ಪರಿಶೀಲಿಸಿ ಅಗತ್ಯ ಸರಿಪಡಿಸುವ ಕ್ರಮ ಕೈಗೊಳ್ಳಬೇಕು"),
}

L = {
    "en": {
        "subject": "Complaint: {issue}{place}",
        "to": "To: {authority}",
        "to_unknown": "To: (responsible authority to be confirmed)",
        "category": "Problem: {cat} — {sub}",
        "visual": "Visible in the attached photo (automated analysis):",
        "no_visual": "Visible in the attached photo: not assessed automatically.",
        "citizen": "Reported by the citizen:",
        "location": "Location: {loc}",
        "location_missing": "Location: to be added by the citizen.",
        "gps": "GPS: {lat:.6f}, {lon:.6f}",
        "duration": "Duration (as reported): {d}",
        "impact": "Public impact (as reported): {cues}",
        "request": "Requested action: Please {action}.",
        "photos": "Attached photos: {refs}",
        "missing": "Details to confirm before submitting:",
        "note_mismatch": "Note: the automated check found that the photo may not show the reported problem ({c}).",
    },
    "kn": {
        "subject": "ದೂರು: {issue}{place}",
        "to": "ಗೆ: {authority}",
        "to_unknown": "ಗೆ: (ಜವಾಬ್ದಾರಿಯುತ ಪ್ರಾಧಿಕಾರವನ್ನು ದೃಢೀಕರಿಸಬೇಕಿದೆ)",
        "category": "ಸಮಸ್ಯೆ: {cat} — {sub}",
        "visual": "ಲಗತ್ತಿಸಿದ ಛಾಯಾಚಿತ್ರದಲ್ಲಿ ಕಾಣುವುದು (ಸ್ವಯಂಚಾಲಿತ ವಿಶ್ಲೇಷಣೆ):",
        "no_visual": "ಲಗತ್ತಿಸಿದ ಛಾಯಾಚಿತ್ರ: ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಪರಿಶೀಲಿಸಲಾಗಿಲ್ಲ.",
        "citizen": "ನಾಗರಿಕರು ತಿಳಿಸಿರುವುದು:",
        "location": "ಸ್ಥಳ: {loc}",
        "location_missing": "ಸ್ಥಳ: ನಾಗರಿಕರು ಸೇರಿಸಬೇಕು.",
        "gps": "ಜಿಪಿಎಸ್: {lat:.6f}, {lon:.6f}",
        "duration": "ಅವಧಿ (ತಿಳಿಸಿದಂತೆ): {d}",
        "impact": "ಸಾರ್ವಜನಿಕ ಪರಿಣಾಮ (ತಿಳಿಸಿದಂತೆ): {cues}",
        "request": "ವಿನಂತಿ: ದಯವಿಟ್ಟು {action}.",
        "photos": "ಲಗತ್ತಿಸಿದ ಛಾಯಾಚಿತ್ರಗಳು: {refs}",
        "missing": "ಸಲ್ಲಿಸುವ ಮೊದಲು ದೃಢೀಕರಿಸಬೇಕಾದ ವಿವರಗಳು:",
        "note_mismatch": "ಸೂಚನೆ: ಛಾಯಾಚಿತ್ರದಲ್ಲಿ ತಿಳಿಸಿದ ಸಮಸ್ಯೆ ಕಾಣದಿರಬಹುದು ಎಂದು ಸ್ವಯಂಚಾಲಿತ ಪರಿಶೀಲನೆ ಕಂಡುಕೊಂಡಿದೆ ({c}).",
    },
}

# Size words raise severity but are not a public impact
SIZE_WORDS = {"big", "huge", "deep", "ದೊಡ್ಡ", "ಆಳ", "doddadu", "aala"}

MISSING_LABELS = {
    "locality_or_landmark": ("locality or nearby landmark", "ಬಡಾವಣೆ ಅಥವಾ ಹತ್ತಿರದ ಹೆಗ್ಗುರುತು"),
    "town_or_city": ("town or city", "ಪಟ್ಟಣ ಅಥವಾ ನಗರ"),
    "city_corporation": ("which Bengaluru city corporation covers this locality",
                         "ಈ ಬಡಾವಣೆ ಯಾವ ಬೆಂಗಳೂರು ನಗರ ಪಾಲಿಕೆಗೆ ಸೇರಿದೆ"),
    "district": ("district", "ಜಿಲ್ಲೆ"),
    "village": ("village", "ಗ್ರಾಮ"),
    "taluk": ("taluk", "ತಾಲ್ಲೂಕು"),
    "asset_owner": ("agency that owns the asset", "ಆಸ್ತಿಯ ಮಾಲೀಕ ಸಂಸ್ಥೆ"),
    "duration": ("since when the problem exists (optional)", "ಸಮಸ್ಯೆ ಎಷ್ಟು ದಿನಗಳಿಂದ ಇದೆ (ಐಚ್ಛಿಕ)"),
    "primary_issue": ("which problem is the main one", "ಮುಖ್ಯ ಸಮಸ್ಯೆ ಯಾವುದು"),
    "photo": ("a clear photo of the problem", "ಸಮಸ್ಯೆಯ ಸ್ಪಷ್ಟ ಛಾಯಾಚಿತ್ರ"),
}


def draft_complaint(lang: str, classification: Classification, image: ImageAnalysis, text: TextAnalysis,
                    fusion: Fusion, routing: Routing, image_refs: list[str],
                    lat: float | None = None, lon: float | None = None, locality: str | None = None) -> ComplaintDraft:
    lang = "kn" if lang == "kn" else "en"
    t = L[lang]
    kn = lang == "kn"
    cat_id = classification.category_id or "OTHER"
    sub_name = (classification.subcategory_name_kn if kn else classification.subcategory_name_en) or ""
    cat_name = classification.category_name_en or ""
    if kn:
        sub = knowledge.subcategory(classification.subcategory_id or "") or {}
        cat_name = sub.get("category_name_kn") or cat_name

    area = routing.jurisdiction.city or routing.jurisdiction.district
    if kn and area:
        gaz = knowledge.gazetteer()
        area = next((c["kn"] for c in gaz["corporation_cities"] if c["city"] == area), None) or \
            next((d["kn"] for d in gaz["districts"] if d["name"] == area), area)
    place_parts = [p for p in [locality, *text.location.localities[:1], *text.location.landmarks[:1], area] if p]
    place_parts = list(dict.fromkeys(place_parts))
    place = ", ".join(place_parts)

    lines = [t["to"].format(authority=(routing.authority_name_kn if kn and routing.authority_name_kn
                                       else routing.authority_name))
             if routing.authority_name and routing.status != RoutingStatus.location_required else t["to_unknown"], ""]
    lines.append(t["category"].format(cat=cat_name, sub=sub_name))
    lines.append("")
    visual = [p for p in image.problems]
    if image.status == StageStatus.ok or visual:
        lines.append(t["visual"])
        if image.scene_description:
            lines.append(f"- {image.scene_description}")
        for p in visual:
            name = knowledge.subcategory(p.subcategory_id)["name_kn" if kn else "name_en"]
            lines.append(f"- {name}: {p.evidence}")
    else:
        lines.append(t["no_visual"])
    if fusion.consistency in (Consistency.contradiction, Consistency.unrelated_image,
                              Consistency.insufficient_image_evidence):
        lines.append(t["note_mismatch"].format(c=fusion.consistency.value.replace("_", " ")))
    lines.append("")
    if text.original_text:
        lines.append(t["citizen"])
        lines.append(f"“{text.original_text}”")
        lines.append("")
    lines.append(t["location"].format(loc=place) if place else t["location_missing"])
    if lat is not None and lon is not None:
        lines.append(t["gps"].format(lat=lat, lon=lon))
    if text.duration:
        lines.append(t["duration"].format(d=text.duration))
    impact = [c for c in text.severity_cues if c not in SIZE_WORDS]
    if impact:
        lines.append(t["impact"].format(cues=", ".join(impact)))
    lines.append("")
    action = ACTIONS.get(cat_id, ACTIONS["OTHER"])[1 if kn else 0]
    lines.append(t["request"].format(action=action))
    if image_refs:
        lines.append(t["photos"].format(refs=", ".join(image_refs)))

    missing = list(routing.missing_location_fields)
    if not place and lat is None:
        missing.append("locality_or_landmark")
    if not text.duration:
        missing.append("duration")
    if classification.primary_selection_required:
        missing.append("primary_issue")
    if not image_refs:
        missing.append("photo")
    missing = list(dict.fromkeys(missing))
    missing_labels = [MISSING_LABELS.get(m, (m, m))[1 if kn else 0] for m in missing]
    if missing_labels:
        lines.append("")
        lines.append(t["missing"])
        lines.extend(f"- {m}" for m in missing_labels)

    subject = t["subject"].format(issue=sub_name or cat_name, place=f" — {place}" if place else "")
    return ComplaintDraft(language=lang, subject=subject, body="\n".join(lines).strip(),
                          missing_details=missing_labels, attachment_refs=image_refs)
