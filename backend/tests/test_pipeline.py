"""Stages 1-7 end to end with a stubbed VLM (no weights loaded, detectors off)."""

import pytest

from backend.ai import pipeline, vision
from backend.ai.routing_engine import CitizenLocation
from backend.ai.schema import Consistency, RoutingStatus, StageStatus, Status
from backend.config import settings
from backend.tests.conftest import jpeg

BLR = CitizenLocation(city="Bengaluru", locality="Jayanagar", lat=12.93, lon=77.58)


def run(text, images=None, citizen=BLR, **kw):
    return pipeline.analyze(images or [], text, citizen, run_detectors=False, **kw)


def vlm_reply(*problems, civic=True, quality="good"):
    return {"scene": "street scene", "is_civic_scene": civic, "image_quality": quality,
            "problems": [{"id": p, "evidence": "visible", "confidence": c} for p, c in problems]}


# ---- image-text consistency -------------------------------------------------------

def test_consistent_image_and_text(vlm, image):
    vlm.reply = vlm_reply(("ROAD.POTHOLE", 0.9))
    r = run("Big pothole on the road", [image])
    assert r.fusion.consistency == Consistency.consistent
    assert r.classification.subcategory_id == "ROAD.POTHOLE"
    assert r.classification.confidence >= 0.9
    assert r.image_analysis.problems[0].source.value == "image_vlm"


def test_contradiction_is_flagged_not_out_of_scope(vlm, image):
    vlm.reply = vlm_reply(("WASTE.UNCOLLECTED", 0.85))
    r = run("Streetlight not working for 10 days", [image])
    assert r.fusion.consistency == Consistency.contradiction
    assert not r.classification.out_of_scope
    assert r.classification.human_review_required
    assert r.status in (Status.needs_review, Status.needs_citizen_input)


def test_unrelated_photo_with_civic_text_stays_in_scope(vlm, image):
    vlm.reply = vlm_reply(civic=False)
    r = run("Garbage not collected for a week", [image])
    assert r.fusion.consistency == Consistency.unrelated_image
    assert r.classification.subcategory_id == "WASTE.UNCOLLECTED"
    assert not r.classification.out_of_scope
    assert any("unrelated image" in x for x in r.classification.review_reasons)


def test_insufficient_image_evidence(vlm, image):
    vlm.reply = vlm_reply()
    r = run("Water pipe leaking", [image])
    assert r.fusion.consistency == Consistency.insufficient_image_evidence
    assert r.classification.subcategory_id == "WATER.PIPE_LEAK"
    assert r.classification.confidence < 0.9


def test_unrelated_photo_without_civic_text_is_out_of_scope(vlm, image):
    vlm.reply = vlm_reply(civic=False)
    r = run("please see", [image])
    assert r.status == Status.out_of_scope
    assert r.classification.subcategory_id == "OUT_OF_SCOPE.UNRELATED_IMAGE"
    assert r.complaint_draft is None


def test_image_only(vlm, image):
    vlm.reply = vlm_reply(("TREE.FALLEN", 0.8))
    r = run("", [image])
    assert r.fusion.consistency == Consistency.image_only
    assert r.classification.subcategory_id == "TREE.FALLEN"


# ---- multiple issues --------------------------------------------------------------

def test_multiple_issues_need_primary_selection(vlm, image):
    vlm.reply = vlm_reply(("WASTE.UNCOLLECTED", 0.85), ("ROAD.POTHOLE", 0.85))
    r = run("Garbage dumped and a pothole on the road", [image])
    assert r.classification.multiple_issues
    assert r.classification.primary_selection_required
    assert r.status == Status.needs_citizen_input
    ids = {r.classification.subcategory_id} | {a.subcategory_id for a in r.classification.alternatives}
    assert {"WASTE.UNCOLLECTED", "ROAD.POTHOLE"} <= ids


def test_citizen_choice_sets_primary(vlm, image):
    vlm.reply = vlm_reply(("WASTE.UNCOLLECTED", 0.85), ("ROAD.POTHOLE", 0.85))
    r = run("Garbage dumped and a pothole on the road", [image], confirmed_subcategory="ROAD.POTHOLE")
    assert r.classification.subcategory_id == "ROAD.POTHOLE"
    assert not r.classification.primary_selection_required
    assert "citizen_confirmed" in [s.value for s in r.classification.sources]


def test_out_of_scope_confirmation_is_rejected():
    r = run("pothole", confirmed_subcategory="OUT_OF_SCOPE.PERSONAL_DISPUTE")
    assert r.errors and r.classification.subcategory_id == "ROAD.POTHOLE"


# ---- out of scope -----------------------------------------------------------------

@pytest.mark.parametrize("text,sub", [
    ("I have not received my old age pension", "OUT_OF_SCOPE.NON_CIVIC_GRIEVANCE"),
    ("My neighbour is fighting with us about the parking", "OUT_OF_SCOPE.PERSONAL_DISPUTE"),
])
def test_out_of_scope(text, sub):
    r = run(text)
    assert r.status == Status.out_of_scope and r.classification.subcategory_id == sub
    assert r.classification.out_of_scope_reason
    assert r.routing.status == RoutingStatus.not_applicable
    assert r.complaint_draft is None


def test_vague_civic_text_goes_to_review_not_out_of_scope():
    r = run("problem in our area please help")
    assert not r.classification.out_of_scope
    assert r.classification.subcategory_id == "OTHER.CIVIC_INFRA"
    assert r.status in (Status.needs_review, Status.needs_citizen_input)


# ---- model failures ---------------------------------------------------------------

def test_malformed_output_is_retried_once(vlm, image):
    vlm.reply = ["Sure! The image shows a pothole.", vlm_reply(("ROAD.POTHOLE", 0.8))]
    r = run("pothole", [image])
    assert r.image_analysis.status == StageStatus.ok and vlm.calls == 2


def test_malformed_output_twice_falls_back_to_text(vlm, image):
    vlm.reply = "not json at all"
    r = run("Drain is blocked", [image])
    assert r.image_analysis.status == StageStatus.invalid_model_output
    assert r.fusion.consistency == Consistency.text_only
    assert r.classification.subcategory_id == "DRAIN.BLOCKED"
    assert any("image:" in e for e in r.errors)


def test_unknown_ids_from_model_are_dropped(vlm, image):
    vlm.reply = vlm_reply(("ROAD.MAGIC", 0.9), ("OUT_OF_SCOPE.UNRELATED_IMAGE", 0.9), ("ROAD.POTHOLE", 0.7))
    r = run("", [image])
    assert [p.subcategory_id for p in r.image_analysis.problems] == ["ROAD.POTHOLE"]
    assert len(r.image_analysis.warnings) == 2


def test_missing_model_degrades_to_text(monkeypatch, image, tmp_path):
    monkeypatch.setattr(settings, "AI_VLM_ENABLED", True)
    monkeypatch.setattr(settings, "AI_VLM_PATH", str(tmp_path / "no-model"))
    r = run("Garbage everywhere", [image])
    assert r.image_analysis.status == StageStatus.model_unavailable
    assert "not installed" in r.image_analysis.error
    assert r.classification.subcategory_id == "WASTE.UNCOLLECTED"
    assert r.model_metadata.vlm["installed"] is False


def test_corrupted_image(vlm):
    r = run("pothole", [b"\xff\xd8 definitely not a jpeg"])
    assert r.image_analysis.status == StageStatus.invalid_input
    assert r.classification.subcategory_id == "ROAD.POTHOLE"


def test_bad_photo_quality_needs_review(vlm, image):
    vlm.reply = vlm_reply(("ROAD.POTHOLE", 0.9), quality="blurry")
    r = run("pothole", [image])
    assert any("quality" in x for x in r.classification.review_reasons)


# ---- authenticity stays separate --------------------------------------------------

def test_flagged_authenticity_adds_review_but_keeps_classification(vlm, image, monkeypatch):
    vlm.reply = vlm_reply(("ROAD.POTHOLE", 0.9))
    fake = {"status": "ok", "verdict": "ai_generated", "flagged": True, "reasons": ["AI-generated"], "images": [],
            "errors": []}
    monkeypatch.setattr(pipeline, "_authenticity", lambda imgs, live: fake)
    monkeypatch.setattr(pipeline, "_detector", lambda imgs: [None] * len(imgs))
    r = pipeline.analyze([image], "pothole", BLR)
    assert r.authenticity == fake
    assert r.classification.subcategory_id == "ROAD.POTHOLE"
    assert any("authenticity" in x for x in r.classification.review_reasons)
    assert r.status != Status.ok


def test_detector_evidence_is_kept_separate(monkeypatch, image):
    det = {"detected": True, "count": 2, "max_confidence": 0.81}
    monkeypatch.setattr(pipeline, "_authenticity", lambda imgs, live: None)
    monkeypatch.setattr(pipeline, "_detector", lambda imgs: [det])
    r = pipeline.analyze([image], "", BLR)          # VLM disabled in tests
    assert r.image_analysis.status == StageStatus.model_unavailable
    assert r.image_analysis.problems[0].source.value == "pothole_detector"
    assert r.classification.subcategory_id == "ROAD.POTHOLE"


# ---- routing ----------------------------------------------------------------------

def test_bengaluru_needs_corporation():
    r = run("pothole on the main road")
    assert r.routing.status == RoutingStatus.confirmation_required
    assert "city_corporation" in r.routing.missing_location_fields
    alts = {a.authority_id for a in r.routing.alternatives}
    assert {"KA.BLR.CENTRAL", "KA.BLR.NORTH", "KA.BLR.SOUTH", "KA.BLR.EAST", "KA.BLR.WEST"} <= alts
    assert r.status == Status.needs_citizen_input


def test_bengaluru_with_confirmed_corporation():
    c = CitizenLocation(city="Bengaluru", city_corporation_id="KA.BLR.SOUTH")
    r = run("pothole on the main road", citizen=c)
    assert r.routing.authority_id == "KA.BLR.SOUTH"
    assert "city_corporation" not in r.routing.missing_location_fields


def test_bengaluru_water_goes_to_bwssb():
    r = run("Water pipe burst near 560041", citizen=CitizenLocation())
    assert r.routing.authority_id == "KA.BWSSB"
    assert r.routing.verified_url == "https://bwssb.karnataka.gov.in/"


def test_mandya_wiring_goes_to_cesc():
    r = run("Live electric wire hanging from the pole", citizen=CitizenLocation(district="Mandya"))
    assert r.routing.authority_id == "KA.ESCOM.CESC"
    assert r.routing.status == RoutingStatus.resolved


def test_rural_drain_is_not_configured():
    r = run("Drain is blocked in our village", citizen=CitizenLocation(district="Tumakuru", rural=True))
    assert r.routing.authority_id == "KA.GP.GENERIC"
    assert r.routing.status == RoutingStatus.authority_not_configured
    assert r.routing.verified_url is None
    assert r.status == Status.needs_review


def test_mysuru_urban_goes_to_city_corporation():
    r = run("Streetlight not working", citizen=CitizenLocation(city="Mysuru"))
    assert r.routing.authority_id == "KA.MC.MYSURU"
    assert r.routing.verified_url is None          # no verified URL in the registry: never invented


def test_no_location_requires_location():
    r = run("Garbage not collected", citizen=CitizenLocation())
    assert r.routing.status == RoutingStatus.location_required
    assert r.status == Status.needs_citizen_input


def test_highway_goes_to_nhai():
    r = run("Big potholes on NH 275 near Mandya", citizen=CitizenLocation(district="Mandya"))
    assert r.routing.authority_id == "IN.NHAI"


def test_metro_goes_to_bmrcl():
    r = run("Escalator broken at the metro station", citizen=BLR)
    ids = {r.routing.authority_id} | {a.authority_id for a in r.routing.alternatives}
    assert "KA.BMRCL" in ids


def test_every_verified_url_comes_from_registry():
    from backend.ai import knowledge
    urls = {a.get("official_source_url") for a in knowledge.all_authorities().values()}
    for text, c in [("pothole", BLR), ("water leak 560041", CitizenLocation()),
                    ("exposed wire", CitizenLocation(district="Mysuru"))]:
        r = run(text, citizen=c)
        assert r.routing.verified_url is None or r.routing.verified_url in urls


# ---- drafting ---------------------------------------------------------------------

def test_english_draft_separates_findings_and_claims(vlm, image):
    vlm.reply = vlm_reply(("ROAD.POTHOLE", 0.9))
    r = run("Big pothole near the bus stop for 2 weeks", [image],
            citizen=CitizenLocation(city="Bengaluru", city_corporation_id="KA.BLR.SOUTH", locality="Jayanagar"))
    d = r.complaint_draft
    assert d.language == "en"
    assert "Big pothole near the bus stop for 2 weeks" in d.body
    assert "Jayanagar" in d.body
    assert d.subject


def test_kannada_draft():
    r = run("ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಗುಂಡಿ ಇದೆ", draft_language="kn",
            citizen=CitizenLocation(city="Bengaluru", city_corporation_id="KA.BLR.EAST"))
    d = r.complaint_draft
    assert d.language == "kn"
    assert any("ಀ" <= ch <= "೿" for ch in d.body)


def test_draft_lists_missing_details():
    r = run("Garbage not collected", citizen=CitizenLocation())
    assert r.complaint_draft.missing_details


def test_result_is_schema_valid_json():
    from backend.ai.schema import AnalysisResult
    r = run("pothole")
    again = AnalysisResult.model_validate_json(r.model_dump_json())
    assert again.analysis_id == r.analysis_id
    assert pipeline.cached(r.analysis_id) is r


def test_non_civic_text_with_civic_photo_is_flagged(monkeypatch, image):
    det = {"detected": True, "count": 1, "max_confidence": 0.8}
    monkeypatch.setattr(pipeline, "_authenticity", lambda imgs, live: None)
    monkeypatch.setattr(pipeline, "_detector", lambda imgs: [det])
    r = pipeline.analyze([image], "I have not received my pension", BLR)
    assert r.classification.subcategory_id == "ROAD.POTHOLE" and not r.classification.out_of_scope
    assert any("non-civic" in x for x in r.classification.review_reasons)


# ---- local LLM reading of the text --------------------------------------------------

def test_llm_reading_adds_paraphrased_problem(vlm, image):
    vlm.reply = vlm_reply()
    vlm.text_reply = {"problems": [{"id": "ROAD.SURFACE_DAMAGE", "confidence": 0.8}], "out_of_scope": None}
    r = run("Tar has peeled off the whole stretch", [image])
    c = r.text_analysis.candidates[0]
    assert c.subcategory_id == "ROAD.SURFACE_DAMAGE" and c.sources == ["local_llm"]
    assert r.classification.subcategory_id == "ROAD.SURFACE_DAMAGE"
    assert r.text_analysis.llm_reading["problems"][0]["id"] == "ROAD.SURFACE_DAMAGE"


def test_llm_agreement_raises_lexicon_score(vlm, image):
    from backend.ai.text_analysis import analyze_text
    base = analyze_text("pothole on the road").candidates[0].score
    vlm.text_reply = {"problems": [{"id": "ROAD.POTHOLE", "confidence": 0.9}], "out_of_scope": None}
    r = run("pothole on the road", [image])
    c = r.text_analysis.candidates[0]
    assert c.score > base and set(c.sources) == {"lexicon", "local_llm"}


def test_llm_out_of_scope_reading(vlm):
    vlm.reply = vlm_reply(civic=False)
    vlm.text_reply = {"problems": [], "out_of_scope": "NON_CIVIC_GRIEVANCE"}
    r = run("Online shopping company did not refund my money", [jpeg()])
    assert r.status == Status.out_of_scope
    assert r.classification.subcategory_id == "OUT_OF_SCOPE.NON_CIVIC_GRIEVANCE"
    assert "language model" in r.classification.out_of_scope_reason


def test_llm_cannot_override_civic_lexicon_match(vlm, image):
    vlm.text_reply = {"problems": [], "out_of_scope": "PERSONAL_DISPUTE"}
    r = run("Garbage not collected near my neighbour's house", [image])
    assert not r.classification.out_of_scope


def test_unparseable_llm_reading_is_ignored(vlm, image):
    vlm.text_reply = "I think it is a road problem."
    r = run("pothole", [image])
    assert r.text_analysis.llm_reading is None
    assert r.classification.subcategory_id == "ROAD.POTHOLE"


def test_llm_unknown_ids_dropped(vlm, image):
    vlm.text_reply = {"problems": [{"id": "ROAD.HOVERCRAFT", "confidence": 0.9}], "out_of_scope": None}
    r = run("pothole", [image])
    assert r.text_analysis.llm_reading["warnings"]
    assert all(c.subcategory_id != "ROAD.HOVERCRAFT" for c in r.text_analysis.candidates)


def test_results_do_not_depend_on_hash_seed():
    import os
    import subprocess
    import sys
    code = ("from backend.ai import pipeline; from backend.ai.routing_engine import CitizenLocation as C;"
            "print(pipeline.analyze([], 'Retaining wall along the lake bund has collapsed and garbage in drain',"
            " C(), run_vlm=False, run_detectors=False).classification.subcategory_id)")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed), "AI_VLM_ENABLED": "false"}).stdout
            for seed in (0, 1, 2, 3)}
    assert len(outs) == 1 and outs.pop().strip()


def test_llm_text_reading_is_off_by_default():
    from backend.config import Settings
    assert Settings().AI_LLM_TEXT_CLASSIFY is False
