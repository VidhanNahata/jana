"""All-category issue models: decoding, thresholds, graceful absence, pipeline integration.
No ONNX files are needed: the sessions are replaced by small fakes."""

import json
import os

import numpy as np
import pytest

from backend.ai import knowledge, pipeline
from backend.ai.routing_engine import CitizenLocation
from backend.ai.schema import EvidenceSource, StageStatus
from backend.config import settings
from backend.services import issue_models
from backend.tests.conftest import ROOT, jpeg

BLR = CitizenLocation(city="Bengaluru", locality="Jayanagar", lat=12.93, lon=77.58)

CLS_META = {
    "input_size": 64,
    "classes": ["pothole", "garbage", "drain_sewage", "background"],
    "background": "background",
    "subcategory": {"pothole": "ROAD.POTHOLE", "garbage": "WASTE.UNCOLLECTED", "drain_sewage": "DRAIN.OVERFLOW"},
    "thresholds": {"pothole": 0.5, "garbage": 0.6, "drain_sewage": 0.7},
}
DET_META = {
    "input_size": 640, "classes": ["pothole", "garbage", "open_manhole"],
    "subcategory": {"pothole": "ROAD.POTHOLE", "garbage": "WASTE.UNCOLLECTED", "open_manhole": "DRAIN.OPEN_MANHOLE"},
    "thresholds": {"pothole": 0.3, "garbage": 0.4, "open_manhole": 0.5},
}


class FakeSession:
    def __init__(self, out):
        self.out = out

    def get_inputs(self):
        return [type("I", (), {"name": "x"})()]

    def run(self, names, feed):
        return [self.out]


@pytest.fixture(autouse=True)
def clean_cache():
    issue_models.reset_cache()
    yield
    issue_models.reset_cache()


def install(kind, session, meta):
    issue_models._loaded[kind] = (session, meta)


def logits_for(probs):
    return np.log(np.array([probs], np.float32))


# ---- not installed ---------------------------------------------------------------

def test_absent_models_are_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ISSUE_CLASSIFIER_PATH", str(tmp_path / "none.onnx"))
    monkeypatch.setattr(settings, "ISSUE_DETECTOR_PATH", str(tmp_path / "none2.onnx"))
    assert issue_models.evidence([jpeg()]) == []
    monkeypatch.setattr(settings, "ISSUE_CLIP_HEAD_PATH", str(tmp_path / "none.npz"))
    st = issue_models.status()
    assert st["classifier"] == {"installed": False} and st["detector"] == {"installed": False}
    assert st["clip"]["installed"] is False


def test_corrupt_image_never_raises():
    install("classifier", FakeSession(logits_for([0.1, 0.8, 0.05, 0.05])), CLS_META)
    assert issue_models.evidence([b"not an image"]) == []


# ---- classifier -------------------------------------------------------------------

def test_classifier_thresholds_per_class_and_background_ignored():
    probs = np.array([0.05, 0.62, 0.30, 0.03])
    ev = issue_models.classifier_evidence(probs, CLS_META)
    assert [e["subcategory_id"] for e in ev] == ["WASTE.UNCOLLECTED"]       # 0.62 >= 0.6; drain 0.30 < 0.7
    assert ev[0]["source"] == "issue_classifier"
    background_wins = issue_models.classifier_evidence(np.array([0.01, 0.01, 0.01, 0.97]), CLS_META)
    assert background_wins == []


def test_evidence_is_capped_and_pothole_left_to_the_pothole_detector():
    # pothole 0.50 and garbage 0.45 both clear nothing special; make garbage 0.97 and pothole 0.9
    install("classifier", FakeSession(logits_for([0.02, 0.97, 0.005, 0.005])), CLS_META)
    ev = issue_models.evidence([jpeg()])
    assert len(ev) == 1 and ev[0]["subcategory_id"] == "WASTE.UNCOLLECTED"
    assert ev[0]["confidence"] == settings.ISSUE_MODEL_MAX_CONFIDENCE
    install("classifier", FakeSession(logits_for([0.9, 0.04, 0.03, 0.03])), CLS_META)
    assert issue_models.evidence([jpeg()]) == []        # ROAD.POTHOLE dropped here


# ---- detector decoding ------------------------------------------------------------

def make_preds(anchors):
    """anchors: list of (cx, cy, w, h, [score per class]) in 640-px letterboxed coordinates."""
    preds = np.zeros((4 + 3, 8400), np.float32)
    for i, (cx, cy, w, h, sc) in enumerate(anchors):
        preds[:4, i] = (cx, cy, w, h)
        preds[4:, i] = sc
    return preds


def test_decode_nms_thresholds_and_unletterbox():
    preds = make_preds([
        (320, 320, 100, 100, [0, 0.90, 0]),     # garbage
        (325, 322, 100, 100, [0, 0.60, 0]),     # same garbage, overlapping -> suppressed by NMS
        (100, 100, 40, 40, [0, 0, 0.45]),       # manhole 0.45 < 0.5 threshold -> dropped
        (500, 500, 60, 60, [0, 0, 0.80]),       # manhole kept
        (200, 500, 50, 50, [0, 0.35, 0]),       # garbage 0.35 < 0.4 -> dropped
    ])
    # 1280x960 original -> scale 0.5, letterbox pad_y = 80
    boxes = issue_models.decode_detections(preds, DET_META, scale=0.5, pad_x=0, pad_y=80, width=1280, height=960)
    assert [(b["class"], round(b["confidence"], 2)) for b in boxes] == [("garbage", 0.9), ("open_manhole", 0.8)]
    g = boxes[0]
    assert g["x1"] == pytest.approx((270 / 0.5) / 1280) and g["x2"] == pytest.approx((370 / 0.5) / 1280)
    assert g["y1"] == pytest.approx(((270 - 80) / 0.5) / 960)
    ev = issue_models.detector_evidence(boxes, DET_META)
    assert {e["subcategory_id"] for e in ev} == {"WASTE.UNCOLLECTED", "DRAIN.OPEN_MANHOLE"}
    assert all(e["source"] == "issue_detector" for e in ev)


def test_detector_end_to_end_with_fake_session():
    preds = make_preds([(320, 320, 200, 200, [0, 0.8, 0])])[None]
    install("detector", FakeSession(preds), DET_META)
    ev = issue_models.evidence([jpeg(size=(640, 480))])
    assert [(e["subcategory_id"], e["source"]) for e in ev] == [("WASTE.UNCOLLECTED", "issue_detector")]


# ---- pipeline integration -----------------------------------------------------------

def run(text, images):
    return pipeline.analyze(images, text, BLR, run_detectors=False)


def test_pipeline_uses_classifier_even_without_the_vlm():
    """VLM is disabled in tests: the trained classifier alone must still give image evidence."""
    install("classifier", FakeSession(logits_for([0.02, 0.88, 0.05, 0.05])), CLS_META)
    r = run("", [jpeg()])
    assert r.image_analysis.status == StageStatus.model_unavailable
    assert [(p.subcategory_id, p.source) for p in r.image_analysis.problems] == \
           [("WASTE.UNCOLLECTED", EvidenceSource.issue_classifier)]
    assert r.classification.category_id == "WASTE"
    assert EvidenceSource.issue_classifier in r.classification.sources
    assert "issue_models" in r.model_metadata.timings_ms


def test_pipeline_text_and_classifier_agree_raises_confidence():
    install("classifier", FakeSession(logits_for([0.02, 0.88, 0.05, 0.05])), CLS_META)
    both = run("Garbage has not been collected for a week", [jpeg()])
    install("classifier", FakeSession(logits_for([0.25, 0.25, 0.25, 0.25])), CLS_META)
    text_only = run("Garbage has not been collected for a week", [jpeg()])
    assert both.classification.confidence > text_only.classification.confidence


def test_pipeline_unchanged_when_no_models(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ISSUE_CLASSIFIER_PATH", str(tmp_path / "x.onnx"))
    r = run("Garbage has not been collected for a week", [jpeg()])
    assert r.image_analysis.problems == []


# ---- class map is valid against the taxonomy --------------------------------------------

def test_classes_json_maps_to_real_taxonomy_ids():
    cfg = json.load(open(os.path.join(ROOT, "training", "issues", "classes.json")))
    ids = list(cfg["classifier"]["classes"].values()) + list(cfg["detector"]["classes"].values())
    for sid in ids:
        assert knowledge.subcategory(sid), f"{sid} is not in the taxonomy"
    assert cfg["classifier"]["background"] not in cfg["classifier"]["classes"]
