"""The pipeline must run with all network access blocked."""

import socket

import pytest

from backend.ai import pipeline
from backend.ai.routing_engine import CitizenLocation


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_full_pipeline_offline(no_network, vlm, image):
    vlm.reply = {"scene": "road", "is_civic_scene": True, "image_quality": "good",
                 "problems": [{"id": "ROAD.POTHOLE", "evidence": "hole", "confidence": 0.9}]}
    vlm.translation = "There is a big pothole on the road"
    r = pipeline.analyze([image], "ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಗುಂಡಿ ಇದೆ", CitizenLocation(district="Mandya"), draft_language="kn")
    assert r.classification.subcategory_id == "ROAD.POTHOLE"
    assert r.text_analysis.translation_en == "There is a big pothole on the road"
    assert r.authenticity["status"] == "ok"       # real fake-image ONNX model, local
    assert r.complaint_draft.language == "kn"


def test_strict_offline_sets_hf_env(monkeypatch):
    import os
    from backend.ai import vision
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    vision.enforce_offline()
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
