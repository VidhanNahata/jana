"""
Test setup: an isolated SQLite database, the vision-language model disabled
by default (tests that need it use the `vlm` fixture, which stubs generation
so no weights are loaded), and helpers for images.
"""

import io
import json
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="janasutra_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["AI_VLM_ENABLED"] = "false"
os.environ["AI_STRICT_OFFLINE"] = "true"

import pytest
from PIL import Image

from backend.ai import vision
from backend.config import settings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def jpeg(color=(120, 120, 120), size=(320, 240)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def image():
    return jpeg()


@pytest.fixture
def vlm(monkeypatch):
    """Stub the VLM: set `vlm.reply` to a dict (serialised as JSON), a raw string,
    or a list of strings returned on successive calls. Translation requests get
    `vlm.translation`, the zero-shot text reading gets `vlm.text_reply`."""
    class Stub:
        reply = {"scene": "a road", "is_civic_scene": True, "image_quality": "good", "problems": []}
        translation = None
        text_reply = None      # dict/str for the zero-shot text reading; None = unparseable
        calls = 0

    stub = Stub()

    def fake_generate(messages, max_new_tokens):
        content = messages[0]["content"]
        if not any(part.get("type") == "image" for part in content):
            prompt = content[0]["text"]
            if "Problem IDs" in prompt:
                r = stub.text_reply
                return "" if r is None else r if isinstance(r, str) else json.dumps(r)
            return stub.translation or ""
        stub.calls += 1
        r = stub.reply
        if isinstance(r, list):
            r = r[min(stub.calls - 1, len(r) - 1)]
        return r if isinstance(r, str) else json.dumps(r)

    monkeypatch.setattr(settings, "AI_VLM_ENABLED", True)
    monkeypatch.setattr(settings, "AI_LLM_TEXT_CLASSIFY", True)   # exercised by tests; off by default
    monkeypatch.setattr(vision, "load", lambda: (object(), object()))
    monkeypatch.setattr(vision, "_generate", fake_generate)
    return stub


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import backend.main as main
    monkeypatch.setattr(main, "warm_up_model", lambda: None)
    monkeypatch.chdir(tmp_path)
    os.makedirs("uploads", exist_ok=True)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def auth(client):
    import uuid
    r = client.post("/api/auth/register", json={"phone_or_email": f"{uuid.uuid4()}@test.local", "password": "pw12345"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
