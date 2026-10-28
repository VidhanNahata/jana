"""CLIP option: the trained head (training/issues/clip_head.py) and the backend must agree exactly,
and the pipeline must survive a missing / broken encoder. The CLIP weights themselves are not needed:
the embedding function is replaced by a stub."""

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")
from backend.config import settings                      # noqa: E402
from backend.services import issue_models                # noqa: E402
from backend.tests.conftest import ROOT, jpeg            # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "training", "issues"))
import clip_head                                         # noqa: E402
from common import classifier_class_names                # noqa: E402

DIM = 32


def clusters(n_per_class, n_classes, rng, noise=0.35):
    centers = clip_head.l2norm(rng.normal(size=(n_classes, DIM)))
    X = np.concatenate([centers[c] + noise * rng.normal(size=(n_per_class, DIM)) / np.sqrt(DIM) * 3
                        for c in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    return clip_head.l2norm(X).astype(np.float32), y, centers


@pytest.fixture(autouse=True)
def clean_cache():
    issue_models.reset_cache()
    yield
    issue_models.reset_cache()


def test_head_learns_from_few_photos_per_class():
    rng = np.random.default_rng(0)
    n = len(classifier_class_names())
    Xtr, ytr, centers = clusters(12, n, rng)             # 12 "photos" per class
    Xte = clip_head.l2norm(centers.repeat(20, 0) + 0.35 * rng.normal(size=(n * 20, DIM)) / np.sqrt(DIM) * 3)
    yte = np.repeat(np.arange(n), 20)
    C, _ = clip_head.choose_C(Xtr, ytr, *clusters(6, n, rng)[:2])
    W, b = clip_head.fit_head(Xtr, ytr, C)
    assert (clip_head.head_probs(Xte, W, b).argmax(1) == yte).mean() > 0.9


def test_class_missing_from_training_can_never_win():
    rng = np.random.default_rng(1)
    X, y, _ = clusters(10, 4, rng)
    keep = y != 2                                         # class 2 has no training photos
    W, b = clip_head.fit_head(X[keep], y[keep], 1.0)
    assert b.shape[0] == 4 and b[2] == -20.0
    assert (clip_head.head_probs(X, W, b).argmax(1) != 2).all()


def install_head(tmp_path, monkeypatch, W, b, thresholds):
    names = classifier_class_names()
    cfg = json.load(open(os.path.join(ROOT, "training", "issues", "classes.json")))["classifier"]
    np.savez(tmp_path / "head.npz", W=W, b=b)
    json.dump({"classes": names, "background": cfg["background"], "subcategory": cfg["classes"],
               "thresholds": thresholds, "embedding_dim": DIM}, open(tmp_path / "head.json", "w"))
    (tmp_path / "clip").mkdir()
    (tmp_path / "clip" / "config.json").write_text("{}")
    monkeypatch.setattr(settings, "ISSUE_CLIP_HEAD_PATH", str(tmp_path / "head.npz"))
    monkeypatch.setattr(settings, "ISSUE_CLIP_MODEL_PATH", str(tmp_path / "clip"))


def test_backend_matches_training_head_exactly(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    names = classifier_class_names()
    X, y, centers = clusters(15, len(names), rng)
    W, b = clip_head.fit_head(X, y, 3.0)
    thr = {n: 0.5 for n in names[:-1]}
    install_head(tmp_path, monkeypatch, W, b, thr)
    garbage = names.index("garbage")
    emb = centers[garbage].astype(np.float32)
    monkeypatch.setattr(issue_models, "_embed_image", lambda image: emb)

    expected = clip_head.head_probs(emb[None], W, b)[0]   # what training computed
    ev = issue_models.evidence([jpeg()])
    assert [e["subcategory_id"] for e in ev] == ["WASTE.UNCOLLECTED"]
    assert ev[0]["source"] == "issue_classifier"
    assert ev[0]["confidence"] == pytest.approx(min(float(expected[garbage]), settings.ISSUE_MODEL_MAX_CONFIDENCE), abs=1e-3)
    assert issue_models.status()["clip"]["installed"] is True


def test_background_embedding_raises_no_issue(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    names = classifier_class_names()
    X, y, centers = clusters(15, len(names), rng)
    W, b = clip_head.fit_head(X, y, 3.0)
    install_head(tmp_path, monkeypatch, W, b, {n: 0.5 for n in names[:-1]})
    monkeypatch.setattr(issue_models, "_embed_image", lambda image: centers[-1].astype(np.float32))
    assert issue_models.evidence([jpeg()]) == []


def test_missing_clip_weights_means_skipped(tmp_path, monkeypatch):
    names = classifier_class_names()
    install_head(tmp_path, monkeypatch, np.zeros((len(names), DIM), np.float32), np.zeros(len(names), np.float32),
                 {n: 0.5 for n in names[:-1]})
    monkeypatch.setattr(settings, "ISSUE_CLIP_MODEL_PATH", str(tmp_path / "no_such_dir"))
    assert issue_models.evidence([jpeg()]) == []
    assert issue_models.status()["clip"]["installed"] is False


def test_broken_encoder_never_breaks_a_complaint(tmp_path, monkeypatch):
    names = classifier_class_names()
    install_head(tmp_path, monkeypatch, np.zeros((len(names), DIM), np.float32), np.zeros(len(names), np.float32),
                 {n: 0.5 for n in names[:-1]})

    def boom(image):
        raise RuntimeError("no torch")
    monkeypatch.setattr(issue_models, "_embed_image", boom)
    assert issue_models.evidence([jpeg()]) == []
