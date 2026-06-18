"""Tests for /api/contribute/* endpoints — public word-contribution workflow."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolate contribute-related storage to tmp_path for each test."""
    import api.routes as routes_mod

    images_dir = tmp_path / "test_images"
    crops_dir = tmp_path / "contribute_crops"
    contributions_dir = tmp_path / "contributions"
    images_dir.mkdir()

    monkeypatch.setattr(routes_mod, "_RESULTS_JSON", tmp_path / "results.json")
    monkeypatch.setattr(routes_mod, "_TEST_IMAGES_DIR", images_dir)
    monkeypatch.setattr(routes_mod, "_CONTRIBUTE_CROPS_DIR", crops_dir)
    monkeypatch.setattr(routes_mod, "_CONTRIBUTIONS_DIR", contributions_dir)
    monkeypatch.setattr(routes_mod, "_CONTRIBUTE_SEEDS_PATH", tmp_path / "contribute_seeds.json")
    monkeypatch.setattr(routes_mod, "_CONTRIBUTE_SESSIONS_PATH", tmp_path / "contribute_sessions.json")

    from api.server import app
    return TestClient(app), tmp_path, images_dir


def _write_page_image(images_dir, filename="1.jpg", w=400, h=200):
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(images_dir / filename), img)


def _write_results(tmp_path, tokens, filename="1.jpg"):
    payload = {"pages": [{"filename": filename, "tokens": tokens}], "total_pages": 1}
    (tmp_path / "results.json").write_text(json.dumps(payload, ensure_ascii=False))


def _write_seeds(tmp_path, seeds):
    (tmp_path / "contribute_seeds.json").write_text(json.dumps({"seeds": seeds}, ensure_ascii=False))


# ── next-word ────────────────────────────────────────────────────────────────

def test_next_word_returns_low_confidence_token(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [
        {"text": "كتاب", "confidence": 0.95, "bbox": [20, 20, 60, 30]},
        {"text": "قلم", "confidence": 0.3, "bbox": [100, 60, 40, 30]},
    ])

    r = c.get("/api/contribute/next-word")
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is False
    assert body["word_id"] == "1.jpg_1"
    assert body["ocr_guess"] == "قلم"
    assert body["context_before"] == "كتاب"
    assert body["image_b64"]
    assert body["queue_total"] == 1


def test_next_word_handles_zero_height_bbox(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [
        {"text": "قلم", "confidence": 0.1, "bbox": [100, 60, 40, 0]},
    ])

    r = c.get("/api/contribute/next-word")
    assert r.status_code == 200
    assert r.json()["image_b64"]


def test_next_word_done_when_queue_empty(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [{"text": "كتاب", "confidence": 0.99, "bbox": [20, 20, 60, 30]}])

    r = c.get("/api/contribute/next-word")
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["word_id"] is None


# ── submit ───────────────────────────────────────────────────────────────────

def test_submit_marks_word_done_and_advances_queue(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [
        {"text": "قلم", "confidence": 0.2, "bbox": [10, 10, 40, 30]},
        {"text": "كتاب", "confidence": 0.1, "bbox": [60, 10, 40, 30]},
    ])

    first = c.get("/api/contribute/next-word").json()
    with patch("main.get_ralm_oracle", return_value=MagicMock()):
        r = c.post("/api/contribute/submit", json={
            "word_id": first["word_id"], "transcription": "قلم",
            "skipped": False, "session_token": "tok-1",
        })
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["next_word_id"] is not None
    assert body["next_word_id"] != first["word_id"]
    assert body["contributor_words_submitted"] == 1

    second = c.get("/api/contribute/next-word").json()
    assert second["word_id"] == body["next_word_id"]


def test_submit_skip_marks_done_without_oracle_call(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [{"text": "قلم", "confidence": 0.1, "bbox": [10, 10, 40, 30]}])

    first = c.get("/api/contribute/next-word").json()
    r = c.post("/api/contribute/submit", json={
        "word_id": first["word_id"], "transcription": "", "skipped": True, "session_token": "tok-1",
    })
    assert r.status_code == 200
    assert r.json()["next_word_id"] is None

    stats = c.get("/api/contribute/stats").json()
    assert stats["total_contributed"] == 1
    assert stats["queue_remaining"] == 0


def test_submit_unknown_word_id_404(client):
    c, tmp_path, images_dir = client
    _write_results(tmp_path, [{"text": "قلم", "confidence": 0.1, "bbox": [10, 10, 40, 30]}])
    r = c.post("/api/contribute/submit", json={
        "word_id": "does_not_exist", "transcription": "x", "skipped": False, "session_token": "tok-1",
    })
    assert r.status_code == 404


def test_seed_word_grades_correctness_without_revealing_it(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [
        {"text": "كتاب", "confidence": 0.99, "bbox": [10, 10, 40, 30]},
        {"text": "قلم", "confidence": 0.1, "bbox": [60, 10, 40, 30]},
    ])
    _write_seeds(tmp_path, [{"word_id": "1.jpg_0", "correct_text": "كتاب"}])

    # Seed word is interleaved into the queue alongside the one abstain word.
    r1 = c.post("/api/contribute/submit", json={
        "word_id": "1.jpg_0", "transcription": "كتاب", "skipped": False, "session_token": "tok-1",
    })
    assert r1.status_code == 200
    # Correctness of seed grading is internal — response never echoes "correct".
    assert "correct" not in r1.json()

    r2 = c.get("/api/contribute/next-word?session_token=tok-1").json()
    assert r2["contributor_accuracy"] == 1.0


# ── stats ────────────────────────────────────────────────────────────────────

def test_stats_reflects_top_contributors(client):
    c, tmp_path, images_dir = client
    _write_page_image(images_dir)
    _write_results(tmp_path, [{"text": "قلم", "confidence": 0.1, "bbox": [10, 10, 40, 30]}])

    first = c.get("/api/contribute/next-word").json()
    with patch("main.get_ralm_oracle", return_value=MagicMock()):
        c.post("/api/contribute/submit", json={
            "word_id": first["word_id"], "transcription": "قلم", "skipped": False, "session_token": "tok-x",
        })
    stats = c.get("/api/contribute/stats").json()
    assert stats["total_contributed"] == 1
    assert stats["queue_remaining"] == 0
    assert any(t["session_token"] == "tok-x" for t in stats["top_contributors"])
