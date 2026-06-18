"""Tests for /api/import/* endpoints — manuscript import workflow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolate import + lines storage to tmp_path for each test."""
    import api.routes as routes_mod

    import_dir = tmp_path / "import"
    lines_dir = tmp_path / "lines"
    import_dir.mkdir()
    lines_dir.mkdir()
    monkeypatch.setattr(routes_mod, "_IMPORT_DIR", import_dir)
    monkeypatch.setattr(routes_mod, "_LINES_DIR", lines_dir)

    from api.server import app
    return TestClient(app)


def _sample_jpg_bytes(w=400, h=200) -> bytes:
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (0, 0, 0), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def _upload_session(client) -> dict:
    files = {"files": ("page.jpg", _sample_jpg_bytes(), "image/jpeg")}
    r = client.post("/api/import/upload-images", files=files)
    assert r.status_code == 200
    return r.json()


# ── upload-images ───────────────────────────────────────────────────────────

def test_upload_images_creates_session(client):
    data = _upload_session(client)
    assert data["page_count"] == 1
    assert data["page_ids"] == ["page_0001"]
    assert data["session_id"]


def test_upload_images_appends_to_existing_session(client):
    first = _upload_session(client)
    session_id = first["session_id"]
    files = {"files": ("page2.jpg", _sample_jpg_bytes(), "image/jpeg")}
    r = client.post("/api/import/upload-images", data={"session_id": session_id}, files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == session_id
    assert data["page_count"] == 2
    assert data["page_ids"] == ["page_0002"]


def test_list_sessions(client):
    data = _upload_session(client)
    r = client.get("/api/import/sessions")
    assert r.status_code == 200
    sessions = r.json()["sessions"]
    assert any(s["session_id"] == data["session_id"] and s["page_count"] == 1 for s in sessions)


# ── page-image ───────────────────────────────────────────────────────────────

def test_page_image_preprocessed_differs_from_raw(client):
    data = _upload_session(client)
    session_id, page_id = data["session_id"], data["page_ids"][0]

    raw = client.get(f"/api/import/page-image/{session_id}/{page_id}", params={"preprocessed": False})
    assert raw.status_code == 200

    pre = client.get(
        f"/api/import/page-image/{session_id}/{page_id}",
        params={"preprocessed": True, "profile": "low_contrast"},
    )
    assert pre.status_code == 200
    assert raw.content != pre.content


def test_page_image_404_for_missing_page(client):
    data = _upload_session(client)
    r = client.get(f"/api/import/page-image/{data['session_id']}/page_9999")
    assert r.status_code == 404


# ── segment ──────────────────────────────────────────────────────────────────

def _fake_seg_lines():
    return [
        {"id": "line_000", "bbox": [10, 10, 300, 40], "baseline": [[10, 40], [310, 40]], "boundary": [[10, 10], [310, 10], [310, 50], [10, 50]]},
        {"id": "line_001", "bbox": [10, 60, 300, 40], "baseline": [[10, 90], [310, 90]], "boundary": [[10, 60], [310, 60], [310, 100], [10, 100]]},
    ]


def test_segment_returns_lines(client):
    data = _upload_session(client)
    session_id, page_id = data["session_id"], data["page_ids"][0]

    with patch("api.routes._run_blla_segment", return_value=_fake_seg_lines()):
        r = client.post(f"/api/import/segment/{session_id}/{page_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["line_count"] == 2
    assert len(body["lines"]) == 2
    assert body["lines"][0]["bbox"] == [10, 10, 300, 40]

    # saved segment result retrievable via GET
    r2 = client.get(f"/api/import/segments/{session_id}/{page_id}")
    assert r2.status_code == 200
    assert r2.json()["line_count"] == 2


def test_get_segments_404_when_not_segmented(client):
    data = _upload_session(client)
    r = client.get(f"/api/import/segments/{data['session_id']}/{data['page_ids'][0]}")
    assert r.status_code == 404


# ── accept-lines ─────────────────────────────────────────────────────────────

def _mock_ocr_result():
    from ocr_engine.schema import OCRResult, WordToken
    words = [
        WordToken(text="كتاب", confidence=0.9, bbox=(20, 15, 60, 20), page_index=0, source="kraken"),
        WordToken(text="قلم", confidence=0.8, bbox=(100, 65, 40, 20), page_index=0, source="kraken"),
    ]
    return OCRResult(text="كتاب\nقلم", words=words, confidence=0.85, page_index=0, source="kraken")


def _accept_body():
    return {
        "lines": [
            {"id": "line_000", "bbox": [10, 10, 300, 40], "baseline": [[10, 40], [310, 40]], "boundary": [[10, 10], [310, 10], [310, 50], [10, 50]]},
            {"id": "line_001", "bbox": [10, 60, 300, 40], "baseline": [[10, 90], [310, 90]], "boundary": [[10, 60], [310, 60], [310, 100], [10, 100]]},
        ],
        "profile_name": "default",
    }


def test_accept_lines_creates_line_crops(client, tmp_path):
    data = _upload_session(client)
    session_id, page_id = data["session_id"], data["page_ids"][0]

    with patch("ocr_engine.kraken_backend.KrakenBackend.process_image", return_value=_mock_ocr_result()):
        r = client.post(f"/api/import/accept-lines/{session_id}/{page_id}", json=_accept_body())
    assert r.status_code == 200
    body = r.json()
    assert body["saved_lines"] == 2
    assert body["page_id"] == page_id

    import api.routes as routes_mod
    out_dir = routes_mod._LINES_DIR / page_id
    assert (out_dir / "line_000.png").exists()
    assert (out_dir / "line_001.png").exists()
    assert (out_dir / "lines.json").exists()


def test_accept_lines_format_matches_existing(client):
    """lines.json structure must match scripts/generate_line_crops.py output."""
    data = _upload_session(client)
    session_id, page_id = data["session_id"], data["page_ids"][0]

    with patch("ocr_engine.kraken_backend.KrakenBackend.process_image", return_value=_mock_ocr_result()):
        r = client.post(f"/api/import/accept-lines/{session_id}/{page_id}", json=_accept_body())
    assert r.status_code == 200

    import api.routes as routes_mod
    saved = json.loads((routes_mod._LINES_DIR / page_id / "lines.json").read_text())

    reference = json.loads(Path("data/lines/1.jpg/lines.json").read_text())

    assert set(saved.keys()) == set(reference.keys())
    assert set(saved["lines"][0].keys()) == set(reference["lines"][0].keys())

    # ocr_text was filled in from the (mocked) OCR pass
    texts = [ln["ocr_text"] for ln in saved["lines"]]
    assert "كتاب" in texts[0] or "كتاب" in texts[1]
