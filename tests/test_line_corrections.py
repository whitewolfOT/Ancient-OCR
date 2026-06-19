"""Tests for line correction API endpoints (Step 1 of Sprint 1)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolate line data storage to tmp_path for each test."""
    import api.routes as routes_mod

    lines_dir = tmp_path / "lines"
    corr_dir  = tmp_path / "corrections"
    lines_dir.mkdir()
    corr_dir.mkdir()

    monkeypatch.setattr(routes_mod, "_LINES_DIR",       lines_dir)
    monkeypatch.setattr(routes_mod, "_CORRECTIONS_DIR", corr_dir)

    from api.server import app
    return TestClient(app)


def _make_lines_json(lines_dir: Path, page_id: str, n: int = 3) -> None:
    """Write a minimal lines.json for testing."""
    page_dir = lines_dir / page_id
    page_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "index": i,
            "image_path": str(page_dir / f"line_{i:03d}.png"),
            "ocr_text": f"نص السطر {i}",
            "baseline": [],
            "bbox": [10, i * 50, 200, 40],
            "confidence": round(0.9 - i * 0.05, 3),
        }
        for i in range(n)
    ]
    (page_dir / "lines.json").write_text(
        json.dumps({"page": page_id, "original_size": [1300, 1800], "lines": lines}),
        encoding="utf-8",
    )


def _make_line_image(lines_dir: Path, page_id: str, index: int) -> None:
    """Write a tiny 1×1 white PNG for the line image."""
    import struct, zlib
    sig = b"\x89PNG\r\n\x1a\n"
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )
    png = sig + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    png += chunk(b"IEND", b"")
    (lines_dir / page_id / f"line_{index:03d}.png").write_bytes(png)


# ── GET /api/lines/{page_id} ──────────────────────────────────────────────────

def test_get_lines_returns_structure(client, tmp_path):
    lines_dir = tmp_path / "lines"
    _make_lines_json(lines_dir, "1.jpg", n=3)

    r = client.get("/api/lines/1.jpg")
    assert r.status_code == 200
    body = r.json()
    assert body["page"] == "1.jpg"
    assert len(body["lines"]) == 3
    assert body["lines"][0]["index"] == 0
    assert body["lines"][0]["status"] == "pending"
    assert "ocr_text" in body["lines"][0]
    assert "bbox" in body["lines"][0]
    assert "confidence" in body["lines"][0]


def test_get_lines_missing_page_404(client):
    r = client.get("/api/lines/nonexistent.jpg")
    assert r.status_code == 404


def test_get_lines_merges_corrections(client, tmp_path):
    lines_dir = tmp_path / "lines"
    _make_lines_json(lines_dir, "1.jpg", n=3)

    # Save a correction first
    client.post(
        "/api/lines/1.jpg/1/correction",
        json={"corrected_text": "نص مصحح", "status": "corrected"},
    )

    r = client.get("/api/lines/1.jpg")
    body = r.json()
    line1 = next(l for l in body["lines"] if l["index"] == 1)
    assert line1["corrected_text"] == "نص مصحح"
    assert line1["status"] == "corrected"
    # Other lines remain pending
    assert body["lines"][0]["status"] == "pending"


# ── POST /api/lines/{page_id}/{line_index}/correction ─────────────────────────

def test_save_correction_persists(client, tmp_path):
    lines_dir = tmp_path / "lines"
    corr_dir  = tmp_path / "corrections"
    _make_lines_json(lines_dir, "2.jpg", n=4)

    r = client.post(
        "/api/lines/2.jpg/0/correction",
        json={"corrected_text": "الكلمة الصحيحة", "status": "corrected"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["saved"] is True
    assert body["total_corrected"] == 1
    assert body["total_lines"] == 4

    # Verify .gt.txt on disk
    gt = corr_dir / "2.jpg" / "line_000.txt"
    assert gt.exists()
    assert gt.read_text(encoding="utf-8") == "الكلمة الصحيحة"

    # Verify corrections.json
    corr_json = corr_dir / "2.jpg" / "corrections.json"
    data = json.loads(corr_json.read_text())
    assert data["0"]["status"] == "corrected"
    assert data["0"]["corrected_text"] == "الكلمة الصحيحة"


def test_save_skipped_status(client, tmp_path):
    _make_lines_json(tmp_path / "lines", "1.jpg", n=2)

    r = client.post(
        "/api/lines/1.jpg/0/correction",
        json={"corrected_text": "", "status": "skipped"},
    )
    assert r.status_code == 200
    assert r.json()["total_corrected"] == 0  # skipped doesn't count as corrected


def test_save_correction_invalid_status(client, tmp_path):
    _make_lines_json(tmp_path / "lines", "1.jpg", n=2)
    r = client.post(
        "/api/lines/1.jpg/0/correction",
        json={"corrected_text": "text", "status": "invalid"},
    )
    assert r.status_code == 422


def test_save_correction_missing_page_404(client):
    r = client.post(
        "/api/lines/ghost.jpg/0/correction",
        json={"corrected_text": "text", "status": "corrected"},
    )
    assert r.status_code == 404


def test_correction_count_accumulates(client, tmp_path):
    _make_lines_json(tmp_path / "lines", "3.jpg", n=5)

    client.post("/api/lines/3.jpg/0/correction",
                json={"corrected_text": "a", "status": "corrected"})
    r = client.post("/api/lines/3.jpg/2/correction",
                    json={"corrected_text": "b", "status": "corrected"})
    assert r.json()["total_corrected"] == 2


# ── GET /api/corrections/export ───────────────────────────────────────────────

def test_export_zip_contains_pairs(client, tmp_path):
    lines_dir = tmp_path / "lines"
    _make_lines_json(lines_dir, "1.jpg", n=2)
    _make_line_image(lines_dir, "1.jpg", 0)

    # Save one correction
    client.post("/api/lines/1.jpg/0/correction",
                json={"corrected_text": "النص الصحيح", "status": "corrected"})

    r = client.get("/api/corrections/export")
    assert r.status_code == 200
    assert "zip" in r.headers["content-type"]

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        assert "manifest.txt" in names
        assert "README.txt" in names
        assert "1.jpg/line_000.gt.txt" in names
        gt = zf.read("1.jpg/line_000.gt.txt").decode("utf-8")
        assert gt == "النص الصحيح"


def test_export_empty_returns_zip(client):
    r = client.get("/api/corrections/export")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "manifest.txt" in zf.namelist()
        assert zf.read("manifest.txt") == b""


def test_export_skipped_not_included(client, tmp_path):
    _make_lines_json(tmp_path / "lines", "1.jpg", n=2)

    client.post("/api/lines/1.jpg/0/correction",
                json={"corrected_text": "", "status": "skipped"})

    r = client.get("/api/corrections/export")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "1.jpg/line_000.gt.txt" not in zf.namelist()


# ── POST /api/lines/{page_id}/replace-line ─────────────────────────────────

def _make_test_image(images_dir: Path, page_id: str, w: int = 400, h: int = 400) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(images_dir / page_id), img)


def _mock_ocr_result(text: str = "نص جديد"):
    from ocr_engine.schema import OCRResult, WordToken
    words = [WordToken(text=text, confidence=0.9, bbox=(0, 0, 50, 20), page_index=0, source="kraken")]
    return OCRResult(text=text, words=words, confidence=0.9, page_index=0, source="kraken")


@pytest.fixture()
def replace_line_setup(client, tmp_path, monkeypatch):
    """3 lines at y=0,50,100 (h=40 each); raw page image staged at a tmp dir."""
    import api.routes as routes_mod

    images_dir = tmp_path / "images"
    _make_test_image(images_dir, "1.jpg")
    monkeypatch.setattr(routes_mod, "_TEST_IMAGES_DIR", images_dir)

    lines_dir = tmp_path / "lines"
    _make_lines_json(lines_dir, "1.jpg", n=3)
    return client, lines_dir


def test_replace_line_creates_new_entry(replace_line_setup):
    client, lines_dir = replace_line_setup

    with patch("ocr_engine.kraken_backend.KrakenBackend.process_image", return_value=_mock_ocr_result()):
        r = client.post(
            "/api/lines/1.jpg/replace-line",
            json={"old_line_index": 1, "new_bbox": [10, 50, 200, 40], "replace_adjacent": False},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["new_line"]["ocr_text"] == "نص جديد"
    assert body["total_lines"] == 3  # one line replaced 1-for-1

    data = json.loads((lines_dir / "1.jpg" / "lines.json").read_text())
    indices = [ln["index"] for ln in data["lines"]]
    assert indices == [0, 1, 2]  # sequential, no gaps
    texts = [ln["ocr_text"] for ln in data["lines"]]
    assert "نص جديد" in texts
    assert "نص السطر 0" in texts  # untouched line survives
    assert "نص السطر 2" in texts


def test_replace_line_removes_overlapping(replace_line_setup):
    client, lines_dir = replace_line_setup

    # Covers y=50..140 -> fully overlaps both line 1 (y50-90) and line 2 (y100-140)
    with patch("ocr_engine.kraken_backend.KrakenBackend.process_image", return_value=_mock_ocr_result()):
        r = client.post(
            "/api/lines/1.jpg/replace-line",
            json={"old_line_index": 1, "new_bbox": [10, 50, 200, 90], "replace_adjacent": True},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total_lines"] == 2  # line 0 + the single new merged line

    data = json.loads((lines_dir / "1.jpg" / "lines.json").read_text())
    texts = [ln["ocr_text"] for ln in data["lines"]]
    assert texts.count("نص السطر 0") == 1
    assert "نص السطر 2" not in texts
    assert "نص السطر 1" not in texts
    indices = [ln["index"] for ln in data["lines"]]
    assert indices == [0, 1]


def test_replace_adjacent_false_keeps_others(replace_line_setup):
    client, lines_dir = replace_line_setup

    # Same overlapping new_bbox as above, but replace_adjacent=False this time
    with patch("ocr_engine.kraken_backend.KrakenBackend.process_image", return_value=_mock_ocr_result()):
        r = client.post(
            "/api/lines/1.jpg/replace-line",
            json={"old_line_index": 1, "new_bbox": [10, 50, 200, 90], "replace_adjacent": False},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total_lines"] == 3  # line 0 + line 2 (kept despite overlap) + new line

    data = json.loads((lines_dir / "1.jpg" / "lines.json").read_text())
    texts = [ln["ocr_text"] for ln in data["lines"]]
    assert "نص السطر 0" in texts
    assert "نص السطر 2" in texts  # not removed — replace_adjacent was False
    assert "نص السطر 1" not in texts  # old_line_index is always replaced
    indices = [ln["index"] for ln in data["lines"]]
    assert indices == [0, 1, 2]


def test_replace_line_missing_page_404(client):
    r = client.post(
        "/api/lines/ghost.jpg/replace-line",
        json={"old_line_index": 0, "new_bbox": [0, 0, 100, 30], "replace_adjacent": False},
    )
    assert r.status_code == 404
