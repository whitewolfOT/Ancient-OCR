"""Tests for /api/training-pairs endpoints."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Isolate training-pairs storage to tmp_path for each test."""
    import api.routes as routes_mod

    pairs_dir = tmp_path / "training_pairs"
    pairs_dir.mkdir()
    monkeypatch.setattr(routes_mod, "_PAIRS_DIR", pairs_dir)
    monkeypatch.setattr(routes_mod, "_MANIFEST", pairs_dir / "manifest.json")

    from api.server import app
    return TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tiny_png_b64() -> str:
    """Minimal 1×1 white PNG as base64."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
    iend = chunk(b"IEND", b"")
    return base64.b64encode(sig + ihdr + idat + iend).decode()


def _form(page_id="1.jpg", token_index=3, label="الخضرة"):
    return {
        "page_id": page_id,
        "token_index": token_index,
        "label": label,
        "patch_b64": _tiny_png_b64(),
        "original_bbox": json.dumps([1158, 46, 77, 18]),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_save_pair_creates_files(client, tmp_path):
    r = client.post("/api/training-pairs", data=_form())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "saved"
    assert body["pair_id"] == "1.jpg/patch_0003"
    assert body["total_pairs"] == 1

    pairs_dir = tmp_path / "training_pairs"
    assert (pairs_dir / "1.jpg" / "patch_0003.png").exists(), "PNG not created"
    assert (pairs_dir / "1.jpg" / "patch_0003.txt").exists(), "TXT not created"
    assert (pairs_dir / "1.jpg" / "patch_0003.txt").read_text(encoding="utf-8") == "الخضرة"


def test_save_pair_updates_manifest(client):
    client.post("/api/training-pairs", data=_form(token_index=0))
    client.post("/api/training-pairs", data=_form(token_index=1))

    r = client.get("/api/training-pairs")
    assert r.status_code == 200
    manifest = r.json()
    assert manifest["total"] == 2
    ids = {p["id"] for p in manifest["pairs"]}
    assert "1.jpg/patch_0000" in ids
    assert "1.jpg/patch_0001" in ids


def test_overwrite_existing_pair(client):
    client.post("/api/training-pairs", data=_form(token_index=5, label="قديم"))
    client.post("/api/training-pairs", data=_form(token_index=5, label="جديد"))

    manifest = client.get("/api/training-pairs").json()
    assert manifest["total"] == 1
    assert manifest["pairs"][0]["label"] == "جديد"


def test_get_manifest_returns_list(client):
    r = client.get("/api/training-pairs")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["pairs"], list)
    assert "total" in body


def test_count_endpoint(client):
    client.post("/api/training-pairs", data=_form(token_index=0))
    r = client.get("/api/training-pairs/count")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_export_returns_zip(client):
    client.post("/api/training-pairs", data=_form())
    r = client.get("/api/training-pairs/export")
    assert r.status_code == 200
    assert "zip" in r.headers["content-type"]

    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert "manifest.json" in zf.namelist()


def test_empty_label_rejected(client):
    form = _form()
    form["label"] = "   "
    assert client.post("/api/training-pairs", data=form).status_code == 422


def test_bad_bbox_rejected(client):
    form = _form()
    form["original_bbox"] = "not-json"
    assert client.post("/api/training-pairs", data=form).status_code == 422
