"""Tests for Step 6 API routes: profiles, preview, suggest_profile, correction."""
from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 64, height: int = 64,
                     brightness: int = 180) -> bytes:
    from PIL import Image
    # Checkerboard with large blocks — survives JPEG quantization, keeps std > 30
    lo = max(0, brightness - 80)
    hi = min(255, brightness + 60)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    block = 16
    for r in range(0, height, block):
        for c in range(0, width, block):
            val = hi if ((r // block + c // block) % 2 == 0) else lo
            arr[r:r+block, c:c+block] = val
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_dark_jpeg_bytes() -> bytes:
    return _make_jpeg_bytes(brightness=50)


@pytest.fixture()
def tmp_profiles_yaml(tmp_path):
    """Copy real profiles.yaml to tmp_path so tests can mutate it safely."""
    src = Path("config/profiles.yaml")
    dst = tmp_path / "profiles.yaml"
    shutil.copy(src, dst)
    return dst


@pytest.fixture()
def client(tmp_profiles_yaml):
    from fastapi.testclient import TestClient
    from api.server import app
    import api.routes as routes_mod

    # Reset the global singleton so it picks up tmp_profiles_yaml
    routes_mod._profile_mgr = None

    with patch("api.routes._get_profile_mgr") as mock_mgr_fn:
        from ocr_engine.profile_loader import ProfileManager
        real_mgr = ProfileManager(tmp_profiles_yaml)
        mock_mgr_fn.return_value = real_mgr
        with TestClient(app) as c:
            yield c, real_mgr


# ---------------------------------------------------------------------------
# GET /api/profiles
# ---------------------------------------------------------------------------

def test_list_profiles_returns_names(client):
    c, mgr = client
    resp = c.get("/api/profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert "profiles" in data
    assert "default" in data["profiles"]


# ---------------------------------------------------------------------------
# GET /api/profiles/{name}
# ---------------------------------------------------------------------------

def test_get_profile_returns_structure(client):
    c, mgr = client
    resp = c.get("/api/profiles/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "default"
    assert "preprocessing" in data
    assert "binarizer" in data
    assert "n_best" in data


def test_get_profile_unknown_falls_back_to_default(client):
    c, mgr = client
    resp = c.get("/api/profiles/nonexistent_xyz")
    assert resp.status_code == 200
    data = resp.json()
    # ProfileManager.get() falls back to "default" for unknown names
    assert data["name"] == "default"


# ---------------------------------------------------------------------------
# PUT /api/profiles/{name}
# ---------------------------------------------------------------------------

def test_update_profile_persists(client):
    c, mgr = client
    body = {
        "name": "test_custom",
        "description": "test profile",
        "binarizer": "sauvola",
        "seg_model": "zenodo:14295555",
        "rec_model": "agapet",
        "rec_model_secondary": None,
        "n_best": 2,
        "rtl": True,
        "device": "cpu",
        "preprocessing": {
            "brightness": 10,
            "contrast": 1.2,
            "gamma": 1.0,
            "saturation": 1.0,
            "stroke_normalization": {"enabled": False, "target_width": 2},
            "denoise_strength": 0,
            "sharpen": 0.0,
        },
    }
    resp = c.put("/api/profiles/test_custom", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "saved"
    assert data["name"] == "test_custom"
    # Verify it's retrievable
    check = c.get("/api/profiles/test_custom")
    assert check.status_code == 200
    assert check.json()["binarizer"] == "sauvola"


# ---------------------------------------------------------------------------
# DELETE /api/profiles/{name}
# ---------------------------------------------------------------------------

def test_delete_default_returns_403(client):
    c, mgr = client
    resp = c.delete("/api/profiles/default")
    assert resp.status_code == 403


def test_delete_custom_profile(client):
    c, mgr = client
    # First create a profile
    body = {
        "description": "", "binarizer": "otsu", "seg_model": "zenodo:14295555",
        "rec_model": "agapet", "rec_model_secondary": None, "n_best": 3,
        "rtl": True, "device": "cpu",
        "preprocessing": {
            "brightness": 0, "contrast": 1.0, "gamma": 1.0, "saturation": 1.0,
            "stroke_normalization": {"enabled": False, "target_width": 2},
            "denoise_strength": 0, "sharpen": 0.0,
        },
    }
    c.put("/api/profiles/to_delete", json=body)
    resp = c.delete("/api/profiles/to_delete")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# POST /api/preview
# ---------------------------------------------------------------------------

def test_preview_returns_b64_image(client):
    c, mgr = client
    img_bytes = _make_jpeg_bytes()
    resp = c.post(
        "/api/preview",
        data={"profile_name": "default", "brightness": "0", "contrast": "1.0",
              "gamma": "1.0", "saturation": "1.0", "stroke_enabled": "false",
              "stroke_width": "2", "denoise_strength": "0", "sharpen": "0.0"},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "processed_image_b64" in data
    assert len(data["processed_image_b64"]) > 0
    assert data["profile_used"] == "default"


def test_preview_invalid_image_returns_400(client):
    c, mgr = client
    resp = c.post(
        "/api/preview",
        data={"profile_name": "default"},
        files={"file": ("bad.jpg", b"not an image", "image/jpeg")},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/suggest_profile
# ---------------------------------------------------------------------------

def test_suggest_profile_bright_image_returns_default(client):
    c, mgr = client
    bright = _make_jpeg_bytes(brightness=200)
    resp = c.post(
        "/api/suggest_profile",
        files={"file": ("bright.jpg", bright, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["suggested_profile"] == "default"


def test_suggest_profile_dark_image_returns_low_contrast(client):
    c, mgr = client
    dark = _make_dark_jpeg_bytes()
    resp = c.post(
        "/api/suggest_profile",
        files={"file": ("dark.jpg", dark, "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["suggested_profile"] == "low_contrast"


# ---------------------------------------------------------------------------
# POST /api/correction
# ---------------------------------------------------------------------------

def test_correction_stores_entry(client, tmp_path):
    c, mgr = client
    with patch("api.routes.submit") if False else patch("training.feedback_store.submit",
                                                        return_value="test-uuid-001"):
        resp = c.post(
            "/api/correction",
            json={
                "token_id": "doc_p0_t5",
                "original_text": "كتب",
                "corrected_text": "كتاب",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "stored"
    assert "entry_id" in data
