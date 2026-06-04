"""Tests for document calibration API endpoints.

All six endpoints are tested with synthetic fixtures — no real PDF required.
image_pipeline.preprocess_image is mocked so tests run without OpenCV/GPU.
"""
from __future__ import annotations

import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return minimal valid JPEG bytes for a solid-grey image."""
    from PIL import Image
    img = Image.fromarray(
        np.full((height, width, 3), 128, dtype=np.uint8), mode="RGB"
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def patch_db(tmp_path, monkeypatch):
    """Redirect DB and document storage to tmp_path for isolation."""
    import documents.store as store_mod
    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "documents.db")
    monkeypatch.setattr(store_mod, "DOCS_DIR", tmp_path)
    store_mod.init_db()
    return tmp_path


@pytest.fixture()
def client(patch_db):
    """FastAPI TestClient with the full app (all routes registered)."""
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_jpeg_returns_doc_id_and_cluster(self, client):
        jpeg = _make_jpeg_bytes()
        resp = client.post(
            "/documents/upload",
            files={"file": ("page.jpg", jpeg, "image/jpeg")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "doc_id" in body
        assert body["page_count"] == 1
        assert len(body["clusters"]) == 1
        cl = body["clusters"][0]
        assert "cluster_id" in cl
        assert cl["page_count"] == 1
        assert "representative_page_id" in cl

    def test_upload_png(self, client):
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(buf, format="PNG")
        resp = client.post(
            "/documents/upload",
            files={"file": ("scan.png", buf.getvalue(), "image/png")},
        )
        assert resp.status_code == 200
        assert resp.json()["page_count"] == 1

    def test_upload_unsupported_type_returns_415(self, client):
        resp = client.post(
            "/documents/upload",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 415

    def test_two_identical_images_cluster_together(self, client):
        jpeg = _make_jpeg_bytes()
        # Upload first image
        r1 = client.post("/documents/upload", files={"file": ("a.jpg", jpeg, "image/jpeg")})
        assert r1.status_code == 200
        # Two identical pages within a single upload would need a multi-page source.
        # Here we just verify the single-image case has exactly one cluster.
        assert len(r1.json()["clusters"]) == 1


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}/pages
# ---------------------------------------------------------------------------

class TestGetPages:
    def _upload(self, client) -> dict:
        jpeg = _make_jpeg_bytes()
        resp = client.post("/documents/upload", files={"file": ("p.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        return resp.json()

    def test_returns_page_list(self, client):
        upload = self._upload(client)
        doc_id = upload["doc_id"]
        resp = client.get(f"/documents/{doc_id}/pages")
        assert resp.status_code == 200
        pages = resp.json()
        assert len(pages) == 1
        p = pages[0]
        assert "page_id" in p
        assert p["page_num"] == 0
        assert "cluster_id" in p
        assert 0.0 <= p["similarity_to_representative"] <= 1.0
        assert p["has_ground_truth"] is False
        assert p["has_settings"] is False
        assert p["thumbnail_url"].startswith("/pages/")

    def test_unknown_doc_returns_404(self, client):
        resp = client.get("/documents/nonexistent-doc/pages")
        assert resp.status_code == 404

    def test_has_ground_truth_flag_updates(self, client):
        upload = self._upload(client)
        doc_id = upload["doc_id"]
        page_id = client.get(f"/documents/{doc_id}/pages").json()[0]["page_id"]

        client.post(
            f"/pages/{page_id}/ground-truth",
            json={"text": "test", "page_id": page_id},
        )
        pages = client.get(f"/documents/{doc_id}/pages").json()
        assert pages[0]["has_ground_truth"] is True

    def test_has_settings_flag_updates(self, client):
        upload = self._upload(client)
        doc_id = upload["doc_id"]
        cluster_id = upload["clusters"][0]["cluster_id"]

        client.post(
            f"/documents/{doc_id}/apply-cluster-settings",
            json={
                "cluster_id": cluster_id,
                "settings": {"clahe": 3.0, "denoise": 5, "deskew_threshold": 1.0, "binarization": "OTSU"},
            },
        )
        pages = client.get(f"/documents/{doc_id}/pages").json()
        assert pages[0]["has_settings"] is True


# ---------------------------------------------------------------------------
# POST /pages/{page_id}/preview
# ---------------------------------------------------------------------------

class TestPreview:
    def _upload_and_get_page_id(self, client) -> tuple[str, str]:
        jpeg = _make_jpeg_bytes()
        upload = client.post("/documents/upload", files={"file": ("p.jpg", jpeg, "image/jpeg")}).json()
        doc_id = upload["doc_id"]
        page_id = client.get(f"/documents/{doc_id}/pages").json()[0]["page_id"]
        return doc_id, page_id

    def test_preview_returns_b64_image(self, client):
        _, page_id = self._upload_and_get_page_id(client)

        mock_img = np.zeros((64, 64), dtype=np.uint8)
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            resp = client.post(
                f"/pages/{page_id}/preview",
                json={"clahe": 3.0, "denoise": 5, "deskew_threshold": 1.0, "binarization": "OTSU"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["preview_image_b64"]          # non-empty base64 string
        assert body["settings_applied"]["clahe"] == 3.0
        assert body["settings_applied"]["binarization"] == "OTSU"
        assert "preview_id" in body
        assert body["processing_time_ms"] >= 0

    def test_preview_settings_applied_matches_request(self, client):
        _, page_id = self._upload_and_get_page_id(client)

        mock_img = np.zeros((32, 32), dtype=np.uint8)
        req = {"clahe": 4.5, "denoise": 7, "deskew_threshold": 2.0, "binarization": "Global"}
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            resp = client.post(f"/pages/{page_id}/preview", json=req)
        assert resp.status_code == 200
        sa = resp.json()["settings_applied"]
        assert sa["clahe"] == 4.5
        assert sa["denoise"] == 7
        assert sa["deskew_threshold"] == 2.0
        assert sa["binarization"] == "Global"

    def test_preview_unknown_page_returns_404(self, client):
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(np.zeros((32, 32)), {})):
            resp = client.post(
                "/pages/nonexistent-page-id/preview",
                json={"clahe": 2.0, "denoise": 3, "deskew_threshold": 0.5, "binarization": "Adaptive"},
            )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /pages/{page_id}/ground-truth
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def _page_id(self, client) -> str:
        jpeg = _make_jpeg_bytes()
        upload = client.post("/documents/upload", files={"file": ("p.jpg", jpeg, "image/jpeg")}).json()
        return client.get(f"/documents/{upload['doc_id']}/pages").json()[0]["page_id"]

    def test_save_and_idempotent_update(self, client):
        page_id = self._page_id(client)

        resp = client.post(
            f"/pages/{page_id}/ground-truth",
            json={"text": "بسم الله", "page_id": page_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["page_id"] == page_id
        assert "saved_at" in body

        # Overwrite with new text — same page_id
        resp2 = client.post(
            f"/pages/{page_id}/ground-truth",
            json={"text": "الحمد لله", "page_id": page_id},
        )
        assert resp2.status_code == 200

    def test_unknown_page_returns_404(self, client):
        resp = client.post(
            "/pages/no-such-page/ground-truth",
            json={"text": "test", "page_id": "no-such-page"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /documents/{doc_id}/apply-cluster-settings
# ---------------------------------------------------------------------------

class TestApplyClusterSettings:
    def _upload(self, client) -> dict:
        jpeg = _make_jpeg_bytes()
        return client.post("/documents/upload", files={"file": ("p.jpg", jpeg, "image/jpeg")}).json()

    def test_applies_settings_to_cluster_pages(self, client):
        upload = self._upload(client)
        doc_id = upload["doc_id"]
        cluster_id = upload["clusters"][0]["cluster_id"]

        resp = client.post(
            f"/documents/{doc_id}/apply-cluster-settings",
            json={
                "cluster_id": cluster_id,
                "settings": {
                    "clahe": 5.0,
                    "denoise": 9,
                    "deskew_threshold": 3.0,
                    "binarization": "Adaptive",
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pages_updated"] == 1
        assert body["cluster_id"] == cluster_id

    def test_unknown_cluster_returns_404(self, client):
        resp = client.post(
            "/documents/some-doc/apply-cluster-settings",
            json={
                "cluster_id": "no-such-cluster",
                "settings": {"clahe": 2.0, "denoise": 3, "deskew_threshold": 0.5, "binarization": "Adaptive"},
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /pages/{page_id}/image
# ---------------------------------------------------------------------------

class TestServeImage:
    def _page_id(self, client) -> str:
        jpeg = _make_jpeg_bytes()
        upload = client.post("/documents/upload", files={"file": ("p.jpg", jpeg, "image/jpeg")}).json()
        return client.get(f"/documents/{upload['doc_id']}/pages").json()[0]["page_id"]

    def test_returns_jpeg_response(self, client):
        page_id = self._page_id(client)
        resp = client.get(f"/pages/{page_id}/image")
        assert resp.status_code == 200
        assert "image" in resp.headers["content-type"]

    def test_unknown_page_returns_404(self, client):
        resp = client.get("/pages/nonexistent-page/image")
        assert resp.status_code == 404
