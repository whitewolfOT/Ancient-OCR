"""Tests for POST /pages/{page_id}/ocr and GET /pages/{page_id}/ocr endpoints."""
from __future__ import annotations

import io
import json
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers (shared with test_document_api)
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    from PIL import Image
    img = Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


_MOCK_ANNOTATED = {
    "mode": "annotated",
    "text": "بسم الله",
    "word_count": 2,
    "page_count": 1,
    "review_queue": {"total": 0, "by_decision": {}, "flagged_tokens": []},
    "tokens": [
        {
            "original": "بسم",
            "selected": "بسم",
            "confidence": 0.92,
            "decision": "accept",
            "reason_code": "ocr_confident",
            "sources": ["lanes"],
            "bbox": [10, 20, 30, 15],
        },
        {
            "original": "الله",
            "selected": "الله",
            "confidence": 0.55,
            "decision": "uncertain",
            "reason_code": "low_confidence",
            "sources": [],
            "bbox": [50, 20, 40, 15],
        },
    ],
}


@pytest.fixture()
def patch_db(tmp_path, monkeypatch):
    import documents.store as store_mod
    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "documents.db")
    monkeypatch.setattr(store_mod, "DOCS_DIR", tmp_path)
    store_mod.init_db()
    return tmp_path


@pytest.fixture()
def client(patch_db):
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app, raise_server_exceptions=True)


def _upload_and_get_page_id(client) -> tuple[str, str]:
    jpeg = _make_jpeg_bytes()
    upload = client.post("/documents/upload", files={"file": ("p.jpg", jpeg, "image/jpeg")}).json()
    doc_id = upload["doc_id"]
    page_id = client.get(f"/documents/{doc_id}/pages").json()[0]["page_id"]
    return doc_id, page_id


# ---------------------------------------------------------------------------
# POST /pages/{page_id}/ocr
# ---------------------------------------------------------------------------

class TestRunOCR:
    def test_run_ocr_saves_result_and_returns_decision_counts(self, client):
        _, page_id = _upload_and_get_page_id(client)

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            with patch("main.run_pipeline", return_value=_MOCK_ANNOTATED):
                resp = client.post(f"/pages/{page_id}/ocr")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["page_id"] == page_id
        assert body["word_count"] == 2
        assert body["decisions"]["accept"] == 1
        assert body["decisions"]["uncertain"] == 1
        assert body["decisions"]["accept_with_note"] == 0
        assert body["decisions"]["review_required"] == 0
        assert "processed_at" in body

    def test_run_ocr_tokens_have_correct_format(self, client):
        _, page_id = _upload_and_get_page_id(client)

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            with patch("main.run_pipeline", return_value=_MOCK_ANNOTATED):
                resp = client.post(f"/pages/{page_id}/ocr")

        tokens = resp.json()["tokens"]
        assert len(tokens) == 2
        t0 = tokens[0]
        assert t0["text"] == "بسم"
        assert t0["decision"] == "accept"
        assert isinstance(t0["confidence"], float)
        # bbox converted from (x, y, w, h) → [x1, y1, x2, y2]
        assert t0["bbox"] == [10, 20, 40, 35]   # x2=10+30, y2=20+15

    def test_run_ocr_updates_page_status_to_ocr_done(self, client):
        doc_id, page_id = _upload_and_get_page_id(client)

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            with patch("main.run_pipeline", return_value=_MOCK_ANNOTATED):
                client.post(f"/pages/{page_id}/ocr")

        pages = client.get(f"/documents/{doc_id}/pages").json()
        assert pages[0]["status"] == "ocr_done"

    def test_run_ocr_result_is_persisted(self, client):
        """POST then GET returns the same data."""
        _, page_id = _upload_and_get_page_id(client)

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            with patch("main.run_pipeline", return_value=_MOCK_ANNOTATED):
                client.post(f"/pages/{page_id}/ocr")

        get_resp = client.get(f"/pages/{page_id}/ocr")
        assert get_resp.status_code == 200
        assert get_resp.json()["word_count"] == 2

    def test_run_ocr_unknown_page_returns_404(self, client):
        resp = client.post("/pages/no-such-page/ocr")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /pages/{page_id}/ocr
# ---------------------------------------------------------------------------

class TestGetOCR:
    def test_get_ocr_before_run_returns_404(self, client):
        _, page_id = _upload_and_get_page_id(client)
        resp = client.get(f"/pages/{page_id}/ocr")
        assert resp.status_code == 404

    def test_get_ocr_unknown_page_returns_404(self, client):
        resp = client.get("/pages/no-such-page/ocr")
        assert resp.status_code == 404

    def test_get_ocr_returns_latest_run(self, client):
        _, page_id = _upload_and_get_page_id(client)

        mock_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch("preprocessing.image_pipeline.preprocess_image", return_value=(mock_img, {})):
            with patch("main.run_pipeline", return_value=_MOCK_ANNOTATED):
                client.post(f"/pages/{page_id}/ocr")
                client.post(f"/pages/{page_id}/ocr")  # run twice — second overwrites

        resp = client.get(f"/pages/{page_id}/ocr")
        assert resp.status_code == 200
        body = resp.json()
        assert body["decisions"]["accept"] == 1
