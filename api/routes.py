"""API route handlers."""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

from utils.logging import get_logger

log = get_logger(__name__)


def register_routes(app):
    """Attach all routes to the FastAPI app."""
    try:
        from fastapi import APIRouter, File, Form, UploadFile, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("fastapi is required for the API") from exc

    from api.schemas import (
        ProcessResponse, HealthResponse,
        FeedbackSubmitRequest, FeedbackSubmitResponse,
        CalibrateResponse, FeedbackStatsResponse,
        VALID_MODES,
    )

    router = APIRouter()

    # ── GET /health ────────────────────────────────────────────────────────
    @router.get("/health", response_model=HealthResponse)
    def health():
        from ocr_engine.paddle_backend import PaddleBackend
        from ocr_engine.tesseract_backend import TesseractBackend
        from ocr_engine.trocr_backend import TrOCRBackend
        return HealthResponse(
            status="ok",
            engines={
                "paddle": PaddleBackend.is_available(),
                "tesseract": TesseractBackend.is_available(),
                "trocr": TrOCRBackend.is_available(),
            },
        )

    # ── POST /process ──────────────────────────────────────────────────────
    @router.post("/process")
    async def process(
        file: UploadFile = File(...),
        mode: str = Form("clean"),
    ):
        if mode not in VALID_MODES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
            )

        suffix = Path(file.filename or "upload").suffix.lower()
        allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
        if suffix not in allowed:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
            )

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            from main import process_file
            result = process_file(tmp_path, mode)
            return JSONResponse(content=result)

        except Exception as exc:
            log.warning(f"process failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── POST /feedback ─────────────────────────────────────────────────────
    @router.post("/feedback", response_model=FeedbackSubmitResponse)
    def feedback(body: FeedbackSubmitRequest):
        from confidence_engine.state import FeedbackEntry
        from training.feedback_store import submit
        from datetime import datetime, timezone

        entry = FeedbackEntry(
            id="",
            image_path=body.image_path,
            bbox=tuple(body.bbox),
            page_index=body.page_index,
            predicted=body.predicted,
            ground_truth=body.ground_truth,
            source_file=body.source_file,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        entry_id = submit(entry)
        return FeedbackSubmitResponse(id=entry_id)

    # ── POST /calibrate ────────────────────────────────────────────────────
    @router.post("/calibrate", response_model=CalibrateResponse)
    def calibrate():
        from training.calibrator import calibrate as do_calibrate
        from utils.config import get_config
        result = do_calibrate(get_config())
        return CalibrateResponse(
            sample_size=result.sample_size,
            current_weights=result.current_weights,
            suggested_weights=result.suggested_weights,
            delta=result.delta,
            warning=result.warning,
        )

    # ── GET /feedback/stats ────────────────────────────────────────────────
    @router.get("/feedback/stats", response_model=FeedbackStatsResponse)
    def feedback_stats():
        from training.feedback_store import stats
        return FeedbackStatsResponse(**stats())

    app.include_router(router)
