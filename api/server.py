"""FastAPI application entry point."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

try:
    from fastapi import FastAPI
    app = FastAPI(title="Ancient-OCR API", version="0.1.0")

    @app.on_event("startup")
    def _startup():
        try:
            from lexicon_ingestion.index_builder import get_index
            idx = get_index()
            log.info(f"startup index_ready=True lemmas={len(idx.all_lemmas)}")
        except Exception as exc:
            log.warning(f"startup index build failed: {exc}")

    from api.routes import register_routes
    register_routes(app)

except ImportError as exc:
    raise RuntimeError(
        "fastapi and uvicorn are required. Install with: pip install fastapi uvicorn"
    ) from exc
