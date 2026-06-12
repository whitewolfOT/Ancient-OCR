"""FastAPI application entry point."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Ancient-OCR API", version="0.1.0")

    # Allow the Vite dev server (and any local origin) to call the API.
    # In production the React build is served from FastAPI itself (same origin),
    # so this middleware is a no-op for same-origin requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

    from fastapi.staticfiles import StaticFiles
    from pathlib import Path as _Path
    _images_dir = _Path("data/test_images")
    _images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=str(_images_dir)), name="images")

except ImportError as exc:
    raise RuntimeError(
        "fastapi and uvicorn are required. Install with: pip install fastapi uvicorn"
    ) from exc
