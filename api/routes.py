"""API route handlers."""

from __future__ import annotations

import base64
import tempfile
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from utils.logging import get_logger

log = get_logger(__name__)

# Module-level FastAPI + schema imports so ForwardRef resolution works with
# `from __future__ import annotations` (PEP 563). FastAPI resolves body
# annotations via typing.get_type_hints(func), which looks up names in
# func.__globals__ (this module's namespace). Anything only imported inside
# register_routes() is invisible to that lookup and causes 422 at request time.
try:
    from fastapi import UploadFile, File, Form, HTTPException
    from api.schemas import (
        ProcessResponse, HealthResponse,
        FeedbackSubmitRequest, FeedbackSubmitResponse,
        CalibrateResponse, FeedbackStatsResponse,
        VALID_MODES,
        UploadResponse, ClusterInfo,
        PageInfo,
        PreviewRequest, PreviewResponse, SettingsApplied,
        GroundTruthRequest, GroundTruthResponse,
        ApplyClusterSettingsRequest, ApplyClusterSettingsResponse,
    )
except ImportError:
    pass


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
        UploadResponse, ClusterInfo,
        PageInfo,
        PreviewRequest, PreviewResponse, SettingsApplied,
        GroundTruthRequest, GroundTruthResponse,
        ApplyClusterSettingsRequest, ApplyClusterSettingsResponse,
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

    # ── Document calibration endpoints ────────────────────────────────────

    @router.post("/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)):
        from documents import store, cluster as cluster_mod

        suffix = Path(file.filename or "upload").suffix.lower()
        allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
        if suffix not in allowed:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}",
            )

        doc_id = str(uuid.uuid4())
        upload_time = datetime.now(timezone.utc).isoformat()
        file_bytes = await file.read()

        pages_dir = store.DOCS_DIR / doc_id / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        store.init_db()

        # Extract page images
        page_records: list[dict] = []
        try:
            if suffix == ".pdf":
                import fitz  # PyMuPDF
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(file_bytes)
                tmp.close()
                try:
                    pdf = fitz.open(tmp.name)
                    for page_num, pdf_page in enumerate(pdf):
                        mat = fitz.Matrix(2, 2)
                        pix = pdf_page.get_pixmap(matrix=mat)
                        img_path = str(pages_dir / f"page_{page_num:04d}.jpg")
                        pix.save(img_path)
                        page_records.append({
                            "page_id": str(uuid.uuid4()),
                            "page_num": page_num,
                            "image_path": img_path,
                        })
                    pdf.close()
                finally:
                    os.unlink(tmp.name)
            else:
                img_path = str(pages_dir / f"page_0000{suffix}")
                with open(img_path, "wb") as fh:
                    fh.write(file_bytes)
                page_records.append({
                    "page_id": str(uuid.uuid4()),
                    "page_num": 0,
                    "image_path": img_path,
                })
        except Exception as exc:
            log.warning(f"upload_document extract failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Page extraction failed: {exc}")

        # Compute pHash for each page
        for rec in page_records:
            try:
                rec["phash"] = cluster_mod.compute_phash_str(rec["image_path"])
            except Exception:
                rec["phash"] = "0" * 16  # fallback: all-zero hash

        # Build clusters
        clusters_raw = cluster_mod.build_clusters(
            [{"page_id": r["page_id"], "phash": r["phash"]} for r in page_records]
        )

        # Build page_id → cluster mapping
        page_to_cluster: dict[str, str] = {}
        page_to_rep: dict[str, str] = {}
        for cl in clusters_raw:
            for pid in cl["page_ids"]:
                page_to_cluster[pid] = cl["cluster_id"]
                page_to_rep[pid] = cl["representative_page_id"]

        # Persist document + pages + clusters
        store.insert_document(doc_id, file.filename or "upload", len(page_records), upload_time)
        for rec in page_records:
            store.insert_page(
                rec["page_id"], doc_id, rec["page_num"],
                rec["image_path"],
                page_to_cluster[rec["page_id"]],
                rec["phash"],
            )
        for cl in clusters_raw:
            store.insert_cluster(cl["cluster_id"], doc_id, cl["label"], cl["representative_page_id"])

        # Build response
        cluster_infos = []
        for cl in clusters_raw:
            cluster_infos.append(ClusterInfo(
                cluster_id=cl["cluster_id"],
                label=cl["label"],
                page_count=len(cl["page_ids"]),
                representative_page_id=cl["representative_page_id"],
            ))

        return UploadResponse(
            doc_id=doc_id,
            page_count=len(page_records),
            clusters=cluster_infos,
        )

    @router.get("/documents/{doc_id}/pages", response_model=list[PageInfo])
    def get_document_pages(doc_id: str):
        from documents import store, cluster as cluster_mod

        store.init_db()
        pages = store.get_pages_for_document(doc_id)
        if not pages:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found or has no pages")

        # Build cluster representative map
        clusters = store.get_clusters_for_document(doc_id)
        rep_map: dict[str, str] = {cl["cluster_id"]: cl["representative_page_id"] for cl in clusters}

        result = []
        for p in pages:
            cluster_id = p["cluster_id"]
            rep_id = rep_map.get(cluster_id, p["page_id"])

            if p["page_id"] == rep_id:
                similarity = 1.0
            else:
                rep_pages = [x for x in pages if x["page_id"] == rep_id]
                if rep_pages:
                    similarity = cluster_mod.similarity_to_rep(p["phash"], rep_pages[0]["phash"])
                else:
                    similarity = 1.0

            gt = store.get_ground_truth(p["page_id"])
            settings = store.get_page_settings(p["page_id"])

            result.append(PageInfo(
                page_id=p["page_id"],
                page_num=p["page_num"],
                cluster_id=cluster_id,
                similarity_to_representative=similarity,
                status=p["status"],
                has_ground_truth=gt is not None,
                has_settings=settings is not None,
                thumbnail_url=f"/pages/{p['page_id']}/image",
            ))

        return result

    @router.post("/pages/{page_id}/preview", response_model=PreviewResponse)
    def preview_page(page_id: str, body: PreviewRequest):
        from documents import store
        from preprocessing import image_pipeline

        store.init_db()
        page = store.get_page(page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        try:
            import cv2
            import numpy as np

            img = cv2.imread(page["image_path"])
            if img is None:
                raise ValueError(f"Cannot read image at {page['image_path']}")

            # Build a config shim that the preprocessing pipeline understands
            class _NS:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)

            preview_cfg = _NS(
                preprocessing=_NS(
                    target_dpi=300,
                    clahe=_NS(enabled=True, clip_limit=body.clahe, tile_grid=[8, 8]),
                    denoise=_NS(enabled=True, kernel_size=body.denoise, method="median"),
                    deskew=_NS(
                        enabled=True,
                        no_op_threshold=body.deskew_threshold,
                        max_angle=15.0,
                        angle_steps=100,
                    ),
                    binarize=_NS(method=body.binarization.lower()),
                )
            )

            t0 = time.monotonic()
            result_img, _ = image_pipeline.preprocess_image(img, config=preview_cfg)
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            # Encode result as JPEG base64
            if result_img.ndim == 2:
                encode_img = result_img
            else:
                encode_img = result_img
            ok, buf = cv2.imencode(".jpg", encode_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("JPEG encoding failed")

            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        except Exception as exc:
            log.warning(f"preview_page {page_id} failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

        return PreviewResponse(
            preview_image_b64=b64,
            settings_applied=SettingsApplied(
                clahe=body.clahe,
                denoise=body.denoise,
                deskew_threshold=body.deskew_threshold,
                binarization=body.binarization,
            ),
            preview_id=str(uuid.uuid4()),
            processing_time_ms=round(elapsed_ms, 1),
        )

    @router.post("/pages/{page_id}/ground-truth", response_model=GroundTruthResponse)
    def save_ground_truth(page_id: str, body: GroundTruthRequest):
        from documents import store

        store.init_db()
        if store.get_page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        saved_at = datetime.now(timezone.utc).isoformat()
        store.upsert_ground_truth(page_id, body.text, saved_at)
        return GroundTruthResponse(page_id=page_id, saved_at=saved_at)

    @router.post(
        "/documents/{doc_id}/apply-cluster-settings",
        response_model=ApplyClusterSettingsResponse,
    )
    def apply_cluster_settings(doc_id: str, body: ApplyClusterSettingsRequest):
        from documents import store

        store.init_db()
        pages = store.get_cluster_pages(body.cluster_id)
        if not pages:
            raise HTTPException(status_code=404, detail=f"Cluster '{body.cluster_id}' not found or empty")

        applied_at = datetime.now(timezone.utc).isoformat()
        s = body.settings
        for p in pages:
            store.upsert_page_settings(
                p["page_id"],
                s.clahe,
                s.denoise,
                s.deskew_threshold,
                s.binarization,
                applied_at,
            )

        return ApplyClusterSettingsResponse(
            pages_updated=len(pages),
            cluster_id=body.cluster_id,
        )

    @router.get("/pages/{page_id}/image")
    def serve_page_image(page_id: str):
        from documents import store
        from fastapi.responses import FileResponse

        store.init_db()
        page = store.get_page(page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        img_path = Path(page["image_path"])
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")

        return FileResponse(str(img_path), media_type="image/jpeg")

    app.include_router(router)
