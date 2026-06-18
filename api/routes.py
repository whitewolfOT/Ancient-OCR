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

# Training-pairs storage — module-level so tests can monkeypatch them
_PAIRS_DIR = Path("data/training_pairs")
_MANIFEST  = _PAIRS_DIR / "manifest.json"

# Line corrections storage — module-level so tests can monkeypatch them
_LINES_DIR       = Path("data/lines")
_CORRECTIONS_DIR = Path("data/corrections")

# Manuscript import workflow storage — module-level so tests can monkeypatch it
_IMPORT_DIR = Path("data/import")

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
        PreviewRequest, PreviewResponse, SettingsApplied, DegradationFlags,
        GroundTruthRequest, GroundTruthResponse, GroundTruthData,
        ApplyClusterSettingsRequest, ApplyClusterSettingsResponse,
        OCRResultResponse, OCRTokenResult,
        ProfileUpdateRequest, ProfileListResponse,
        ProfileSuggestResponse,
        CorrectionSubmitRequest, CorrectionSubmitResponse,
        LineGroundTruthRequest, LineGroundTruthResponse, LineGroundTruthItem,
        LineRecord, LinesPageResponse, LineSaveRequest, LineSaveResponse,
        ImportUploadResponse, ImportSessionInfo, ImportSessionsResponse,
        ImportLineItem, ImportSegmentResponse,
        ImportAcceptLinesRequest, ImportAcceptLinesResponse,
    )
except ImportError:
    pass

# Lazy profile manager singleton — initialised on first route call, not at import
_profile_mgr = None


def _get_profile_mgr():
    global _profile_mgr
    if _profile_mgr is None:
        from ocr_engine.profile_loader import ProfileManager
        _profile_mgr = ProfileManager(Path("config/profiles.yaml"))
    return _profile_mgr


# Lazy Kraken segmentation model singleton, dedicated to /api/import/segment.
# Always uses the staged muharaf_seg_best.mlmodel, independent of profile.seg_model.
_import_seg_model = None


def _get_import_seg_model():
    global _import_seg_model
    if _import_seg_model is None:
        from kraken.lib import vgsl as kraken_vgsl
        model_path = Path("models/kraken/muharaf_seg_best.mlmodel")
        _import_seg_model = kraken_vgsl.TorchVGSLModel.load_model(str(model_path))
    return _import_seg_model


def _run_blla_segment(image, rtl: bool = True) -> list:
    """Run Kraken blla segmentation. Returns line dicts with id/bbox/baseline/boundary."""
    import cv2 as _cv2
    from kraken import blla
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(_cv2.cvtColor(image, _cv2.COLOR_BGR2RGB))
    text_dir = "horizontal-rl" if rtl else "horizontal-lr"
    seg = blla.segment(pil_img, model=_get_import_seg_model(), text_direction=text_dir)

    lines = []
    for i, line in enumerate(seg.lines):
        boundary = [[int(p[0]), int(p[1])] for p in (getattr(line, "boundary", None) or [])]
        baseline = [[int(p[0]), int(p[1])] for p in (getattr(line, "baseline", None) or [])]
        if not boundary:
            continue
        bxs = [p[0] for p in boundary]
        bys = [p[1] for p in boundary]
        lines.append({
            "id": f"line_{i:03d}",
            "bbox": [min(bxs), min(bys), max(bxs) - min(bxs), max(bys) - min(bys)],
            "baseline": baseline,
            "boundary": boundary,
        })
    return lines


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
        PreviewRequest, PreviewResponse, SettingsApplied, DegradationFlags,
        GroundTruthRequest, GroundTruthResponse, GroundTruthData,
        ApplyClusterSettingsRequest, ApplyClusterSettingsResponse,
        OCRResultResponse, OCRTokenResult,
        ImportUploadResponse, ImportSessionInfo, ImportSessionsResponse,
        ImportLineItem, ImportSegmentResponse,
        ImportAcceptLinesRequest, ImportAcceptLinesResponse,
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
            result_img, preview_meta = image_pipeline.preprocess_image(img, config=preview_cfg)
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            # Encode result as JPEG base64
            ok, buf = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("JPEG encoding failed")

            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            raw_flags = preview_meta.get("degradation_flags", {})
            deg_flags = DegradationFlags(
                low_contrast=bool(raw_flags.get("low_contrast", False)),
                faded_ink=bool(raw_flags.get("faded_ink", False)),
                high_noise=bool(raw_flags.get("high_noise", False)),
                bleed_through=bool(raw_flags.get("bleed_through", False)),
            )
            suggested = preview_meta.get("suggested_settings", {})

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
            degradation_flags=deg_flags,
            suggested_settings=suggested,
        )

    @router.post("/pages/{page_id}/ground-truth", response_model=GroundTruthResponse)
    def save_ground_truth(page_id: str, body: GroundTruthRequest):
        from documents import store

        store.init_db()
        if not body.text.strip():
            raise HTTPException(status_code=422, detail="text must not be empty")
        if store.get_page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        saved_at = datetime.now(timezone.utc).isoformat()
        store.upsert_ground_truth(page_id, body.text, saved_at)
        return GroundTruthResponse(page_id=page_id, saved_at=saved_at)

    @router.get("/pages/{page_id}/ground-truth", response_model=GroundTruthData)
    def get_ground_truth(page_id: str):
        from documents import store

        store.init_db()
        if store.get_page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        gt = store.get_ground_truth(page_id)
        if gt is None:
            raise HTTPException(status_code=404, detail="No ground truth saved for this page")

        return GroundTruthData(
            page_id=gt["page_id"],
            text=gt["text"],
            submitted_at=gt["submitted_at"],
        )

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

    @router.post("/pages/{page_id}/ocr", response_model=OCRResultResponse)
    def run_page_ocr(page_id: str):
        import json as _json
        from documents import store

        store.init_db()
        page = store.get_page(page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        try:
            import cv2 as _cv2
            import numpy as _np
            from main import run_pipeline
            from utils.config import get_config

            base_cfg = get_config()

            # Apply saved page settings if present
            saved = store.get_page_settings(page_id)

            class _NS:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)

            if saved:
                ocr_cfg = _NS(
                    preprocessing=_NS(
                        target_dpi=300,
                        clahe=_NS(enabled=True, clip_limit=saved["clahe"], tile_grid=[8, 8]),
                        denoise=_NS(enabled=True, kernel_size=saved["denoise"], method="median"),
                        deskew=_NS(
                            enabled=True,
                            no_op_threshold=saved["deskew_threshold"],
                            max_angle=15.0,
                            angle_steps=100,
                        ),
                        binarize=_NS(method=saved["binarization"].lower()),
                    ),
                    ocr=base_cfg.ocr,
                    normalization=base_cfg.normalization,
                    morphology=base_cfg.morphology,
                    lexicon=base_cfg.lexicon,
                    scoring=base_cfg.scoring,
                    decision=base_cfg.decision,
                    context_scorer=base_cfg.context_scorer,
                    output=base_cfg.output,
                    cache=base_cfg.cache,
                    logging=base_cfg.logging,
                )
            else:
                ocr_cfg = base_cfg

            img = _cv2.imread(page["image_path"])
            if img is None:
                raise ValueError(f"Cannot read image at {page['image_path']}")

            page_image = {
                "image": img,
                "page_index": page["page_num"],
                "dpi": 300,
                "source_path": page["image_path"],
                "page_count": 1,
            }

            import concurrent.futures as _cf
            log.info(f"run_page_ocr {page_id}: submitting pipeline")
            with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                _fut = _pool.submit(run_pipeline, [page_image], mode="annotated", cfg=ocr_cfg)
                try:
                    annotated = _fut.result(timeout=60)
                except _cf.TimeoutError:
                    raise TimeoutError("OCR pipeline exceeded 60-second timeout")
            log.info(f"run_page_ocr {page_id}: pipeline complete")

        except Exception as exc:
            log.warning(f"run_page_ocr {page_id} failed: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

        # Build structured response from annotated output
        raw_tokens = annotated.get("tokens", [])
        decision_counts = {"accept": 0, "accept_with_note": 0, "uncertain": 0, "review_required": 0}

        ocr_tokens: list[OCRTokenResult] = []
        for t in raw_tokens:
            dec = t.get("decision", "")
            if dec in decision_counts:
                decision_counts[dec] += 1
            raw_bbox = t.get("bbox")
            if raw_bbox and len(raw_bbox) == 4:
                x, y, w, h = raw_bbox
                bbox_out = [x, y, x + w, y + h]
            else:
                bbox_out = [0, 0, 0, 0]
            ocr_tokens.append(OCRTokenResult(
                text=t.get("selected") or t.get("original", ""),
                bbox=bbox_out,
                confidence=round(float(t.get("confidence", 0.0)), 4),
                decision=dec,
                sources=t.get("sources", []),
            ))

        processed_at = datetime.now(timezone.utc).isoformat()
        result_payload = {
            "page_id": page_id,
            "word_count": len(ocr_tokens),
            "decisions": decision_counts,
            "tokens": [t.model_dump() for t in ocr_tokens],
            "processed_at": processed_at,
        }

        store.upsert_ocr_result(page_id, _json.dumps(result_payload), processed_at)
        store.update_page_status(page_id, "ocr_done")

        return OCRResultResponse(**result_payload)

    @router.get("/pages/{page_id}/ocr", response_model=OCRResultResponse)
    def get_page_ocr(page_id: str):
        import json as _json
        from documents import store

        store.init_db()
        if store.get_page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        saved = store.get_ocr_result(page_id)
        if saved is None:
            raise HTTPException(status_code=404, detail="No OCR result for this page yet")

        return OCRResultResponse(**_json.loads(saved["result_json"]))

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

    # ── POST /api/preview ─────────────────────────────────────────────────
    @router.post("/api/preview")
    async def api_preview(
        file: UploadFile = File(...),
        profile_name: str = Form("default"),
        brightness:       float = Form(0.0),
        contrast:         float = Form(1.0),
        gamma:            float = Form(1.0),
        saturation:       float = Form(1.0),
        stroke_enabled:   bool  = Form(False),
        stroke_width:     int   = Form(2),
        denoise_strength: int   = Form(0),
        sharpen:          float = Form(0.0),
    ):
        import copy
        import base64
        import cv2 as _cv2
        import numpy as _np
        from preprocessing.adjustments import apply_profile_adjustments

        mgr = _get_profile_mgr()
        profile = copy.deepcopy(mgr.get(profile_name))
        profile.preprocessing.brightness = int(brightness)
        profile.preprocessing.contrast = contrast
        profile.preprocessing.gamma = gamma
        profile.preprocessing.saturation = saturation
        profile.preprocessing.stroke_normalization_enabled = stroke_enabled
        profile.preprocessing.stroke_target_width = stroke_width
        profile.preprocessing.denoise_strength = int(denoise_strength)
        profile.preprocessing.sharpen = sharpen

        contents = await file.read()
        img = _cv2.imdecode(_np.frombuffer(contents, _np.uint8), _cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        processed = apply_profile_adjustments(img, profile.preprocessing)
        _, buf = _cv2.imencode(".jpg", processed, [_cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()
        return {"processed_image_b64": b64, "profile_used": profile_name}

    # ── GET /api/profiles ──────────────────────────────────────────────────
    @router.get("/api/profiles", response_model=ProfileListResponse)
    def api_list_profiles():
        return ProfileListResponse(profiles=_get_profile_mgr().list())

    # ── GET /api/profiles/{name} ───────────────────────────────────────────
    @router.get("/api/profiles/{name}")
    def api_get_profile(name: str):
        p = _get_profile_mgr().get(name)
        return {
            "name": p.name,
            "description": p.description,
            "binarizer": p.binarizer,
            "seg_model": p.seg_model,
            "rec_model": p.rec_model,
            "rec_model_secondary": p.rec_model_secondary,
            "n_best": p.n_best,
            "rtl": p.rtl,
            "device": p.device,
            "preprocessing": {
                "brightness": p.preprocessing.brightness,
                "contrast": p.preprocessing.contrast,
                "gamma": p.preprocessing.gamma,
                "saturation": p.preprocessing.saturation,
                "stroke_normalization": {
                    "enabled": p.preprocessing.stroke_normalization_enabled,
                    "target_width": p.preprocessing.stroke_target_width,
                },
                "denoise_strength": p.preprocessing.denoise_strength,
                "sharpen": p.preprocessing.sharpen,
            },
        }

    # ── PUT /api/profiles/{name} ───────────────────────────────────────────
    @router.put("/api/profiles/{name}")
    async def api_update_profile(name: str, body: ProfileUpdateRequest):
        from ocr_engine.profile_loader import OCRProfile, PreprocessingParams

        profile_name = body.name or name
        pp = body.preprocessing
        sn = pp.stroke_normalization
        pre = PreprocessingParams(
            brightness=int(pp.brightness),
            contrast=float(pp.contrast),
            gamma=float(pp.gamma),
            saturation=float(pp.saturation),
            stroke_normalization_enabled=bool(sn.enabled),
            stroke_target_width=int(sn.target_width),
            denoise_strength=int(pp.denoise_strength),
            sharpen=float(pp.sharpen),
        )
        profile = OCRProfile(
            name=profile_name,
            description=body.description,
            binarizer=body.binarizer,
            seg_model=body.seg_model,
            rec_model=body.rec_model,
            rec_model_secondary=body.rec_model_secondary,
            n_best=body.n_best,
            rtl=body.rtl,
            device=body.device,
            preprocessing=pre,
        )
        mgr = _get_profile_mgr()
        mgr.upsert(profile)
        mgr.save()
        return {"status": "saved", "name": profile_name}

    # ── DELETE /api/profiles/{name} ────────────────────────────────────────
    @router.delete("/api/profiles/{name}")
    def api_delete_profile(name: str):
        mgr = _get_profile_mgr()
        if not mgr.delete(name):
            raise HTTPException(status_code=403, detail="Cannot delete protected profile")
        mgr.save()
        return {"status": "deleted", "name": name}

    # ── POST /api/suggest_profile ──────────────────────────────────────────
    @router.post("/api/suggest_profile", response_model=ProfileSuggestResponse)
    async def api_suggest_profile(file: UploadFile = File(...)):
        import numpy as _np
        import cv2 as _cv2

        contents = await file.read()
        img = _cv2.imdecode(_np.frombuffer(contents, _np.uint8), _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ProfileSuggestResponse(suggested_profile="default", confidence=0.0)

        mean_brightness = float(_np.mean(img))
        std = float(_np.std(img))

        if mean_brightness < 100:
            return ProfileSuggestResponse(suggested_profile="low_contrast", confidence=0.6)
        elif std < 30:
            return ProfileSuggestResponse(suggested_profile="low_contrast", confidence=0.5)
        else:
            return ProfileSuggestResponse(suggested_profile="default", confidence=0.9)

    # ── GET /api/suggest_profile_for_page/{page_id} ───────────────────────
    @router.get("/api/suggest_profile_for_page/{page_id}",
                response_model=ProfileSuggestResponse)
    def api_suggest_profile_for_page(page_id: str):
        import numpy as _np
        import cv2 as _cv2
        from documents import store

        store.init_db()
        page = store.get_page(page_id)
        if page is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        img = _cv2.imread(page["image_path"], _cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ProfileSuggestResponse(suggested_profile="default", confidence=0.0)

        mean_b = float(_np.mean(img))
        std_b = float(_np.std(img))
        if mean_b < 100:
            return ProfileSuggestResponse(suggested_profile="low_contrast", confidence=0.6)
        elif std_b < 30:
            return ProfileSuggestResponse(suggested_profile="low_contrast", confidence=0.5)
        else:
            return ProfileSuggestResponse(suggested_profile="default", confidence=0.9)

    # ── POST /pages/{page_id}/line-ground-truth ────────────────────────────
    @router.post("/pages/{page_id}/line-ground-truth",
                 response_model=LineGroundTruthResponse)
    def save_line_ground_truth(page_id: str, body: LineGroundTruthRequest):
        from documents import store

        store.init_db()
        if store.get_page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")
        if not body.lines:
            raise HTTPException(status_code=422, detail="lines must not be empty")

        saved_at = datetime.now(timezone.utc).isoformat()
        store.upsert_line_ground_truth(page_id, body.lines, saved_at)
        items = [LineGroundTruthItem(line_index=i, text=t, submitted_at=saved_at)
                 for i, t in enumerate(body.lines)]
        return LineGroundTruthResponse(page_id=page_id, lines=items, saved_at=saved_at)

    # ── GET /pages/{page_id}/line-ground-truth ─────────────────────────────
    @router.get("/pages/{page_id}/line-ground-truth",
                response_model=LineGroundTruthResponse)
    def get_line_ground_truth(page_id: str):
        from documents import store

        store.init_db()
        if store.get_page(page_id) is None:
            raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")

        rows = store.get_line_ground_truth(page_id)
        if rows is None:
            raise HTTPException(status_code=404, detail="No line ground truth for this page")

        items = [LineGroundTruthItem(**r) for r in rows]
        saved_at = rows[0]["submitted_at"] if rows else ""
        return LineGroundTruthResponse(page_id=page_id, lines=items, saved_at=saved_at)

    # ── POST /api/correction ───────────────────────────────────────────────
    @router.post("/api/correction", response_model=CorrectionSubmitResponse)
    async def api_correction(body: CorrectionSubmitRequest):
        import uuid as _uuid
        from datetime import datetime, timezone
        from training.feedback_store import submit
        from confidence_engine.state import FeedbackEntry

        entry = FeedbackEntry(
            id=str(_uuid.uuid4()),
            image_path="",
            bbox=(0, 0, 0, 0),
            page_index=0,
            predicted=body.original_text,
            ground_truth=body.corrected_text,
            source_file=body.token_id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        entry_id = submit(entry)

        # RALM Bayesian update from crowd correction
        try:
            from utils.config import get_config as _get_config
            from main import get_ralm_oracle as _get_ralm_oracle
            _cfg = _get_config()
            if getattr(getattr(_cfg, 'ralm', None), 'enabled', False):
                _oracle = _get_ralm_oracle(_cfg)
                _oracle.update(
                    corrected_word=body.corrected_text,
                    context_words=body.context_words,
                    user_trust_score=body.user_trust_score,
                )
        except Exception as _ralm_exc:
            import logging as _log
            _log.getLogger(__name__).warning(f"RALM update failed (non-fatal): {_ralm_exc}")

        return CorrectionSubmitResponse(status="stored", entry_id=entry_id)

    # ── Training pairs helpers ─────────────────────────────────────────────

    import api.routes as _self

    def _load_manifest() -> dict:
        if _self._MANIFEST.exists():
            import json as _j
            return _j.loads(_self._MANIFEST.read_text(encoding="utf-8"))
        return {"pairs": [], "total": 0}

    def _save_manifest(manifest: dict) -> None:
        import json as _j
        _self._PAIRS_DIR.mkdir(parents=True, exist_ok=True)
        _self._MANIFEST.write_text(_j.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── POST /api/training-pairs ───────────────────────────────────────────
    @router.post("/api/training-pairs")
    async def save_training_pair(
        page_id:      str = Form(...),
        token_index:  int = Form(...),
        label:        str = Form(...),
        patch_b64:    str = Form(...),
        original_bbox: str = Form(...),
    ):
        import json as _j, base64 as _b64
        from datetime import datetime, timezone

        if not label.strip():
            raise HTTPException(status_code=422, detail="label must not be empty")

        try:
            bbox = _j.loads(original_bbox)
        except Exception:
            raise HTTPException(status_code=422, detail="original_bbox must be a JSON array [x,y,w,h]")

        try:
            img_bytes = _b64.b64decode(patch_b64)
        except Exception:
            raise HTTPException(status_code=422, detail="patch_b64 is not valid base64")

        page_dir = _self._PAIRS_DIR / page_id
        page_dir.mkdir(parents=True, exist_ok=True)

        patch_stem   = f"patch_{token_index:04d}"
        patch_png    = page_dir / f"{patch_stem}.png"
        patch_txt    = page_dir / f"{patch_stem}.txt"
        patch_png.write_bytes(img_bytes)
        patch_txt.write_text(label, encoding="utf-8")

        pair_id      = f"{page_id}/{patch_stem}"
        created_at   = datetime.now(timezone.utc).isoformat()

        manifest = _load_manifest()
        existing = next((i for i, p in enumerate(manifest["pairs"]) if p["id"] == pair_id), None)
        entry = {
            "id":          pair_id,
            "page":        page_id,
            "token_index": token_index,
            "label":       label,
            "patch_path":  str(patch_png),
            "bbox":        bbox,
            "created_at":  created_at,
        }
        if existing is not None:
            manifest["pairs"][existing] = entry
        else:
            manifest["pairs"].append(entry)
        manifest["total"] = len(manifest["pairs"])
        _save_manifest(manifest)

        return {"status": "saved", "pair_id": pair_id, "total_pairs": manifest["total"]}

    # ── GET /api/training-pairs ────────────────────────────────────────────
    @router.get("/api/training-pairs")
    def get_training_pairs():
        return _load_manifest()

    # ── GET /api/training-pairs/count ─────────────────────────────────────
    @router.get("/api/training-pairs/count")
    def get_training_pairs_count():
        return {"total": _load_manifest()["total"]}

    # ── GET /api/training-pairs/export ────────────────────────────────────
    @router.get("/api/training-pairs/export")
    def export_training_pairs():
        import zipfile, io as _io, json as _j
        from fastapi.responses import StreamingResponse

        manifest = _load_manifest()
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", _j.dumps(manifest, ensure_ascii=False, indent=2))
            for pair in manifest["pairs"]:
                p = Path(pair["patch_path"])
                if p.exists():
                    zf.write(str(p), arcname=f"{pair['page']}/{p.name}")
                txt = p.with_suffix(".txt")
                if txt.exists():
                    zf.write(str(txt), arcname=f"{pair['page']}/{txt.name}")
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=training_pairs.zip"},
        )

    # ── Line correction helpers ────────────────────────────────────────────

    import api.routes as _self_mod

    def _load_lines_json(page_id: str) -> dict | None:
        p = _self_mod._LINES_DIR / page_id / "lines.json"
        if not p.exists():
            return None
        import json as _j
        return _j.loads(p.read_text(encoding="utf-8"))

    def _load_corrections(page_id: str) -> dict:
        p = _self_mod._CORRECTIONS_DIR / page_id / "corrections.json"
        if not p.exists():
            return {}
        import json as _j
        return _j.loads(p.read_text(encoding="utf-8"))

    def _save_corrections(page_id: str, corrections: dict) -> None:
        import json as _j
        d = _self_mod._CORRECTIONS_DIR / page_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "corrections.json").write_text(
            _j.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── GET /api/lines/{page_id} ───────────────────────────────────────────
    @router.get("/api/lines/{page_id}", response_model=LinesPageResponse)
    def api_get_lines(page_id: str):
        data = _load_lines_json(page_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No lines data for '{page_id}'")

        corrections = _load_corrections(page_id)
        records = []
        for ln in data.get("lines", []):
            idx = ln["index"]
            key = str(idx)
            corr = corrections.get(key, {})
            status = corr.get("status", "pending")
            records.append(LineRecord(
                index=idx,
                image_path=ln.get("image_path", ""),
                ocr_text=ln.get("ocr_text", ""),
                corrected_text=corr.get("corrected_text"),
                bbox=ln.get("bbox", [0, 0, 0, 0]),
                confidence=ln.get("confidence", 0.0),
                status=status,
            ))
        return LinesPageResponse(page=page_id, lines=records)

    # ── POST /api/lines/{page_id}/{line_index}/correction ──────────────────
    @router.post(
        "/api/lines/{page_id}/{line_index}/correction",
        response_model=LineSaveResponse,
    )
    def api_save_line_correction(page_id: str, line_index: int, body: LineSaveRequest):
        if body.status not in ("corrected", "skipped"):
            raise HTTPException(status_code=422, detail="status must be 'corrected' or 'skipped'")

        data = _load_lines_json(page_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No lines data for '{page_id}'")
        total_lines = len(data.get("lines", []))

        # Save plain-text .gt.txt
        txt_dir = _self_mod._CORRECTIONS_DIR / page_id
        txt_dir.mkdir(parents=True, exist_ok=True)
        gt_file = txt_dir / f"line_{line_index:03d}.txt"
        gt_file.write_text(body.corrected_text, encoding="utf-8")

        # Update corrections.json metadata
        corrections = _load_corrections(page_id)
        from datetime import datetime, timezone
        corrections[str(line_index)] = {
            "corrected_text": body.corrected_text,
            "status": body.status,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_corrections(page_id, corrections)

        total_corrected = sum(
            1 for v in corrections.values() if v.get("status") == "corrected"
        )
        return LineSaveResponse(
            saved=True,
            total_corrected=total_corrected,
            total_lines=total_lines,
        )

    # ── GET /api/corrections/export ────────────────────────────────────────
    @router.get("/api/corrections/export")
    def api_export_corrections():
        import io as _io, zipfile as _zf, json as _j
        from fastapi.responses import StreamingResponse

        def _get_png_bytes(page_id: str, idx: int) -> bytes | None:
            """Return PNG bytes for a line crop.

            Priority:
            1. Pre-generated file at data/lines/{page_id}/line_NNN.png
            2. Generate on-the-fly from original image + bbox in lines.json
            """
            cached = _self_mod._LINES_DIR / page_id / f"line_{idx:03d}.png"
            if cached.exists():
                return cached.read_bytes()

            # On-the-fly generation
            lines_data = _load_lines_json(page_id)
            if not lines_data:
                return None
            line_records = lines_data.get("lines", [])
            if idx >= len(line_records):
                return None
            bbox = line_records[idx].get("bbox", [])
            if len(bbox) < 4:
                return None

            src_img = Path("data/test_images") / page_id
            if not src_img.exists():
                return None

            try:
                import cv2 as _cv2, numpy as _np
                img = _cv2.imread(str(src_img))
                if img is None:
                    return None
                bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                PAD = 4
                h_img, w_img = img.shape[:2]
                x1 = max(0, bx - PAD); y1 = max(0, by - PAD)
                x2 = min(w_img, bx + bw + PAD); y2 = min(h_img, by + bh + PAD)
                crop = img[y1:y2, x1:x2]
                ok, buf_enc = _cv2.imencode(".png", crop)
                return bytes(buf_enc) if ok else None
            except Exception:
                return None

        buf = _io.BytesIO()
        pair_count = 0
        readme = (
            "Kraken training data\n"
            "====================\n"
            "To fine-tune Kraken: ketos train -f alto *.gt.txt\n"
            "See https://kraken.re/main/training.html\n"
        )

        with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", readme)
            manifest_lines = []

            corrections_root = _self_mod._CORRECTIONS_DIR
            if corrections_root.exists():
                for page_dir in sorted(corrections_root.iterdir()):
                    if not page_dir.is_dir():
                        continue
                    page_id = page_dir.name
                    corr_json = page_dir / "corrections.json"
                    if not corr_json.exists():
                        continue
                    corrections = _j.loads(corr_json.read_text(encoding="utf-8"))
                    for idx_str, corr in corrections.items():
                        if corr.get("status") != "corrected":
                            continue
                        idx = int(idx_str)
                        gt_text = corr.get("corrected_text", "")
                        arc_base = f"{page_id}/line_{idx:03d}"

                        png_bytes = _get_png_bytes(page_id, idx)
                        if png_bytes:
                            zf.writestr(f"{arc_base}.png", png_bytes)
                        zf.writestr(f"{arc_base}.gt.txt", gt_text)
                        manifest_lines.append(f"{arc_base}.png\t{arc_base}.gt.txt")
                        pair_count += 1

            zf.writestr("manifest.txt", "\n".join(manifest_lines))

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=training_data.zip",
                "X-Pair-Count": str(pair_count),
            },
        )

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=training_data.zip",
                "X-Pair-Count": str(pair_count),
            },
        )

    # ── GET /api/lines/{page_id}/image/{filename} ──────────────────────────
    @router.get("/api/lines/{page_id}/image/{filename}")
    def api_serve_line_image(page_id: str, filename: str):
        from fastapi.responses import FileResponse
        img_path = _self_mod._LINES_DIR / page_id / filename
        if not img_path.exists() or img_path.suffix not in (".png", ".jpg", ".jpeg"):
            raise HTTPException(status_code=404, detail="Line image not found")
        return FileResponse(str(img_path), media_type="image/png")

    # ── GET /api/corrections/summary ───────────────────────────────────────
    @router.get("/api/corrections/summary")
    def api_corrections_summary():
        """Per-page and aggregate correction progress. Fast — reads only JSONs."""
        import json as _j
        _KNOWN_PAGES = [f"{i}.jpg" for i in range(1, 10)]
        pages_out = []
        grand_total = grand_corrected = 0

        for page_id in _KNOWN_PAGES:
            lines_data = _load_lines_json(page_id)
            total = len(lines_data.get("lines", [])) if lines_data else 0
            corrections = _load_corrections(page_id)
            corrected = sum(1 for v in corrections.values() if v.get("status") == "corrected")
            skipped   = sum(1 for v in corrections.values() if v.get("status") == "skipped")
            pending   = max(0, total - corrected - skipped)
            pages_out.append({
                "page_id":   page_id,
                "total":     total,
                "corrected": corrected,
                "skipped":   skipped,
                "pending":   pending,
            })
            grand_total     += total
            grand_corrected += corrected

        return {
            "total_corrected": grand_corrected,
            "total_lines":     grand_total,
            "pages":           pages_out,
        }

    # ── Manuscript import workflow ──────────────────────────────────────────

    def _import_session_paths(session_id: str) -> dict:
        base = _self_mod._IMPORT_DIR / session_id
        return {
            "base": base,
            "pages": base / "pages",
            "segments": base / "segments",
            "accepted": base / "accepted",
        }

    def _assign_tokens_to_lines(tokens: list, line_bboxes: list) -> dict:
        """Assign each OCR token to the best-overlapping line bbox by y-overlap."""
        assignments: dict = {i: [] for i in range(len(line_bboxes))}
        if not line_bboxes:
            return assignments
        for tok in tokens:
            tx, ty, tw, th = tok["bbox"]
            best_i, best_overlap = 0, None
            for i, bbox in enumerate(line_bboxes):
                ly, lh = bbox[1], bbox[3]
                overlap = min(ty + th, ly + lh) - max(ty, ly)
                if best_overlap is None or overlap > best_overlap:
                    best_overlap = overlap
                    best_i = i
            assignments[best_i].append(tok)
        return assignments

    def _line_text_rtl(tokens: list) -> str:
        sorted_toks = sorted(tokens, key=lambda t: -t["bbox"][0])
        return " ".join(t["text"] for t in sorted_toks if t.get("text", "").strip())

    def _mean_confidence(tokens: list) -> float:
        confs = [t["confidence"] for t in tokens if "confidence" in t]
        return round(float(sum(confs) / len(confs)), 4) if confs else 0.0

    def _crop_line_region(img, boundary: list, padding: int = 4):
        import cv2 as _cv2
        import numpy as _np
        pts = _np.array(boundary, dtype=_np.int32)
        x, y, w, h = _cv2.boundingRect(pts)
        h_img, w_img = img.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)
        crop = img[y1:y2, x1:x2].copy()
        mask = _np.zeros(img.shape[:2], dtype=_np.uint8)
        shifted = pts - _np.array([x1, y1])
        _cv2.fillPoly(mask[y1:y2, x1:x2], [shifted], 255)
        result = _np.full_like(crop, 255)
        result[mask[y1:y2, x1:x2] > 0] = crop[mask[y1:y2, x1:x2] > 0]
        return result

    # ── POST /api/import/upload-pdf ─────────────────────────────────────────
    @router.post("/api/import/upload-pdf", response_model=ImportUploadResponse)
    async def import_upload_pdf(
        file: UploadFile = File(...),
        profile_name: str = Form("default"),
    ):
        from starlette.concurrency import run_in_threadpool
        from scripts.extract_pdf_pages import extract_pdf_pages

        session_id = uuid.uuid4().hex[:12]
        paths = _import_session_paths(session_id)
        paths["pages"].mkdir(parents=True, exist_ok=True)

        contents = await file.read()
        tmp_path = Path(tempfile.mktemp(suffix=".pdf"))
        tmp_path.write_bytes(contents)

        try:
            written = await run_in_threadpool(
                extract_pdf_pages,
                pdf_path=tmp_path,
                output_dir=paths["pages"],
                prefix="page",
                dpi=300,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        page_ids = [p.stem for p in written]
        return ImportUploadResponse(session_id=session_id, page_count=len(page_ids), page_ids=page_ids)

    # ── POST /api/import/upload-images ──────────────────────────────────────
    @router.post("/api/import/upload-images", response_model=ImportUploadResponse)
    async def import_upload_images(
        files: list[UploadFile] = File(...),
        session_id: str = Form(None),
    ):
        import cv2 as _cv2
        import numpy as _np

        if not session_id:
            session_id = uuid.uuid4().hex[:12]
        paths = _import_session_paths(session_id)
        paths["pages"].mkdir(parents=True, exist_ok=True)

        next_num = len(list(paths["pages"].glob("page_*.jpg"))) + 1
        page_ids = []
        for f in files:
            contents = await f.read()
            img = _cv2.imdecode(_np.frombuffer(contents, _np.uint8), _cv2.IMREAD_COLOR)
            if img is None:
                continue
            out_path = paths["pages"] / f"page_{next_num:04d}.jpg"
            _cv2.imwrite(str(out_path), img)
            page_ids.append(out_path.stem)
            next_num += 1

        total_pages = len(list(paths["pages"].glob("page_*.jpg")))
        return ImportUploadResponse(session_id=session_id, page_count=total_pages, page_ids=page_ids)

    # ── GET /api/import/sessions ────────────────────────────────────────────
    @router.get("/api/import/sessions", response_model=ImportSessionsResponse)
    def import_list_sessions():
        sessions = []
        if _self_mod._IMPORT_DIR.exists():
            for d in sorted(_self_mod._IMPORT_DIR.iterdir()):
                if not d.is_dir():
                    continue
                pages_dir = d / "pages"
                count = len(list(pages_dir.glob("page_*.jpg"))) if pages_dir.exists() else 0
                sessions.append(ImportSessionInfo(session_id=d.name, page_count=count))
        return ImportSessionsResponse(sessions=sessions)

    # ── GET /api/import/page-image/{session_id}/{page_id} ──────────────────
    @router.get("/api/import/page-image/{session_id}/{page_id}")
    def import_page_image(session_id: str, page_id: str, preprocessed: bool = False, profile: str = "default"):
        import cv2 as _cv2
        from fastapi.responses import FileResponse, Response

        paths = _import_session_paths(session_id)
        img_path = paths["pages"] / f"{page_id}.jpg"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Page not found")

        if not preprocessed:
            return FileResponse(str(img_path), media_type="image/jpeg")

        from preprocessing.adjustments import apply_profile_adjustments

        img = _cv2.imread(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        prof = _get_profile_mgr().get(profile)
        processed = apply_profile_adjustments(img, prof.preprocessing)
        _, buf = _cv2.imencode(".jpg", processed, [_cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(content=bytes(buf), media_type="image/jpeg")

    # ── POST /api/import/segment/{session_id}/{page_id} ────────────────────
    @router.post("/api/import/segment/{session_id}/{page_id}", response_model=ImportSegmentResponse)
    def import_segment_page(session_id: str, page_id: str, profile: str = "default"):
        import json as _j
        import cv2 as _cv2

        paths = _import_session_paths(session_id)
        img_path = paths["pages"] / f"{page_id}.jpg"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Page not found")

        from preprocessing.adjustments import apply_profile_adjustments

        img = _cv2.imread(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        prof = _get_profile_mgr().get(profile)
        processed = apply_profile_adjustments(img, prof.preprocessing)

        lines = _self_mod._run_blla_segment(processed, rtl=prof.rtl)

        paths["segments"].mkdir(parents=True, exist_ok=True)
        payload = {"page_id": page_id, "line_count": len(lines), "lines": lines}
        (paths["segments"] / f"{page_id}.json").write_text(
            _j.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ImportSegmentResponse(**payload)

    # ── GET /api/import/segments/{session_id}/{page_id} ────────────────────
    @router.get("/api/import/segments/{session_id}/{page_id}", response_model=ImportSegmentResponse)
    def import_get_segments(session_id: str, page_id: str):
        import json as _j

        paths = _import_session_paths(session_id)
        seg_path = paths["segments"] / f"{page_id}.json"
        if not seg_path.exists():
            raise HTTPException(status_code=404, detail="Page not segmented yet")
        return ImportSegmentResponse(**_j.loads(seg_path.read_text(encoding="utf-8")))

    # ── POST /api/import/accept-lines/{session_id}/{page_id} ───────────────
    @router.post("/api/import/accept-lines/{session_id}/{page_id}", response_model=ImportAcceptLinesResponse)
    def import_accept_lines(session_id: str, page_id: str, body: ImportAcceptLinesRequest):
        import json as _j
        import cv2 as _cv2

        paths = _import_session_paths(session_id)
        img_path = paths["pages"] / f"{page_id}.jpg"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Page not found")

        raw_img = _cv2.imread(str(img_path))
        if raw_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        h_img, w_img = raw_img.shape[:2]

        profile = _get_profile_mgr().get(body.profile_name)
        from preprocessing.adjustments import apply_profile_adjustments
        preprocessed_img = apply_profile_adjustments(raw_img, profile.preprocessing)

        from ocr_engine.kraken_backend import KrakenBackend
        backend = KrakenBackend(profile=profile)
        ocr_result = backend.process_image(preprocessed_img, page_index=0)
        ocr_tokens = [
            {"text": w.text, "bbox": list(w.bbox), "confidence": w.confidence}
            for w in ocr_result.words
        ]

        line_bboxes = [line.bbox for line in body.lines]
        token_map = _assign_tokens_to_lines(ocr_tokens, line_bboxes)

        out_dir = _self_mod._LINES_DIR / page_id
        out_dir.mkdir(parents=True, exist_ok=True)

        line_records = []
        for i, line in enumerate(body.lines):
            boundary = line.boundary if line.boundary else [
                [line.bbox[0], line.bbox[1]],
                [line.bbox[0] + line.bbox[2], line.bbox[1]],
                [line.bbox[0] + line.bbox[2], line.bbox[1] + line.bbox[3]],
                [line.bbox[0], line.bbox[1] + line.bbox[3]],
            ]
            crop = _crop_line_region(raw_img, boundary)
            img_path_out = out_dir / f"line_{i:03d}.png"
            _cv2.imwrite(str(img_path_out), crop)

            matched = token_map.get(i, [])
            line_records.append({
                "index": i,
                "image_path": str(img_path_out),
                "ocr_text": _line_text_rtl(matched),
                "baseline": line.baseline,
                "bbox": line.bbox,
                "confidence": _mean_confidence(matched),
            })

        manifest = {
            "page": page_id,
            "original_size": [w_img, h_img],
            "seg_source": "import",
            "lines": line_records,
        }
        (out_dir / "lines.json").write_text(
            _j.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        paths["accepted"].mkdir(parents=True, exist_ok=True)
        (paths["accepted"] / f"{page_id}.json").write_text(
            _j.dumps({"lines": [l.model_dump() for l in body.lines]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return ImportAcceptLinesResponse(saved_lines=len(line_records), page_id=page_id)

    app.include_router(router)
