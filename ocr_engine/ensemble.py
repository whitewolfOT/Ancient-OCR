"""OCR ensemble: run enabled backends, align tokens, produce merged OCRResult."""

from __future__ import annotations

import numpy as np

from ocr_engine.schema import OCRResult, WordToken
from utils.logging import get_logger

log = get_logger(__name__)

# Module-level backend singletons — each backend loads its models once per
# process. Creating a new instance per crop would reload all 5 PaddleOCR
# models on every region, which is prohibitively slow.
_paddle_instance = None
_tess_instance = None
_trocr_instance = None


def _get_paddle(config):
    global _paddle_instance
    if _paddle_instance is None:
        from ocr_engine.paddle_backend import PaddleBackend
        _paddle_instance = PaddleBackend(config)
    return _paddle_instance


def _get_tess(config):
    global _tess_instance
    if _tess_instance is None:
        from ocr_engine.tesseract_backend import TesseractBackend
        _tess_instance = TesseractBackend(config)
    return _tess_instance


def _get_trocr(config):
    global _trocr_instance
    if _trocr_instance is None:
        from ocr_engine.trocr_backend import TrOCRBackend
        _trocr_instance = TrOCRBackend(config)
    return _trocr_instance


def run_ensemble(image: np.ndarray, page_index: int, config, crop_bbox: tuple = None) -> OCRResult:
    """Run all enabled OCR backends and merge their results.

    Args:
        image:      The image to OCR (crop or full page).
        page_index: Page index in the source document.
        config:     Pipeline config object.
        crop_bbox:  (x, y, w, h) offset of this image in page space.
                    Stored in OCRResult.raw for stitch_results.

    Steps:
        1. Check availability and run each enabled backend.
        2. If only one backend ran, return its result directly.
        3. Align tokens across engines via token_matcher.
        4. Per-cluster weighted vote: pick text with highest weight×confidence.
        5. TrOCR verifier pass for clusters below conf_threshold.
        6. Return merged OCRResult with all raw per-engine outputs.
    """
    from ocr_engine.paddle_backend import PaddleBackend
    from ocr_engine.tesseract_backend import TesseractBackend
    from alignment.token_matcher import match_tokens

    paddle_cfg = getattr(getattr(config, "ocr", None), "paddle", None)
    tess_cfg = getattr(getattr(config, "ocr", None), "tesseract", None)
    trocr_cfg = getattr(getattr(config, "ocr", None), "trocr", None)

    paddle_weight = getattr(paddle_cfg, "weight", 0.6)
    tess_weight = getattr(tess_cfg, "weight", 0.4)
    trocr_weight = getattr(trocr_cfg, "weight", 0.0)
    trocr_threshold = getattr(trocr_cfg, "conf_threshold", 0.5)

    paddle_enabled = getattr(paddle_cfg, "enabled", True)
    tess_enabled = getattr(tess_cfg, "enabled", True)
    trocr_enabled = getattr(trocr_cfg, "enabled", False)

    engine_results: dict[str, OCRResult] = {}
    token_lists: list[list[WordToken]] = []
    weights: list[float] = []

    # --- Run backends ---
    if paddle_enabled and PaddleBackend.is_available():
        try:
            result = _get_paddle(config).extract(image, page_index)
            engine_results["paddle"] = result
            token_lists.append(result.words)
            weights.append(paddle_weight)
        except Exception as exc:
            log.warning(f"paddle extract failed: {exc}")

    if tess_enabled and TesseractBackend.is_available():
        try:
            result = _get_tess(config).extract(image, page_index)
            engine_results["tesseract"] = result
            token_lists.append(result.words)
            weights.append(tess_weight)
        except Exception as exc:
            log.warning(f"tesseract extract failed: {exc}")

    # Handle zero or one backend
    if not engine_results:
        log.warning("no OCR backend available; returning empty result")
        return _empty_result(page_index, crop_bbox)

    if len(engine_results) == 1:
        only = next(iter(engine_results.values()))
        return _with_crop_raw(only, engine_results, crop_bbox)

    # --- Align + merge ---
    clusters = match_tokens(token_lists, config)
    merged_words: list[WordToken] = []
    texts: list[str] = []

    trocr_backend = _get_trocr(config) if trocr_enabled else None

    for cluster in clusters:
        # Weighted vote: best text = token with highest weight*confidence
        best_token = cluster.tokens[0]
        best_score = 0.0
        for token in cluster.tokens:
            w = weights[token_lists.index(
                next(lst for lst in token_lists if token in lst)
            )] if token_lists else 1.0
            s = w * token.confidence
            if s > best_score:
                best_score, best_token = s, token

        merged_conf = best_token.confidence

        # TrOCR verifier pass for low-confidence clusters
        if trocr_backend and trocr_backend.is_ready() and merged_conf < trocr_threshold:
            x, y, w, h = best_token.bbox
            crop = image[y:y+h, x:x+w] if h > 0 and w > 0 else image
            trocr_text, trocr_conf = trocr_backend.recognize_line(crop)
            if trocr_text and trocr_conf > merged_conf:
                best_token = WordToken(
                    text=trocr_text, confidence=trocr_conf,
                    bbox=best_token.bbox, page_index=page_index,
                    source="trocr", region_id=best_token.region_id,
                )
                merged_conf = trocr_conf

        merged_words.append(WordToken(
            text=best_token.text,
            confidence=merged_conf,
            bbox=best_token.bbox,
            page_index=page_index,
            source="ensemble",
            region_id=best_token.region_id,
        ))
        texts.append(best_token.text)

    page_conf = float(np.mean([w.confidence for w in merged_words])) if merged_words else 0.0
    log.debug(f"ensemble page={page_index} tokens={len(merged_words)} conf={page_conf:.3f}")

    raw = {"engines": {k: v.model_dump() for k, v in engine_results.items()}}
    if crop_bbox:
        raw["crop_bbox_in_page"] = list(crop_bbox)
    # Pass through paddle alternatives for use by candidate_generator
    if "paddle" in engine_results:
        paddle_alts = engine_results["paddle"].raw.get("paddle_alternatives", {})
        if paddle_alts:
            raw["paddle_alternatives"] = paddle_alts

    return OCRResult(
        text=" ".join(texts),
        words=merged_words,
        confidence=page_conf,
        page_index=page_index,
        source="ensemble",
        raw=raw,
    )


def _empty_result(page_index: int, crop_bbox) -> OCRResult:
    raw = {}
    if crop_bbox:
        raw["crop_bbox_in_page"] = list(crop_bbox)
    return OCRResult(text="", words=[], confidence=0.0,
                     page_index=page_index, source="ensemble", raw=raw)


def _with_crop_raw(result: OCRResult, engine_results: dict, crop_bbox) -> OCRResult:
    raw = {"engines": {k: v.model_dump() for k, v in engine_results.items()}}
    if crop_bbox:
        raw["crop_bbox_in_page"] = list(crop_bbox)
    # Pass through paddle_alternatives when paddle was the sole engine
    if "paddle" in engine_results:
        paddle_alts = engine_results["paddle"].raw.get("paddle_alternatives", {})
        if paddle_alts:
            raw["paddle_alternatives"] = paddle_alts
    return OCRResult(
        text=result.text, words=result.words, confidence=result.confidence,
        page_index=result.page_index, source=result.source, raw=raw,
    )
