"""Pipeline orchestrator — wire-up only, no business logic."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _require(module: str) -> Any:
    """Import a module or raise a clear RuntimeError naming the missing piece."""
    try:
        import importlib
        return importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing module '{module}'. Install dependencies or check build phase order. "
            f"Original error: {exc}"
        ) from exc


def process_file(file_path: str, mode: str = "clean") -> dict:
    """Top-level entry point: ingest one file and return formatted output.

    Pipeline order (canonical — must match MASTER_PLAN §1):
      load_document → preprocess → crop_regions → ensemble_ocr_per_region →
      stitch_to_page_space → normalize → morph_analyze →
      lexicon_query + candidate_generate + score + rank →
      confidence + decide → format + review_export
    """
    from utils.config import get_config
    from utils.logging import get_logger

    cfg = get_config()
    log = get_logger(__name__)
    log.info(f"process_file path={file_path} mode={mode}")

    if mode not in ("clean", "annotated", "debug"):
        raise ValueError(f"Invalid mode '{mode}'. Must be clean | annotated | debug.")

    # --- Stage 1: document loading ---
    doc_loader = _require("ingest.document_loader")
    pages = doc_loader.load_document(file_path)
    log.info(f"loaded page_count={len(pages)}")

    return run_pipeline(pages, mode=mode, cfg=cfg)


def run_pipeline(pages: list, mode: str = "clean", cfg=None) -> dict:
    """Run the full pipeline over already-loaded pages.

    Each stage is imported inside _require() so that a missing module produces
    a clear error rather than a bare ImportError at startup.
    """
    from utils.logging import get_logger

    if cfg is None:
        from utils.config import get_config
        cfg = get_config()

    log = get_logger(__name__)

    preprocessing = _require("preprocessing.image_pipeline")
    region_cropper = _require("ingest.region_cropper")
    layout_detection = _require("preprocessing.layout_detection")
    ensemble = _require("ocr_engine.ensemble")
    noise_filter = _require("normalization.noise_filter")
    arabic_normalizer = _require("normalization.arabic_normalizer")
    stopword_filter = _require("normalization.stopword_filter")
    camel_adapter = _require("morphology.camel_adapter")
    root_extractor = _require("morphology.root_extractor")
    query_engine = _require("lexicon_engine.query_engine")
    candidate_generator = _require("lexicon_engine.candidate_generator")
    scorer_mod = _require("lexicon_engine.scorer")
    ranker_mod = _require("lexicon_engine.ranker")
    confidence_scoring = _require("confidence_engine.scoring")
    decision_mod = _require("confidence_engine.decision")
    state_mod = _require("confidence_engine.state")
    formatter = _require("output.formatter")

    _accept_threshold = 0.90
    try:
        _accept_threshold = cfg.decision.accept
    except AttributeError:
        pass

    all_token_states: list = []
    all_raw_ocr: list = []

    for page in pages:
        image = page["image"]
        page_index = page["page_index"]

        # Stage 2: preprocess
        processed_image, preprocess_meta = preprocessing.preprocess_image(image, cfg)
        log.debug(f"preprocess page={page_index} steps={list(preprocess_meta.keys())}")

        # Stage 3: layout detection + region cropping
        regions = layout_detection.detect_layout(processed_image, cfg)
        crops = region_cropper.crop_regions(processed_image, regions)

        # Stage 4: OCR per region + stitch
        region_results = []
        for crop in crops:
            ocr_result = ensemble.run_ensemble(crop["image"], page_index, cfg)
            # Translate crop-space bboxes to page-space before stitching
            region_results.append((crop, ocr_result))

        page_ocr = region_cropper.stitch_results(
            [r for _, r in region_results], page_index
        )
        all_raw_ocr.append(page_ocr)

        # Stages 5–8: per-token processing (index-based for phrase lookahead)
        words = list(page_ocr.words)
        i = 0
        while i < len(words):
            word_token = words[i]

            # Stage 5: normalization
            _, noise_log = noise_filter.clean_noise(word_token.text)
            normalized_text, norm_log = arabic_normalizer.normalize_text(
                word_token.text, cfg
            )
            full_norm_log = noise_log + norm_log

            # ── Tier 1a: phrase stopword — consume window, skip pipeline ──────
            phrase_consumed = False
            for window in range(stopword_filter.MAX_PHRASE_WORDS, 1, -1):
                if i + window > len(words):
                    continue
                phrase_tokens = words[i:i + window]
                phrase_text = " ".join(
                    [normalized_text] + [
                        arabic_normalizer.normalize_text(pt.text, cfg)[0]
                        for pt in phrase_tokens[1:]
                    ]
                )
                if stopword_filter.is_stopword_phrase(phrase_text, cfg):
                    for j, pt in enumerate(phrase_tokens):
                        if j == 0:
                            pt_norm, pt_norm_log, pt_noise_log = (
                                normalized_text, norm_log, noise_log
                            )
                        else:
                            _, pt_noise_log = noise_filter.clean_noise(pt.text)
                            pt_norm, pt_norm_log = arabic_normalizer.normalize_text(
                                pt.text, cfg
                            )
                        all_token_states.append(state_mod.TokenState(
                            original=pt.text,
                            normalized=pt_norm,
                            normalization_log=pt_noise_log + pt_norm_log,
                            candidates=[],
                            selected=pt_norm,
                            confidence=1.0,
                            sources=[],
                            decision="accept",
                            reason_code="stopword_phrase",
                            bbox=pt.bbox,
                            page_index=page_index,
                        ))
                    i += window
                    phrase_consumed = True
                    break
            if phrase_consumed:
                continue

            # ── Tier 1b: single-token stopword — skip morphology and lexicon ──
            if stopword_filter.is_stopword(normalized_text, cfg):
                all_token_states.append(state_mod.TokenState(
                    original=word_token.text,
                    normalized=normalized_text,
                    normalization_log=full_norm_log,
                    candidates=[],
                    selected=normalized_text,
                    confidence=1.0,
                    sources=[],
                    decision="accept",
                    reason_code="stopword",
                    bbox=word_token.bbox,
                    page_index=page_index,
                ))
                i += 1
                continue

            # ── Tier 3: high OCR confidence — trust the engine, skip pipeline ─
            if word_token.confidence >= _accept_threshold:
                all_token_states.append(state_mod.TokenState(
                    original=word_token.text,
                    normalized=normalized_text,
                    normalization_log=full_norm_log,
                    candidates=[],
                    selected=normalized_text,
                    confidence=word_token.confidence,
                    sources=[],
                    decision="accept",
                    reason_code="ocr_confident",
                    bbox=word_token.bbox,
                    page_index=page_index,
                ))
                i += 1
                continue

            # ── Tier 4: full resolution — morphology + lexicon + scoring ──────

            # Stage 6: morphological analysis
            morph_result = camel_adapter.analyze(normalized_text)
            if morph_result is None:
                root_candidates = root_extractor.extract_root(normalized_text)
                morph_result = {"root_candidates": root_candidates}

            # Stage 7: lexicon + candidate pipeline
            candidates = candidate_generator.generate(word_token, morph_result, cfg)
            left_ctx = [all_token_states[-1].normalized] if all_token_states else []
            right_ctx = []  # right context requires two-pass processing — not yet implemented
            scored = [
                scorer_mod.score(c, (left_ctx, right_ctx), word_token.confidence, cfg)
                for c in candidates
            ]
            ranked = ranker_mod.rank(scored)

            # Stage 8: confidence + decision
            features = ranked.best.features if ranked.best else {}
            conf = confidence_scoring.final_confidence(features, cfg)
            decision_label, reason_code = decision_mod.decide(conf, cfg)

            token_state = state_mod.build_token_state(
                original=word_token.text,
                normalized=normalized_text,
                norm_log=full_norm_log,
                candidates=scored,
                ranked_result=ranked,
                conf=conf,
                decision=decision_label,
                reason_code=reason_code,
                bbox=word_token.bbox,
                page_index=page_index,
            )
            all_token_states.append(token_state)
            i += 1

    # Stage 9: format output
    result = formatter.format_output(all_token_states, all_raw_ocr, mode, cfg)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <file_path> [mode]")
        print("  mode: clean (default) | annotated | debug")
        sys.exit(0)

    file_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "clean"

    try:
        output = process_file(file_path, mode)
        import json
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
