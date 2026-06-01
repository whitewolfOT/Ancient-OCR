"""Dispatch output formatting by mode."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)


def format_output(
    token_states: list,
    raw_ocr: list,
    mode: str,
    config=None,
) -> dict:
    """Format pipeline results for the requested output mode.

    Modes:
      clean      — corrected Arabic text string only
      annotated  — text + per-token provenance
      debug      — full trace (all raw OCR, scores, candidates)

    Returns a dict with at minimum: {mode, text, word_count, page_count}
    """
    from output.json_export import build_clean, build_annotated
    from output.debug_export import build_debug

    page_count = len(raw_ocr) if raw_ocr else 0
    word_count = len(token_states)

    if mode == "clean":
        text = " ".join(ts.selected for ts in token_states if ts.selected)
        return {
            "mode": "clean",
            "text": text,
            "word_count": word_count,
            "page_count": page_count,
        }

    if mode == "annotated":
        base = build_annotated(token_states)
        base.update({
            "mode": "annotated",
            "word_count": word_count,
            "page_count": page_count,
        })
        return base

    if mode == "debug":
        base = build_debug(token_states, raw_ocr)
        base.update({
            "mode": "debug",
            "word_count": word_count,
            "page_count": page_count,
        })
        return base

    raise ValueError(f"Unknown mode '{mode}'. Must be clean | annotated | debug.")
