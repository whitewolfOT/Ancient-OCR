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
            "review_queue": _review_summary(token_states),
        })
        return base

    if mode == "debug":
        base = build_debug(token_states, raw_ocr)
        base.update({
            "mode": "debug",
            "word_count": word_count,
            "page_count": page_count,
            "review_queue": _review_summary(token_states),
        })
        return base

    raise ValueError(f"Unknown mode '{mode}'. Must be clean | annotated | debug.")


def _review_summary(token_states: list) -> dict:
    """Build a text-level review summary (no image crops) for embedding in annotated/debug output."""
    flagged = [
        ts for ts in token_states
        if ts.decision in ("uncertain", "review_required")
    ]
    by_decision: dict[str, int] = {"uncertain": 0, "review_required": 0}
    for ts in flagged:
        by_decision[ts.decision] = by_decision.get(ts.decision, 0) + 1

    flagged_tokens = [
        {
            "original": ts.original,
            "selected": ts.selected,
            "confidence": ts.confidence,
            "decision": ts.decision,
            "reason_code": ts.reason_code,
            "sources": ts.sources,
        }
        for ts in flagged
    ]

    return {
        "total": len(flagged),
        "by_decision": by_decision,
        "flagged_tokens": flagged_tokens,
    }
