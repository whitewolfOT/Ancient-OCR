"""Full pipeline trace for debug mode."""

from __future__ import annotations


def build_debug(token_states: list, raw_ocr: list) -> dict:
    """Build the full debug structure.

    Per-token trace includes: raw per-engine OCR, normalization deltas,
    all scored candidates, decision breakdown.
    """
    token_traces = []
    for ts in token_states:
        candidates_trace = []
        for c in (ts.candidates or []):
            candidates_trace.append({
                "text": c.text,
                "reason": c.reason,
                "score": c.score,
                "features": c.features,
                "lexicon_entries": [
                    {
                        "lemma": e.lemma,
                        "root": e.root,
                        "gloss": e.gloss,
                        "source": e.source,
                        "era": e.era,
                        "priority": e.priority,
                    }
                    for e in (c.lexicon_entries or [])
                ],
            })

        token_traces.append({
            "original": ts.original,
            "normalized": ts.normalized,
            "normalization_log": ts.normalization_log,
            "selected": ts.selected,
            "confidence": ts.confidence,
            "decision": ts.decision,
            "reason_code": ts.reason_code,
            "sources": ts.sources,
            "bbox": list(ts.bbox) if ts.bbox else None,
            "page_index": ts.page_index,
            "candidates": candidates_trace,
        })

    raw_ocr_trace = []
    for page_result in (raw_ocr or []):
        engines = page_result.raw.get("engines", {}) if page_result.raw else {}
        raw_ocr_trace.append({
            "page_index": page_result.page_index,
            "source": page_result.source,
            "confidence": page_result.confidence,
            "text_preview": page_result.text[:200] if page_result.text else "",
            "token_count": len(page_result.words),
            "engines": list(engines.keys()),
        })

    return {
        "text": " ".join(ts.selected for ts in token_states if ts.selected),
        "tokens": token_traces,
        "raw_ocr": raw_ocr_trace,
    }
