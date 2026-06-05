"""JSON export helpers for clean and annotated modes."""

from __future__ import annotations

import json


def build_clean(token_states: list) -> dict:
    text = " ".join(ts.selected for ts in token_states if ts.selected)
    return {"text": text}


def build_annotated(token_states: list) -> dict:
    text = " ".join(ts.selected for ts in token_states if ts.selected)
    tokens = []
    for ts in token_states:
        tokens.append({
            "original": ts.original,
            "selected": ts.selected,
            "confidence": ts.confidence,
            "decision": ts.decision,
            "reason_code": ts.reason_code,
            "sources": ts.sources,
            "bbox": list(ts.bbox) if ts.bbox else None,
            "alternatives": [
                {"text": c.text, "score": c.score, "reason": c.reason}
                for c in (ts.candidates or [])
                if c.text != ts.selected
            ][:5],
        })
    return {"text": text, "tokens": tokens}


def to_json(data: dict, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=indent)
