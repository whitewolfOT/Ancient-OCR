"""Weighted confidence scoring formula."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)


def final_confidence(features: dict, config=None) -> float:
    """Compute the final confidence score from feature components.

    Formula: 0.30*ocr + 0.30*lexicon + 0.20*morphology + 0.20*context
    All components are clamped to [0, 1] before weighting.
    """
    ocr_w, lex_w, mor_w, ctx_w = 0.30, 0.30, 0.20, 0.20

    if config is not None:
        s = getattr(config, "scoring", None)
        if s is not None:
            ocr_w = getattr(s, "ocr_weight", ocr_w)
            lex_w = getattr(s, "lexicon_weight", lex_w)
            mor_w = getattr(s, "morphology_weight", mor_w)
            ctx_w = getattr(s, "context_weight", ctx_w)

    def _clamp(v) -> float:
        return max(0.0, min(1.0, float(v))) if v is not None else 0.0

    ocr = _clamp(features.get("ocr_score"))
    lex = _clamp(features.get("lexicon_score"))
    mor = _clamp(features.get("morph_score"))
    ctx = _clamp(features.get("context_score"))

    result = ocr_w * ocr + lex_w * lex + mor_w * mor + ctx_w * ctx
    log.debug(
        f"final_confidence ocr={ocr:.2f} lex={lex:.2f} mor={mor:.2f} ctx={ctx:.2f} "
        f"→ {result:.4f}"
    )
    return round(result, 4)


def breakdown(features: dict, config=None) -> dict:
    """Return per-component weighted contributions (for debug output)."""
    ocr_w, lex_w, mor_w, ctx_w = 0.30, 0.30, 0.20, 0.20
    if config is not None:
        s = getattr(config, "scoring", None)
        if s is not None:
            ocr_w = getattr(s, "ocr_weight", ocr_w)
            lex_w = getattr(s, "lexicon_weight", lex_w)
            mor_w = getattr(s, "morphology_weight", mor_w)
            ctx_w = getattr(s, "context_weight", ctx_w)

    def _clamp(v):
        return max(0.0, min(1.0, float(v))) if v is not None else 0.0

    return {
        "ocr_contribution":     round(ocr_w * _clamp(features.get("ocr_score")), 4),
        "lexicon_contribution": round(lex_w * _clamp(features.get("lexicon_score")), 4),
        "morph_contribution":   round(mor_w * _clamp(features.get("morph_score")), 4),
        "context_contribution": round(ctx_w * _clamp(features.get("context_score")), 4),
        "weights": {"ocr": ocr_w, "lexicon": lex_w, "morphology": mor_w, "context": ctx_w},
    }
