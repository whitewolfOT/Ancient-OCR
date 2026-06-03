"""Score each candidate by filling its features dict."""

from __future__ import annotations

from confidence_engine.state import Candidate
from utils.logging import get_logger

log = get_logger(__name__)


def score(
    candidate: Candidate,
    context_pair: tuple[list[str], list[str]],
    ocr_conf: float,
    morph_result=None,
    config=None,
) -> Candidate:
    """Fill candidate.features and set candidate.score.

    Features:
      ocr_score      — from the original WordToken confidence
      lexicon_score  — based on matching entries (priority + era)
      morph_score    — root agreement between morphology output and lexicon entries
      morph_evidence — "confirmed" | "partial" | "none" (human-readable in debug)
      context_score  — from context_scorer

    Returns a new Candidate with features and score set.
    """
    from lexicon_engine.context_scorer import context_score as ctx_score

    left_ctx, right_ctx = context_pair

    # ocr_score: pass through the original token confidence
    ocr_s = max(0.0, min(1.0, float(ocr_conf)))

    # lexicon_score: weighted sum over matching entries
    lexicon_s = _lexicon_score(candidate)

    # morph_score: root agreement between morphology result and lexicon entries
    morph_s, morph_evidence = _morph_score(candidate, morph_result)

    # context_score: from context_scorer
    ctx_s = ctx_score(candidate.text, left_ctx, right_ctx, config)

    features = {
        "ocr_score": round(ocr_s, 4),
        "lexicon_score": round(lexicon_s, 4),
        "morph_score": round(morph_s, 4),
        "morph_evidence": morph_evidence,
        "context_score": round(ctx_s, 4),
    }

    # Weighted composite (weights from config)
    ocr_w = 0.30
    lex_w = 0.30
    mor_w = 0.20
    ctx_w = 0.20
    if config is not None:
        s = getattr(config, "scoring", None)
        if s is not None:
            ocr_w = getattr(s, "ocr_weight", ocr_w)
            lex_w = getattr(s, "lexicon_weight", lex_w)
            mor_w = getattr(s, "morphology_weight", mor_w)
            ctx_w = getattr(s, "context_weight", ctx_w)

    composite = ocr_w * ocr_s + lex_w * lexicon_s + mor_w * morph_s + ctx_w * ctx_s

    log.debug(
        f"score text={candidate.text!r} reason={candidate.reason} "
        f"lex={lexicon_s:.2f} morph={morph_s:.2f} ({morph_evidence}) "
        f"ctx={ctx_s:.2f} total={composite:.3f}"
    )

    return Candidate(
        text=candidate.text,
        reason=candidate.reason,
        lexicon_entries=candidate.lexicon_entries,
        features=features,
        score=round(composite, 4),
    )


def _lexicon_score(candidate: Candidate) -> float:
    if not candidate.lexicon_entries:
        return 0.0

    best = 0.0
    for entry in candidate.lexicon_entries:
        # Priority contributes up to 0.7; era bonus up to 0.3
        pri_score = min(entry.priority / 10.0, 1.0) * 0.7
        era_bonus = 0.3 if entry.era == "classical" else 0.15
        s = pri_score + era_bonus
        best = max(best, min(s, 1.0))
    return best


def _morph_score(candidate: Candidate, morph_result) -> tuple[float, str]:
    """Return (morph_score, morph_evidence) based on root agreement.

    Scores:
      0.5–1.0  root confirmed by morphology  → "confirmed"
      0.4      identity candidate, no confirmation → "partial"
      0.3      morphology available but no root agreement (non-identity) → "none"
      0.3      no morphological information available → "none"
    """
    if morph_result is None:
        return 0.3, "none"

    # Build {root: confidence} from either result shape:
    # rule-based → {"root_candidates": [RootCandidate(root, confidence, method), ...]}
    # CAMeL      → {"root": "كتب", "lemma": ..., ...}
    morph_root_conf: dict[str, float] = {}
    for rc in morph_result.get("root_candidates", []):
        morph_root_conf[rc.root] = max(morph_root_conf.get(rc.root, 0.0), rc.confidence)
    camel_root = morph_result.get("root")
    if camel_root and camel_root not in morph_root_conf:
        morph_root_conf[camel_root] = 0.9  # CAMeL analysis is more reliable than rule-based

    if not morph_root_conf:
        return 0.3, "none"

    candidate_roots = {e.root for e in candidate.lexicon_entries if e.root}
    matching = candidate_roots & morph_root_conf.keys()

    if matching:
        best_conf = max(morph_root_conf[r] for r in matching)
        return 0.5 + 0.5 * best_conf, "confirmed"

    if candidate.reason == "identity":
        return 0.4, "partial"

    return 0.2, "none"
