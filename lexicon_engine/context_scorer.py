"""Context scorer — produces context_score for the confidence formula.

Default: deterministic bigram frequency over the fixture lexicon examples.
Optional: AraBERT masked-LM behind [lm] extra + config flag.

This module MUST always return a float in [0, 1] and must never raise.
"""

from __future__ import annotations

import math
from collections import defaultdict

from utils.logging import get_logger

log = get_logger(__name__)

_FALLBACK_SCORE = 0.5  # neutral; preserves 20% weight without zeroing it

_ngram_model: dict | None = None
_arabert_available = False

try:
    import transformers as _transformers  # noqa: F401
    _arabert_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def context_score(
    candidate_text: str,
    left_context: list[str],
    right_context: list[str],
    config=None,
) -> float:
    """Score how well candidate_text fits its context.

    Returns a float in [0, 1]. Never raises.
    Falls back to _FALLBACK_SCORE if context is empty or scoring fails.
    """
    try:
        backend = "ngram"
        fallback = _FALLBACK_SCORE
        if config is not None:
            cs = getattr(config, "context_scorer", None)
            if cs is not None:
                backend = getattr(cs, "backend", backend)
                fallback = getattr(cs, "fallback_score", fallback)

        if not left_context and not right_context:
            return fallback

        if backend == "arabert" and _arabert_available:
            score = _arabert_score(candidate_text, left_context, right_context, config)
        else:
            score = _ngram_score(candidate_text, left_context, right_context, config)

        return max(0.0, min(1.0, score))
    except Exception as exc:
        log.warning(f"context_score failed ({exc}); returning fallback={_FALLBACK_SCORE}")
        return _FALLBACK_SCORE


# ---------------------------------------------------------------------------
# N-gram scorer
# ---------------------------------------------------------------------------

def _get_ngram_model(config=None) -> dict:
    """Build or return a cached unigram/bigram model from fixture examples."""
    global _ngram_model
    if _ngram_model is not None:
        return _ngram_model

    try:
        from lexicon_ingestion.storage import load_entries
        entries = load_entries(config=config)
        corpus: list[str] = []
        for entry in entries:
            corpus.extend(entry.examples)
            corpus.append(entry.lemma)
    except Exception:
        corpus = []

    unigrams: dict[str, int] = defaultdict(int)
    bigrams: dict[tuple[str, str], int] = defaultdict(int)

    for text in corpus:
        tokens = text.split()
        for tok in tokens:
            unigrams[tok] += 1
        for i in range(len(tokens) - 1):
            bigrams[(tokens[i], tokens[i + 1])] += 1

    total_uni = sum(unigrams.values()) or 1
    _ngram_model = {
        "unigrams": dict(unigrams),
        "bigrams": dict(bigrams),
        "total": total_uni,
        "vocab": len(unigrams),
    }
    log.debug(f"ngram model built unigrams={total_uni} bigram_types={len(bigrams)}")
    return _ngram_model


def _ngram_score(
    candidate: str,
    left: list[str],
    right: list[str],
    config=None,
) -> float:
    model = _get_ngram_model(config)
    unigrams = model["unigrams"]
    bigrams = model["bigrams"]
    total = model["total"]
    vocab = model["vocab"] or 1

    scores: list[float] = []

    # Bigram P(candidate | left[-1]) with Laplace smoothing
    if left:
        prev = left[-1]
        count_bi = bigrams.get((prev, candidate), 0)
        count_prev = unigrams.get(prev, 0)
        p_bi = (count_bi + 1) / (count_prev + vocab)
        scores.append(p_bi)

    # Bigram P(right[0] | candidate)
    if right:
        nxt = right[0]
        count_bi = bigrams.get((candidate, nxt), 0)
        count_cand = unigrams.get(candidate, 0)
        p_bi = (count_bi + 1) / (count_cand + vocab)
        scores.append(p_bi)

    # Unigram presence bonus
    if candidate in unigrams:
        scores.append(0.7)

    if not scores:
        return _FALLBACK_SCORE

    # Geometric mean, then scale to [0.2, 0.9]
    log_sum = sum(math.log(max(s, 1e-9)) for s in scores)
    geo_mean = math.exp(log_sum / len(scores))
    # Map (0, 1] → (0.2, 0.9] so even unknown context doesn't zero out
    scaled = 0.2 + 0.7 * min(geo_mean * 100, 1.0)
    return scaled


# ---------------------------------------------------------------------------
# AraBERT scorer (optional [lm] extra)
# ---------------------------------------------------------------------------

def _arabert_score(
    candidate: str,
    left: list[str],
    right: list[str],
    config=None,
) -> float:
    try:
        model_id = None
        if config is not None:
            cs = getattr(config, "context_scorer", None)
            if cs is not None:
                model_id = getattr(cs, "arabert_model", None)

        if not model_id:
            return _ngram_score(candidate, left, right, config)

        from transformers import pipeline
        context = " ".join(left) + " [MASK] " + " ".join(right)
        fill = pipeline("fill-mask", model=model_id)
        results = fill(context, top_k=50)
        for r in results:
            if r["token_str"].strip() == candidate:
                return float(r["score"])
        return 0.1  # candidate not in top-50
    except Exception as exc:
        log.warning(f"arabert scoring failed: {exc}; falling back to ngram")
        return _ngram_score(candidate, left, right, config)
