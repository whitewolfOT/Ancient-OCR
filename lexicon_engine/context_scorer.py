"""Context scorer — produces context_score for the confidence formula.

N-gram model built lazily from Arabic lemmas and example phrases across all
loaded lexicon entries. English glosses are excluded — the pipeline scores
Arabic tokens against Arabic context, and English bigrams would produce
systematic negative bias when context is wired in. Bigrams seen fewer than
min_bigram_count times are treated as unseen.

Optional: AraBERT masked-LM behind [lm] extra + config flag.

This module MUST always return a float in [0, 1] and must never raise.
"""

from __future__ import annotations

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
    left_context,
    right_context,
    config=None,
) -> float:
    """Score how well candidate_text fits its context.

    left_context / right_context accept list[str] or a bare str for
    convenience (a bare str is wrapped in a one-element list; empty string
    is treated as no context).

    Returns a float in [0, 1]. Never raises.
    Falls back to _FALLBACK_SCORE if context is empty or scoring fails.
    """
    try:
        # Coerce string inputs to lists
        if isinstance(left_context, str):
            left_context = [left_context] if left_context else []
        if isinstance(right_context, str):
            right_context = [right_context] if right_context else []

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

def build_quranic_bigrams(morphology_path: str) -> dict:
    """Build unigram/bigram counts from Quranic morphology file.

    Groups rows by verse (first 3 location parts), collects lemmas in order
    (skipping TAG==P rows and rows with no LEM), and emits consecutive lemma
    pairs within each verse as bigrams.

    Returns {"bigrams": {(lem1, lem2): count}, "unigrams": {lem: count}}.
    """
    from pathlib import Path

    path = Path(morphology_path)
    if not path.exists():
        log.debug(f"build_quranic_bigrams: file not found at {path}; skipping")
        return {"bigrams": {}, "unigrams": {}}

    # verse_key → [lemma, ...]  (in order of token appearance)
    verse_lemmas: dict[str, list[str]] = {}

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                location, _form, tag, features_raw = parts[0], parts[1], parts[2], parts[3]

                if tag == "P":
                    continue

                lemma = ""
                for feat in features_raw.split("|"):
                    if feat.startswith("LEM:"):
                        lemma = feat[4:].strip()
                        break
                if not lemma:
                    continue

                # Verse key: first 3 parts of location (chapter:verse:word_position → chapter:verse)
                loc_parts = location.split(":")
                verse_key = ":".join(loc_parts[:2]) if len(loc_parts) >= 2 else location

                if verse_key not in verse_lemmas:
                    verse_lemmas[verse_key] = []
                verse_lemmas[verse_key].append(lemma)

    except Exception as exc:
        log.warning(f"build_quranic_bigrams: read error: {exc}")
        return {"bigrams": {}, "unigrams": {}}

    unigrams: dict[str, int] = defaultdict(int)
    bigrams: dict[tuple[str, str], int] = defaultdict(int)

    for lemmas in verse_lemmas.values():
        for lem in lemmas:
            unigrams[lem] += 1
        for i in range(len(lemmas) - 1):
            bigrams[(lemmas[i], lemmas[i + 1])] += 1

    log.debug(
        f"build_quranic_bigrams: {len(verse_lemmas)} verses → "
        f"{len(unigrams)} unigrams, {len(bigrams)} bigram types"
    )
    return {"bigrams": dict(bigrams), "unigrams": dict(unigrams)}


def _get_ngram_model(config=None) -> dict:
    """Build or return a cached unigram/bigram model.

    Primary corpus: entry.lemma from all loaded lexicon entries (unigrams).
    Secondary corpus: Quranic verse co-occurrence bigrams from
    config.lexicon.quranic_morphology_path (when the file is present).
    """
    global _ngram_model
    if _ngram_model is not None:
        return _ngram_model

    min_bigram = 2
    if config is not None:
        cs = getattr(config, "context_scorer", None)
        if cs is not None:
            min_bigram = getattr(cs, "min_bigram_count", min_bigram)

    try:
        from lexicon_ingestion.storage import load_entries
        entries = load_entries(config=config)
        corpus: list[str] = []
        for entry in entries:
            corpus.extend(entry.examples)   # Arabic example phrases
            corpus.append(entry.lemma)      # Arabic lemma unigram
    except Exception:
        corpus = []

    unigrams: dict[str, int] = defaultdict(int)
    bigrams_raw: dict[tuple[str, str], int] = defaultdict(int)

    for text in corpus:
        tokens = text.split()
        for tok in tokens:
            unigrams[tok] += 1
        for i in range(len(tokens) - 1):
            bigrams_raw[(tokens[i], tokens[i + 1])] += 1

    # Merge Quranic verse bigrams if the morphology file is configured
    quranic_path = None
    if config is not None:
        try:
            quranic_path = config.lexicon.quranic_morphology_path
        except AttributeError:
            pass
    if quranic_path:
        q = build_quranic_bigrams(quranic_path)
        for lem, cnt in q["unigrams"].items():
            unigrams[lem] += cnt
        for pair, cnt in q["bigrams"].items():
            bigrams_raw[pair] += cnt

    # Filter low-frequency bigrams
    bigrams_filtered = {k: v for k, v in bigrams_raw.items() if v >= min_bigram}

    total_uni = sum(unigrams.values()) or 1
    _ngram_model = {
        "unigrams": dict(unigrams),
        "bigrams": bigrams_filtered,
        "total": total_uni,
        "vocab": len(unigrams),
    }
    log.debug(
        f"ngram model built corpus_sentences={len(corpus)} "
        f"unigrams={len(unigrams)} "
        f"bigram_types_raw={len(bigrams_raw)} "
        f"bigram_types_filtered={len(bigrams_filtered)} "
        f"(min_count={min_bigram})"
    )
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

    if not bigrams:
        # No co-occurrence data — unigram-only mode.
        if candidate in unigrams:
            return 0.6
        return _FALLBACK_SCORE

    # Three-tier hierarchy (mutually exclusive per direction):
    #   1. Known bigram hit  → score proportional to frequency, always > 0.6
    #   2. Known unigram, no bigram context → 0.6
    #   3. Unknown word      → _FALLBACK_SCORE (0.5)
    # Unseen bigrams are "no evidence", not a penalty — Laplace smoothing is
    # wrong here because the corpus is Quranic only; absence means out-of-domain,
    # not incorrectness.
    bigram_scores: list[float] = []

    if left:
        prev = left[-1]
        count_bi = bigrams.get((prev, candidate), 0)
        if count_bi > 0:
            count_prev = unigrams.get(prev, 0) + 1
            p_bi = count_bi / count_prev
            bigram_scores.append(min(p_bi * 5, 0.95))

    if right:
        nxt = right[0]
        count_bi = bigrams.get((candidate, nxt), 0)
        if count_bi > 0:
            count_cand = unigrams.get(candidate, 0) + 1
            p_bi = count_bi / count_cand
            bigram_scores.append(min(p_bi * 5, 0.95))

    if bigram_scores:
        return sum(bigram_scores) / len(bigram_scores)

    if candidate in unigrams:
        return 0.6

    return _FALLBACK_SCORE


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
