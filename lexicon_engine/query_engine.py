"""Query the lexicon index for a word — four-tier gate."""

from __future__ import annotations

from confidence_engine.state import LexiconEntry
from utils.logging import get_logger

log = get_logger(__name__)

def _trigrams(text: str) -> set[str]:
    return {text[i:i + 3] for i in range(len(text) - 2)} if len(text) >= 3 else set()


def _trigram_narrow(word: str, all_lemmas: list[str], max_cands: int = 300) -> list[str]:
    """Filter lemmas to those sharing ≥1 trigram with word and within ±3 chars length."""
    w_tg = _trigrams(word)
    wl = len(word)
    if not w_tg:
        return [l for l in all_lemmas if abs(len(l) - wl) <= 1]
    return [
        l for l in all_lemmas
        if abs(len(l) - wl) <= 3 and bool(w_tg & _trigrams(l))
    ][:max_cands]


def query(
    word: str,
    context: list[str] | None = None,
    config=None,
) -> list[LexiconEntry]:
    """Look up *word* in all enabled lexicon sources.

    Three-tier gate:
      Tier 1 — stopword             → return [] immediately
      Tier 2 — persistent cache hit → return cached entries
      Tier 4 — full resolution      → exact → normalized → root → trigram+Levenshtein

    Tier 3 (OCR confidence bypass) lives in main.py, not here — [] would be
    ambiguous between "confident skip" and "genuinely not found."

    Returns list[LexiconEntry] ordered by priority desc.  Always returns [].
    """
    from normalization.arabic_normalizer import normalize_text
    from normalization.stopword_filter import is_stopword

    norm_word, _ = normalize_text(word, config)

    # ── Tier 1: stopword ──────────────────────────────────────────────────────
    if is_stopword(norm_word, config):
        log.debug(f"query tier=1 stopword word={word!r}")
        return []

    # ── Tier 2: persistent cache ──────────────────────────────────────────────
    import utils.cache as _cache_mod
    _cache_ns = "lexicon_query"
    _cache_key = f"{norm_word}:{_cache_mod.config_hash()}"
    cached = _cache_mod.get(_cache_ns, _cache_key)
    if cached is not None:
        log.debug(f"query tier=2 cache_hit word={word!r}")
        return cached

    # ── Tier 4: full resolution ───────────────────────────────────────────────
    from lexicon_ingestion.index_builder import get_index
    from alignment.string_similarity import similarity

    try:
        idx = get_index(config)
    except Exception as exc:
        log.warning(f"index unavailable: {exc}")
        return []

    if idx.is_empty():
        return []

    results: list[LexiconEntry] = []
    seen: set[tuple] = set()

    def _add(entries):
        for e in entries:
            key = (e.lemma, e.source)
            if key not in seen:
                seen.add(key)
                results.append(e)

    # 4a. Exact lemma match
    _add(idx.by_lemma.get(word, []))

    # 4b. Normalized match
    if norm_word != word:
        _add(idx.by_normalized.get(norm_word, []))

    # 4c. Root-based
    if not results:
        try:
            from morphology.root_extractor import extract_root
            cands = extract_root(word, config)
            for rc in cands[:2]:
                _add(idx.by_root.get(rc.root, []))
        except Exception as exc:
            log.debug(f"root-based lookup failed: {exc}")

    # 4d. Trigram-narrowed approximate match (avoids O(n) Levenshtein over full index)
    approx_threshold = 0.8
    if config is not None:
        try:
            approx_threshold = config.lexicon.index.approximate_match_threshold
        except AttributeError:
            pass

    if not results:
        narrowed = _trigram_narrow(norm_word, idx.all_lemmas)
        for lemma in narrowed:
            if similarity(word, lemma) >= approx_threshold:
                _add(idx.by_lemma.get(lemma, []))

    results.sort(key=lambda e: e.priority, reverse=True)
    log.debug(f"query tier=4 word={word!r} results={len(results)}")

    _cache_mod.set(_cache_ns, _cache_key, results)
    return results
