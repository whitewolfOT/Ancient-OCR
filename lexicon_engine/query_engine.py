"""Query the lexicon index for a word."""

from __future__ import annotations

from confidence_engine.state import LexiconEntry
from utils.logging import get_logger

log = get_logger(__name__)


def query(word: str, context: list[str] | None = None, config=None) -> list[LexiconEntry]:
    """Look up a word in all enabled lexicon sources.

    Lookup chain:
      1. Exact lemma match
      2. Normalized-form match
      3. Root-based search (if morphology available)
      4. Approximate match (Levenshtein over index keys)

    Returns a list of LexiconEntry ordered by priority (descending).
    Always returns [] — never None.
    """
    from lexicon_ingestion.index_builder import get_index
    from normalization.arabic_normalizer import normalize_text
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

    # 1. Exact match
    _add(idx.by_lemma.get(word, []))

    # 2. Normalized match
    norm_word, _ = normalize_text(word, config)
    if norm_word != word:
        _add(idx.by_normalized.get(norm_word, []))

    # 3. Root-based
    if not results:
        try:
            from morphology.root_extractor import extract_root
            cands = extract_root(word, config)
            for rc in cands[:2]:
                _add(idx.by_root.get(rc.root, []))
        except Exception as exc:
            log.debug(f"root-based lookup failed: {exc}")

    # 4. Approximate match
    threshold = 0.8
    if config is not None:
        try:
            threshold = config.lexicon.index.approximate_match_threshold
        except AttributeError:
            pass

    if not results:
        for lemma in idx.all_lemmas:
            if similarity(word, lemma) >= threshold:
                _add(idx.by_lemma.get(lemma, []))

    results.sort(key=lambda e: e.priority, reverse=True)
    log.debug(f"query word={word!r} results={len(results)}")
    return results
