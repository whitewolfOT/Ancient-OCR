"""Generate correction candidates for a word token."""

from __future__ import annotations

from confidence_engine.state import Candidate, LexiconEntry
from utils.logging import get_logger

log = get_logger(__name__)


def generate(
    token,
    morph_result: dict | None,
    config=None,
    ocr_alternatives: list[dict] | None = None,
) -> list[Candidate]:
    """Generate correction candidates for a WordToken.

    Sources:
      - identity: the word as-is (always included)
      - normalization variants: re-query with alef/yaa/taa variants
      - root alternatives: lemmas sharing the same root
      - spelling variants: edit-distance-1 neighbours from index keys

    Returns a list of Candidate with reason set; lexicon_entries populated;
    features left empty (filled by scorer).
    """
    from lexicon_engine.query_engine import query
    from lexicon_ingestion.index_builder import get_index
    from alignment.string_similarity import similarity
    from normalization.arabic_normalizer import normalize_text

    word = token.text
    candidates: list[Candidate] = []
    seen_texts: set[str] = set()

    def _add(text: str, reason: str, entries: list[LexiconEntry]):
        if text and text not in seen_texts:
            seen_texts.add(text)
            candidates.append(Candidate(text=text, reason=reason, lexicon_entries=entries))

    # 1. Identity — always first
    identity_entries = query(word, config=config)
    _add(word, "identity", identity_entries)

    # 2. Normalization variants: query the normalized form
    norm_word, _ = normalize_text(word, config)
    if norm_word != word:
        norm_entries = query(norm_word, config=config)
        _add(norm_word, "normalization", norm_entries)

    # 3. Root-based alternatives
    try:
        from morphology.root_extractor import extract_root
        idx = get_index(config)
        root_cands = extract_root(word, config)
        for rc in root_cands[:2]:
            for entry in idx.by_root.get(rc.root, [])[:5]:
                _add(entry.lemma, "root_alt", [entry])
    except Exception as exc:
        log.debug(f"root alt generation failed: {exc}")

    # 4. Morphology-informed alternatives from CAMeL result
    if morph_result:
        lemma = morph_result.get("lemma")
        root = morph_result.get("root")
        if lemma and lemma != word:
            lemma_entries = query(lemma, config=config)
            _add(lemma, "morph_alt", lemma_entries)
        if root:
            try:
                idx = get_index(config)
                for entry in idx.by_root.get(root, [])[:3]:
                    _add(entry.lemma, "morph_alt", [entry])
            except Exception:
                pass

    # 5. Spelling variants — trigram-narrow first, then similarity on the reduced set.
    # Full O(N) scan over 40k+ lemmas is unusably slow; skip entirely if no
    # trigram neighbors exist (they will not have useful spelling variants anyway).
    try:
        max_sv = 5
        if config is not None:
            try:
                max_sv = config.lexicon.max_spelling_variants
            except AttributeError:
                pass
        from lexicon_engine.query_engine import _trigram_narrow
        idx = get_index(config)
        narrowed = _trigram_narrow(word, idx.all_lemmas, max_cands=200)
        if narrowed:
            sv_count = 0
            for lemma in narrowed:
                if lemma == word:
                    continue
                if similarity(word, lemma) >= 0.85:
                    sv_entries = query(lemma, config=config)
                    _add(lemma, "spelling_variant", sv_entries)
                    sv_count += 1
                    if sv_count >= max_sv:
                        break
    except Exception as exc:
        log.debug(f"spelling variant search failed: {exc}")

    # 6. OCR confusion-pair alternatives from paddle multi-hypothesis
    for alt in (ocr_alternatives or []):
        alt_text = alt.get("text", "")
        if alt_text:
            alt_entries = query(alt_text, config=config)
            _add(alt_text, "ocr_alternative", alt_entries)

    log.debug(f"generate word={word!r} candidates={len(candidates)}")
    return candidates
