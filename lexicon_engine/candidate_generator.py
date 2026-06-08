"""Generate correction candidates for a word token."""

from __future__ import annotations

from confidence_engine.state import Candidate, LexiconEntry
from utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# OCR confusion cost helpers
# ---------------------------------------------------------------------------

# Arabic character classes whose members are visually confused by OCR engines.
# Each class maps to a config key (config.lexicon.confusion_costs.<class>).
_CONFUSION_CLASSES: dict[str, list[str]] = {
    "dot_pairs":      ["ب", "ت", "ث", "ي", "ن"],          # differ only in dot count/position
    "body_pairs":     ["ع", "غ", "ف", "ق"],               # same body silhouette
    "qaf_faa":        ["ق", "ف"],                          # most-confused pair
    "emphatic_pairs": ["س", "ش", "ص", "ض", "ط", "ظ"],     # emphatic vs. non-emphatic
    "tail_pairs":     ["و", "ر", "ز"],                     # similar descenders
    "taa_haa":        ["ت", "ة"],                          # taa marboota confusion
}


def _load_confusion_costs(config=None) -> dict[str, float]:
    """Return confusion costs keyed by class name + 'default' threshold."""
    defaults: dict[str, float] = {k: 0.0 for k in _CONFUSION_CLASSES}
    defaults["default"] = 0.85
    if config is None:
        return defaults
    try:
        cc = config.lexicon.confusion_costs
        return {
            k: float(getattr(cc, k, defaults.get(k, 0.0)))
            for k in list(_CONFUSION_CLASSES) + ["default"]
        }
    except AttributeError:
        return defaults


def _build_confusion_pair_map(costs: dict[str, float]) -> dict[tuple[str, str], float]:
    """Map every (char_a, char_b) confusion pair to its configured weight."""
    pair_map: dict[tuple[str, str], float] = {}
    for class_name, chars in _CONFUSION_CLASSES.items():
        weight = costs.get(class_name, 0.0)
        for i, a in enumerate(chars):
            for b in chars[i + 1:]:
                pair_map[(a, b)] = weight
                pair_map[(b, a)] = weight
    return pair_map


def _confusion_similarity(word: str, lemma: str,
                          pair_map: dict[tuple[str, str], float]) -> float:
    """Levenshtein similarity with confusion-weighted substitution costs.

    A substitution between a known OCR-confused pair costs (1 - weight) instead
    of 1.0, so the effective edit distance shrinks and similarity rises.
    """
    if word == lemma:
        return 1.0
    n, m = len(word), len(lemma)
    if n == 0 or m == 0:
        return 0.0

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if word[i - 1] == lemma[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost = 1.0 - pair_map.get((word[i - 1], lemma[j - 1]), 0.0)
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,          # deletion
                    dp[i][j - 1] + 1.0,           # insertion
                    dp[i - 1][j - 1] + sub_cost,  # substitution
                )

    return 1.0 - dp[n][m] / max(n, m)


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

    # 5. Spelling variants — trigram-narrow first, then confusion-weighted similarity.
    # Full O(N) scan over 40k+ lemmas is unusably slow; skip entirely if no
    # trigram neighbors exist (they will not have useful spelling variants anyway).
    # Confusion costs from config boost similarity for OCR-common char pairs so that
    # e.g. ب↔ن substitutions don't discard valid candidates.
    try:
        max_sv = 5
        if config is not None:
            try:
                max_sv = config.lexicon.max_spelling_variants
            except AttributeError:
                pass
        confusion_costs = _load_confusion_costs(config)
        pair_map = _build_confusion_pair_map(confusion_costs)
        sv_threshold = float(confusion_costs.get("default", 0.85))

        from lexicon_engine.query_engine import _trigram_narrow
        idx = get_index(config)
        narrowed = _trigram_narrow(word, idx.all_lemmas, max_cands=200)
        if narrowed:
            sv_count = 0
            for lemma in narrowed:
                if lemma == word:
                    continue
                csim = _confusion_similarity(word, lemma, pair_map)
                if csim >= sv_threshold:
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
