"""Arabic wazn (morphological pattern) detection."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

# Common classical Arabic wazn patterns mapped to their descriptions
_PATTERNS: list[tuple[str, str]] = [
    # Verb patterns
    ("فَعَلَ",   "Form I verb"),
    ("فَعِلَ",   "Form I verb (middle kasra)"),
    ("فَعُلَ",   "Form I verb (middle damma)"),
    ("فَاعَلَ",  "Form III verb"),
    ("أَفْعَلَ", "Form IV verb"),
    ("فَعَّلَ",  "Form II verb"),
    ("تَفَعَّلَ","Form V verb"),
    ("تَفَاعَلَ","Form VI verb"),
    ("انْفَعَلَ","Form VII verb"),
    ("افْتَعَلَ","Form VIII verb"),
    ("اسْتَفْعَلَ","Form X verb"),
    # Noun/adjective patterns
    ("فَعْل",    "masdar / simple noun"),
    ("فِعْل",    "noun (kasra)"),
    ("فُعْل",    "noun (damma)"),
    ("فَعَل",    "noun (fathatayn)"),
    ("فَعِل",    "adjective"),
    ("فَعُل",    "adjective (damma)"),
    ("فَاعِل",   "active participle"),
    ("مَفْعُول", "passive participle"),
    ("فِعَال",   "noun / plural pattern"),
    ("فُعَال",   "noun pattern"),
    ("فَعَال",   "noun / masdar pattern"),
    ("فَعِيل",   "adjective / noun"),
    ("فَعَّال",  "intensified agent"),
    ("مِفْعَال", "instrument noun"),
    ("مَفْعَل",  "noun of place/time"),
    ("مَفْعِل",  "noun of place/time"),
    ("فُعَلاء",  "broken plural"),
    ("أَفْعَال", "broken plural"),
    ("فُعُول",   "broken plural"),
    ("فِعَلة",   "broken plural"),
    ("فُعْلة",   "noun (unity)"),
    ("فَعْلة",   "noun (unity / instance)"),
    ("فِعَالة",  "trade / occupation noun"),
    ("فَاعِلة",  "active participle (fem)"),
    ("فَعِيلة",  "noun (fem)"),
    ("تَفْعِيل", "masdar Form II"),
    ("فَعْلِيَّة","abstract noun (modern)"),
    ("مُفَعِّل", "Form II active participle"),
    ("مُفَاعِل", "Form III active participle"),
    ("مُفْعِل",  "Form IV active participle"),
]


def detect_pattern(word: str, root: str | None = None) -> dict:
    """Detect the morphological wazn pattern of a word.

    Uses a heuristic: match the word's length and letter positions against
    known patterns. Returns the best match or None if unrecognised.

    Returns:
        {pattern: str | None, confidence: float, wazn: str | None, description: str}
    """
    # Remove diacritics for matching
    clean = _strip_diacritics(word)
    rlen = len(clean)

    best_pattern = None
    best_wazn = None
    best_conf = 0.0
    best_desc = ""

    for wazn, desc in _PATTERNS:
        clean_wazn = _strip_diacritics(wazn)
        if len(clean_wazn) != rlen:
            continue

        # Length match: score by position matches against ف/ع/ل skeleton
        score = _skeleton_score(clean, clean_wazn, root)
        if score > best_conf:
            best_conf = score
            best_pattern = wazn
            best_wazn = wazn
            best_desc = desc

    log.debug(f"detect_pattern word={word!r} pattern={best_wazn!r} conf={best_conf:.2f}")
    return {
        "pattern": best_pattern,
        "confidence": best_conf,
        "wazn": best_wazn,
        "description": best_desc,
    }


def list_patterns() -> list[str]:
    """Return all known wazn patterns."""
    return [p for p, _ in _PATTERNS]


def _skeleton_score(word: str, wazn: str, root: str | None) -> float:
    """Score how well a word fits a wazn given an optional root."""
    if not root:
        # Without root, score purely on length match (weak signal)
        return 0.30 if len(word) == len(wazn) else 0.0

    root_letters = list(root[:3])  # take trilateral core
    wazn_root_positions = [i for i, c in enumerate(wazn) if c in "فعل"]

    if len(wazn_root_positions) < len(root_letters):
        return 0.20

    matches = 0
    for pos, rl in zip(wazn_root_positions, root_letters):
        if pos < len(word) and word[pos] == rl:
            matches += 1

    return matches / max(len(root_letters), 1)


def _strip_diacritics(text: str) -> str:
    import re
    return re.sub(r"[ً-ٰٟ]", "", text)
