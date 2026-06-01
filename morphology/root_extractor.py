"""Rule-based Arabic root extraction with trilateral-first approach."""

from __future__ import annotations

from dataclasses import dataclass

from utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_PREFIXES = ["الـ", "ال", "وال", "فال", "بال", "لل", "و", "ف", "ب", "ك", "ل"]
_DEFAULT_SUFFIXES = ["ات", "ون", "ين", "ان", "اء", "ية", "ية", "هم", "هن", "كم", "نا", "ها", "ي", "ة", "ه"]


@dataclass
class RootCandidate:
    root: str
    confidence: float
    method: str  # "rule_based" | "camel_tools"


def extract_root(word: str, config=None) -> list[RootCandidate]:
    """Extract root candidates for an Arabic word.

    Returns a list of RootCandidate ordered by descending confidence.
    Always returns at least one candidate (the stripped stem).
    """
    prefixes = _DEFAULT_PREFIXES
    suffixes = _DEFAULT_SUFFIXES

    if config is not None:
        m = getattr(config, "morphology", None)
        if m is not None:
            pl = getattr(m, "prefix_list", None)
            if pl:
                prefixes = list(pl)
            sl = getattr(m, "suffix_list", None)
            if sl:
                suffixes = list(sl)

    candidates: list[RootCandidate] = []

    # Strip prefixes (longest first to avoid partial matches).
    # Single-letter prefixes (و/ف/ب/ك/ل) require >= 4 remaining chars to avoid
    # stripping root-initial letters — the rule-based fallback is conservative.
    stem = word
    for prefix in sorted(prefixes, key=len, reverse=True):
        min_remaining = 4 if len(prefix) == 1 else 2
        if word.startswith(prefix) and len(word) - len(prefix) >= min_remaining:
            stem = word[len(prefix):]
            break

    # Strip suffixes (longest first)
    root_candidate = stem
    for suffix in sorted(suffixes, key=len, reverse=True):
        if stem.endswith(suffix) and len(stem) - len(suffix) >= 2:
            root_candidate = stem[: len(stem) - len(suffix)]
            break

    # Classify by length
    rlen = len(root_candidate)
    if rlen == 3:
        conf = 0.75
    elif rlen == 4:
        conf = 0.65  # quadrilateral root
    elif rlen == 2:
        # Weak/hollow root; attempt reconstruction
        root_candidate = _expand_weak(root_candidate)
        conf = 0.50
    else:
        conf = 0.35

    candidates.append(RootCandidate(root=root_candidate, confidence=conf, method="rule_based"))

    # If stem != root_candidate, also include the raw stem as an alternative
    if stem != root_candidate and len(stem) >= 2:
        candidates.append(RootCandidate(root=stem, confidence=conf * 0.6, method="rule_based"))

    max_cands = 5
    if config is not None:
        m = getattr(config, "morphology", None)
        max_cands = getattr(m, "max_root_candidates", max_cands) if m else max_cands

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    log.debug(f"extract_root word={word!r} top={candidates[0].root if candidates else 'none'}")
    return candidates[:max_cands]


def _expand_weak(stem: str) -> str:
    """Attempt to restore a missing weak letter for 2-letter stems."""
    # Common pattern: و or ي as middle letter
    if len(stem) == 2:
        return stem[0] + "و" + stem[1]
    return stem
