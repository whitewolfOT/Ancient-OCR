"""String similarity utilities for Arabic OCR token matching."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)


def similarity(a: str, b: str) -> float:
    """Return normalised edit-distance similarity in [0, 1].

    1.0 = identical, 0.0 = completely different.
    Falls back to a pure-Python Levenshtein if python-Levenshtein is absent.
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    try:
        from Levenshtein import ratio as lev_ratio
        return float(lev_ratio(a, b))
    except ImportError:
        return _python_ratio(a, b)


def _python_ratio(a: str, b: str) -> float:
    """Pure-Python fallback: 2 * matches / total_chars."""
    dist = _levenshtein(a, b)
    total = len(a) + len(b)
    return 1.0 - dist / total if total else 1.0


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]
