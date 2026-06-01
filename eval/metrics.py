"""Evaluation metrics for Arabic OCR pipeline."""
from __future__ import annotations


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: Levenshtein distance / len(reference).

    Returns 0.0 for identical strings, 1.0 if reference is empty and
    hypothesis is non-empty (fully wrong), and 0.0 if both are empty.
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0

    ref = list(reference)
    hyp = list(hypothesis)
    dist = _levenshtein(ref, hyp)
    return dist / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: Levenshtein distance on word sequences / len(reference words).

    Returns 0.0 for identical strings, 1.0 if reference has words and
    hypothesis is empty, and 0.0 if both are empty.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0

    dist = _levenshtein(ref_words, hyp_words)
    return dist / len(ref_words)


def root_accuracy(ref_roots: list[str], pred_roots: list[str]) -> float:
    """Fraction of predicted roots that match the reference roots.

    Compares position-wise. If lists differ in length, only the overlapping
    prefix is compared. Returns 0.0 if either list is empty.
    """
    if not ref_roots or not pred_roots:
        return 0.0

    n = min(len(ref_roots), len(pred_roots))
    matches = sum(1 for r, p in zip(ref_roots[:n], pred_roots[:n]) if r == p)
    return matches / n


def unresolved_rate(token_states: list) -> float:
    """Fraction of TokenState objects with decision == 'review_required'.

    Returns 0.0 if token_states is empty.
    """
    if not token_states:
        return 0.0
    unresolved = sum(1 for ts in token_states if ts.decision == "review_required")
    return unresolved / len(token_states)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: list, b: list) -> int:
    """Standard dynamic-programming Levenshtein distance over arbitrary sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]
