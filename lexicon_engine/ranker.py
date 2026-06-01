"""Rank scored candidates; identity wins tiebreaks."""

from __future__ import annotations

from confidence_engine.state import Candidate, RankedResult
from utils.logging import get_logger

log = get_logger(__name__)


def rank(candidates: list[Candidate]) -> RankedResult:
    """Sort candidates by score (descending); identity wins tiebreaks.

    Returns a RankedResult with best candidate and full ordering.
    If candidates is empty, returns an empty RankedResult.
    """
    if not candidates:
        return RankedResult(best=None, ranked=[], selected_text="")

    def _sort_key(c: Candidate):
        score = c.score if c.score is not None else 0.0
        # Identity wins ties — use a secondary sort flag
        identity_bonus = 1 if c.reason == "identity" else 0
        return (score, identity_bonus)

    ranked = sorted(candidates, key=_sort_key, reverse=True)
    best = ranked[0]

    log.debug(
        f"rank candidates={len(ranked)} best={best.text!r} "
        f"reason={best.reason} score={best.score}"
    )

    return RankedResult(
        best=best,
        ranked=ranked,
        selected_text=best.text,
    )
