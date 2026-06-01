"""Bounding-box IoU matching for cross-engine token alignment."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

# (x, y, w, h) tuples throughout — top-left origin, page space


def iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = ix * iy
    if intersection == 0:
        return 0.0

    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def align_by_bbox(
    tokens_a: list,
    tokens_b: list,
    iou_threshold: float = 0.3,
) -> list[tuple]:
    """Match tokens from two engines by bbox overlap (greedy, one-to-one).

    Returns a list of (token_a, token_b | None) pairs. Unmatched tokens
    from tokens_a appear with None. tokens_b tokens with no match are
    appended as (None, token_b) at the end.
    """
    if not tokens_a:
        return [(None, tb) for tb in tokens_b]
    if not tokens_b:
        return [(ta, None) for ta in tokens_a]

    matched_b: set[int] = set()
    pairs: list[tuple] = []

    for ta in tokens_a:
        best_j, best_score = -1, 0.0
        for j, tb in enumerate(tokens_b):
            if j in matched_b:
                continue
            score = iou(ta.bbox, tb.bbox)
            if score > best_score:
                best_score, best_j = score, j
        if best_j >= 0 and best_score >= iou_threshold:
            matched_b.add(best_j)
            pairs.append((ta, tokens_b[best_j]))
        else:
            pairs.append((ta, None))

    for j, tb in enumerate(tokens_b):
        if j not in matched_b:
            pairs.append((None, tb))

    return pairs
