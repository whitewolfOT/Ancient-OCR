"""Cross-engine token clustering combining bbox IoU and string similarity."""

from __future__ import annotations

from dataclasses import dataclass, field

from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TokenCluster:
    tokens: list = field(default_factory=list)  # list[WordToken]
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    agreement: float = 0.0  # avg pairwise string similarity across engines


def match_tokens(
    lists: list[list],  # one list[WordToken] per engine
    config=None,
) -> list[TokenCluster]:
    """Group tokens from 2-3 engines into clusters.

    Strategy:
    - Primary: bbox IoU (when bboxes are available and non-zero).
    - Secondary: string similarity for tokens whose bboxes don't overlap.
    - Produces one cluster per unique word position across engines.

    Returns clusters ordered by x-position (RTL: descending x).
    """
    from alignment.bbox_alignment import align_by_bbox, iou
    from alignment.string_similarity import similarity

    iou_threshold = 0.3
    sim_threshold = 0.6
    if config is not None:
        iou_threshold = getattr(
            getattr(config, "ocr", None), "ensemble_iou_threshold", iou_threshold
        )

    if not lists or all(not lst for lst in lists):
        return []

    # Flatten into (token, engine_index) pairs
    non_empty = [(lst, i) for i, lst in enumerate(lists) if lst]
    if len(non_empty) == 1:
        return [
            TokenCluster(tokens=[t], bbox=t.bbox, agreement=1.0)
            for t in non_empty[0][0]
        ]

    # Start with the first engine's tokens as seeds
    base_tokens, _ = non_empty[0]
    clusters: list[TokenCluster] = [
        TokenCluster(tokens=[t], bbox=t.bbox, agreement=1.0)
        for t in base_tokens
    ]

    # Merge each subsequent engine into existing clusters
    for other_tokens, _ in non_empty[1:]:
        cluster_rep = [c.tokens[0] for c in clusters]
        pairs = align_by_bbox(cluster_rep, other_tokens, iou_threshold)

        new_clusters: list[TokenCluster] = []
        matched_other: set[int] = set()

        for i, (ca, cb) in enumerate(pairs):
            if ca is not None and cb is not None:
                # Matched pair — extend the cluster
                clusters[i].tokens.append(cb)
                matched_other.add(id(cb))
                # Recompute agreement as mean pairwise similarity
                texts = [t.text for t in clusters[i].tokens]
                sims = [
                    similarity(texts[a], texts[b])
                    for a in range(len(texts))
                    for b in range(a + 1, len(texts))
                ]
                clusters[i].agreement = float(sum(sims) / len(sims)) if sims else 1.0
                new_clusters.append(clusters[i])
            elif ca is not None:
                new_clusters.append(clusters[i])
            # ca is None → new token from other engine, added below

        # Unmatched tokens from the other engine become new clusters
        for tb in other_tokens:
            if id(tb) not in matched_other:
                new_clusters.append(TokenCluster(tokens=[tb], bbox=tb.bbox, agreement=1.0))

        clusters = new_clusters

    # Sort RTL: descending x
    clusters.sort(key=lambda c: -(c.bbox[0] + c.bbox[2]))
    log.debug(f"match_tokens clusters={len(clusters)}")
    return clusters
