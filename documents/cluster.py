"""pHash-based page clustering for document calibration.

Algorithm:
  1. Compute pHash (64-bit perceptual hash) for each page image.
  2. Build a graph: pages are nodes; an edge exists when Hamming distance <= 10.
  3. Find connected components — each component becomes a cluster.
  4. Identify the medoid of each cluster: the page with the lowest average
     Hamming distance to all other pages in the same cluster.

pHash bit-width = 64, so max Hamming distance = 64.
Similarity to representative = 1.0 - (hamming_dist / 64).
"""
from __future__ import annotations

import uuid
from collections import deque


_PHASH_BITS = 64          # standard pHash hash_size=8 → 8×8=64 bits
_CLUSTER_THRESHOLD = 10   # pages with Hamming dist ≤ this share a cluster


def compute_phash_str(image_path: str) -> str:
    """Return the pHash of an image file as a hex string."""
    from PIL import Image
    import imagehash

    img = Image.open(image_path).convert("RGB")
    h = imagehash.phash(img)
    return str(h)


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two pHash hex strings."""
    import imagehash
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def build_clusters(
    pages: list[dict],
    threshold: int = _CLUSTER_THRESHOLD,
) -> list[dict]:
    """Cluster pages by pHash similarity using connected components.

    Args:
        pages: list of {"page_id": str, "phash": str}
        threshold: max Hamming distance for two pages to be in the same cluster

    Returns:
        list of {
            "cluster_id": str,
            "label": str,            # "Cluster 1", "Cluster 2", ...
            "page_ids": list[str],
            "representative_page_id": str,
        }
    """
    n = len(pages)
    if n == 0:
        return []

    # Precompute pairwise distances (n is small — at most a few hundred pages)
    # distances[i][j] = Hamming distance between pages[i] and pages[j]
    import imagehash
    hashes = [imagehash.hex_to_hash(p["phash"]) for p in pages]
    distances: list[list[int]] = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = hashes[i] - hashes[j]
            distances[i][j] = d
            distances[j][i] = d

    # Connected components via BFS
    visited = [False] * n
    components: list[list[int]] = []

    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        queue: deque[int] = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in range(n):
                if not visited[neighbor] and distances[node][neighbor] <= threshold:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)

    clusters: list[dict] = []
    for idx, component in enumerate(components):
        cluster_id = str(uuid.uuid4())
        label = f"Cluster {idx + 1}"

        # Medoid: page with lowest average distance to all others in the cluster
        if len(component) == 1:
            rep_idx = component[0]
        else:
            best_avg = float("inf")
            rep_idx = component[0]
            for i in component:
                avg = sum(distances[i][j] for j in component if j != i) / (len(component) - 1)
                if avg < best_avg:
                    best_avg = avg
                    rep_idx = i

        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "page_ids": [pages[i]["page_id"] for i in component],
                "representative_page_id": pages[rep_idx]["page_id"],
            }
        )

    return clusters


def similarity_to_rep(page_phash: str, rep_phash: str) -> float:
    """Similarity score (0–1) between a page and its cluster representative."""
    if page_phash == rep_phash:
        return 1.0
    dist = hamming_distance(page_phash, rep_phash)
    return round(max(0.0, 1.0 - dist / _PHASH_BITS), 4)
