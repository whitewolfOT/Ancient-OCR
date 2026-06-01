"""Layout detection: column/region segmentation via projection profiles (default)
or layoutparser (optional [layout] extra)."""

from __future__ import annotations

import uuid

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def detect_layout(image: np.ndarray, config=None) -> list[dict]:
    """Detect text regions in a page image.

    Default backend: pure-OpenCV projection-profile column detection.
    Optional backend: layoutparser (requires [layout] extra and config).

    Returns a list of region dicts:
        {x, y, w, h, type: str, region_id: str}

    Arabic text is RTL — regions are ordered right-to-left.
    If no regions are detected, returns one region covering the full page.
    """
    backend = "opencv"
    enabled = True
    if config is not None:
        lay = getattr(getattr(config, "preprocessing", None), "layout", None)
        if lay is not None:
            enabled = getattr(lay, "enabled", True)
            backend = getattr(lay, "backend", "opencv")

    if not enabled:
        return [_full_page_region(image)]

    if backend == "layoutparser":
        regions = _try_layoutparser(image, config)
        if regions is not None:
            return regions
        log.warning("layoutparser unavailable; falling back to opencv")

    return _opencv_projection(image)


# ---------------------------------------------------------------------------
# OpenCV projection-profile detector
# ---------------------------------------------------------------------------

def _opencv_projection(image: np.ndarray) -> list[dict]:
    try:
        import cv2
    except ImportError:
        log.warning("opencv not available; returning full-page region")
        return [_full_page_region(image)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal projection: find rows with significant ink
    row_sums = binary.sum(axis=1)
    row_threshold = row_sums.max() * 0.05
    text_rows = row_sums > row_threshold

    # Group consecutive text rows into vertical bands
    bands = _group_runs(text_rows)
    if not bands:
        return [_full_page_region(image)]

    h_img, w_img = image.shape[:2]
    regions: list[dict] = []

    for (r0, r1) in bands:
        strip = binary[r0:r1, :]
        col_sums = strip.sum(axis=0)
        col_threshold = col_sums.max() * 0.05 if col_sums.max() > 0 else 1
        text_cols = col_sums > col_threshold
        col_runs = _group_runs(text_cols)
        if not col_runs:
            continue
        for (c0, c1) in col_runs:
            regions.append({
                "x": int(c0), "y": int(r0),
                "w": int(c1 - c0), "h": int(r1 - r0),
                "type": "text_block",
                "region_id": str(uuid.uuid4()),
            })

    if not regions:
        return [_full_page_region(image)]

    # Arabic RTL: sort regions right-to-left (descending x)
    regions.sort(key=lambda r: -(r["x"] + r["w"]))
    log.debug(f"layout_detection backend=opencv regions={len(regions)}")
    return regions


def _group_runs(mask: np.ndarray, min_gap: int = 5) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for True runs in a boolean array."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            if i - start > min_gap:
                runs.append((start, i))
            in_run = False
    if in_run and len(mask) - start > min_gap:
        runs.append((start, len(mask)))
    return runs


# ---------------------------------------------------------------------------
# Optional layoutparser backend
# ---------------------------------------------------------------------------

def _try_layoutparser(image: np.ndarray, config) -> list[dict] | None:
    try:
        import layoutparser as lp  # noqa: F401
    except ImportError:
        return None

    try:
        model = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config")
        layout = model.detect(image)
        regions = []
        for block in layout:
            x1, y1, x2, y2 = (int(v) for v in block.coordinates)
            regions.append({
                "x": x1, "y": y1,
                "w": x2 - x1, "h": y2 - y1,
                "type": block.type.lower() if block.type else "text_block",
                "region_id": str(uuid.uuid4()),
            })
        regions.sort(key=lambda r: -(r["x"] + r["w"]))
        log.debug(f"layout_detection backend=layoutparser regions={len(regions)}")
        return regions if regions else None
    except Exception as exc:
        log.warning(f"layoutparser detection failed: {exc}")
        return None


def _full_page_region(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    return {"x": 0, "y": 0, "w": w, "h": h, "type": "text_block", "region_id": "full"}
