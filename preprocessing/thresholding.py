"""Contrast enhancement and adaptive binarization."""

from __future__ import annotations

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def apply_clahe(image: np.ndarray, config=None) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization.

    Improves local contrast before binarization; safe for faint/uneven ink.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required") from exc

    clip_limit = 2.0
    tile_grid = (8, 8)
    if config is not None:
        c = getattr(getattr(config, "preprocessing", None), "clahe", None)
        if c is not None:
            clip_limit = getattr(c, "clip_limit", clip_limit)
            tg = getattr(c, "tile_grid", None)
            if tg is not None:
                tile_grid = (int(tg[0]), int(tg[1]))

    gray = _ensure_gray(image, cv2)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    result = clahe.apply(gray)
    log.debug(f"clahe clip={clip_limit} grid={tile_grid}")
    return result


def adaptive_binarization(image: np.ndarray, config=None) -> np.ndarray:
    """Binarization with configurable method.

    Reads config.preprocessing.binarize.method:
      "adaptive" (default) — Gaussian adaptive thresholding
      "otsu"               — Otsu's global thresholding
      "global"             — fixed global threshold at 127

    Adaptive is preferred for documents with uneven ink or lighting.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required") from exc

    method = "adaptive"
    if config is not None:
        b = getattr(getattr(config, "preprocessing", None), "binarize", None)
        if b is not None:
            method = getattr(b, "method", method).lower()

    gray = _ensure_gray(image, cv2)

    if method == "otsu":
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "global":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        result = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 10,
        )

    log.debug(f"binarization method={method}")
    return result


def _ensure_gray(image: np.ndarray, cv2) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
