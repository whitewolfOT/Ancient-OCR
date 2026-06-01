"""Noise reduction for Arabic document images."""

from __future__ import annotations

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def denoise(image: np.ndarray, config=None) -> np.ndarray:
    """Reduce noise while preserving Arabic dots and diacritics.

    Uses fast non-local means (default) or median blur depending on config.
    Conservative kernel to avoid over-smoothing fine strokes.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for denoising") from exc

    method = "fast_nl_means"
    kernel_size = 3
    if config is not None:
        d = getattr(config, "preprocessing", None)
        if d is not None:
            dn = getattr(d, "denoise", None)
            if dn is not None:
                method = getattr(dn, "method", method)
                kernel_size = getattr(dn, "kernel_size", kernel_size)

    gray = _to_gray(image, cv2)

    if method == "median":
        result = median_filter(gray, kernel_size)
    else:
        result = _fast_nl_means(gray, cv2)

    log.debug(f"denoise method={method} kernel={kernel_size}")
    return result


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median blur. kernel_size must be odd."""
    import cv2
    gray = _to_gray(image, cv2)
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(gray, k)


def _fast_nl_means(gray: np.ndarray, cv2) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _to_gray(image: np.ndarray, cv2) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
