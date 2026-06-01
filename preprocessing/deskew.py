"""Skew detection and correction via projection-profile analysis."""

from __future__ import annotations

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def detect_skew(image: np.ndarray, config=None) -> float:
    """Estimate the skew angle of a document image in degrees.

    Uses projection-profile energy: rotates the image across candidate angles
    and picks the angle that maximises row-sum variance (sharp horizontal
    text lines produce high variance; skewed text smears it).

    Returns 0.0 on failure (blank, noise-only, or very short images).
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for deskew") from exc

    max_angle = 15.0
    steps = 100
    if config is not None:
        d = getattr(getattr(config, "preprocessing", None), "deskew", None)
        if d is not None:
            max_angle = getattr(d, "max_angle", max_angle)
            steps = getattr(d, "angle_steps", steps)

    gray = _ensure_gray(image, cv2)
    if gray.shape[0] < 10 or gray.shape[1] < 10:
        return 0.0

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = -1.0

    angles = np.linspace(-max_angle, max_angle, steps)
    h, w = binary.shape
    cx, cy = w / 2, h / 2

    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_LINEAR)
        row_sums = rotated.sum(axis=1).astype(np.float64)
        score = float(row_sums.var())
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    log.debug(f"detect_skew angle={best_angle:.2f} score={best_score:.1f}")
    return best_angle


def correct_skew(image: np.ndarray, angle: float, config=None) -> np.ndarray:
    """Rotate image by -angle to deskew; no-op if angle is below threshold.

    Preserves the original image dimensions.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for deskew") from exc

    no_op_threshold = 0.5
    if config is not None:
        d = getattr(getattr(config, "preprocessing", None), "deskew", None)
        if d is not None:
            no_op_threshold = getattr(d, "no_op_threshold", no_op_threshold)

    if abs(angle) < no_op_threshold:
        log.debug(f"deskew skipped angle={angle:.2f} below threshold={no_op_threshold}")
        return image

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    border = 255 if image.ndim == 2 else (255, 255, 255)
    corrected = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    log.debug(f"correct_skew angle={angle:.2f}")
    return corrected


def _ensure_gray(image: np.ndarray, cv2) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
