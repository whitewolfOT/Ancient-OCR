"""
Profile-driven image adjustments.
All functions: take np.ndarray (uint8, grayscale or BGR), return np.ndarray.
No import-time side effects. cv2 and numpy only — no kraken dependency.
"""
from __future__ import annotations

import cv2
import numpy as np

from ocr_engine.profile_loader import PreprocessingParams


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0,
                                contrast: float = 1.0) -> np.ndarray:
    """alpha=contrast, beta=brightness via convertScaleAbs."""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01:
        return img
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def adjust_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """No-op on grayscale. Adjusts HSV saturation channel for colour."""
    if len(img.shape) == 2 or abs(factor - 1.0) < 0.01:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def normalize_stroke_thickness(img: np.ndarray, target_width: int = 2) -> np.ndarray:
    """Estimate median stroke width via distance transform, then dilate/erode to target.

    Input/output: grayscale uint8, white background (standard).
    Apply only after denoise, never on colour images.
    """
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    non_zero = dist[dist > 0]
    if len(non_zero) == 0:
        return img
    current_width = float(np.median(non_zero)) * 2
    diff = target_width - current_width
    if abs(diff) < 0.5:
        return img
    ksize = max(1, int(abs(diff) + 0.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    adjusted = cv2.dilate(binary, kernel) if diff > 0 else cv2.erode(binary, kernel)
    return 255 - adjusted  # back to white-background grayscale


def denoise(img: np.ndarray, strength: int = 15) -> np.ndarray:
    if strength <= 0:
        return img
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(img, None, h=float(strength),
                                    templateWindowSize=7, searchWindowSize=21)


def sharpen(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def apply_profile_adjustments(img: np.ndarray, params: PreprocessingParams) -> np.ndarray:
    """Apply all profile-driven adjustments in canonical order.

    Called by preprocessing/image_pipeline.py when a profile is active.
    The binarizer (nlbin/sauvola/otsu) is NOT applied here — that's KrakenBackend's job.
    """
    # 1. Saturation (colour only — no-op on grayscale)
    img = adjust_saturation(img, params.saturation)

    # 2. Convert to grayscale for remaining steps
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Brightness / contrast
    img = adjust_brightness_contrast(img, params.brightness, params.contrast)

    # 4. Gamma
    img = adjust_gamma(img, params.gamma)

    # 5. Denoise
    img = denoise(img, params.denoise_strength)

    # 6. Stroke normalization (expensive — only when enabled)
    if params.stroke_normalization_enabled:
        img = normalize_stroke_thickness(img, params.stroke_target_width)

    # 7. Sharpen
    img = sharpen(img, params.sharpen)

    return img
