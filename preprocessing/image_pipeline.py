"""Orchestrate all preprocessing steps for a single page image."""

from __future__ import annotations

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def preprocess_image(image: np.ndarray, config=None) -> tuple[np.ndarray, dict]:
    """Run enabled preprocessing steps in canonical order.

    Steps: grayscale → denoise → CLAHE → deskew → adaptive binarization

    Each step is wrapped in try/except. On failure: log the exception, keep
    the pre-step image, record status as 'failed' — never propagate.

    Returns:
        (processed_image, metadata)  where metadata keys are step names and
        values are 'applied' | 'skipped' | 'failed'.
    """
    meta: dict[str, str] = {}
    current = image.copy()

    # Step 1: denoise
    current, meta = _run_step(
        "denoise", current, meta,
        _is_enabled(config, "denoise"),
        lambda img: _do_denoise(img, config),
    )

    # Step 2: CLAHE contrast enhancement
    current, meta = _run_step(
        "clahe", current, meta,
        _is_enabled(config, "clahe"),
        lambda img: _do_clahe(img, config),
    )

    # Step 3: deskew
    current, meta = _run_step(
        "deskew", current, meta,
        _is_enabled(config, "deskew"),
        lambda img: _do_deskew(img, config),
    )

    # Step 4: adaptive binarization
    current, meta = _run_step(
        "binarize", current, meta,
        True,  # always run binarization
        lambda img: _do_binarize(img, config),
    )

    log.debug(f"preprocess_image steps={meta}")
    return current, meta


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def _run_step(
    name: str,
    image: np.ndarray,
    meta: dict,
    enabled: bool,
    fn,
) -> tuple[np.ndarray, dict]:
    if not enabled:
        meta[name] = "skipped"
        return image, meta
    try:
        result = fn(image)
        meta[name] = "applied"
        return result, meta
    except Exception as exc:
        log.warning(f"preprocess step={name} failed: {exc}")
        meta[name] = "failed"
        return image, meta


def _do_denoise(image: np.ndarray, config) -> np.ndarray:
    from preprocessing.denoise import denoise
    return denoise(image, config)


def _do_clahe(image: np.ndarray, config) -> np.ndarray:
    from preprocessing.thresholding import apply_clahe
    return apply_clahe(image, config)


def _do_deskew(image: np.ndarray, config) -> np.ndarray:
    from preprocessing.deskew import detect_skew, correct_skew
    angle = detect_skew(image, config)
    return correct_skew(image, angle, config)


def _do_binarize(image: np.ndarray, config) -> np.ndarray:
    from preprocessing.thresholding import adaptive_binarization
    return adaptive_binarization(image, config)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _is_enabled(config, step: str) -> bool:
    try:
        pre = getattr(config, "preprocessing", None)
        if pre is None:
            return True
        step_cfg = getattr(pre, step, None)
        if step_cfg is None:
            return True
        return bool(getattr(step_cfg, "enabled", True))
    except Exception:
        return True
