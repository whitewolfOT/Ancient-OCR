"""Orchestrate all preprocessing steps for a single page image."""

from __future__ import annotations

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


def suggest_settings_from_degradation(flags: dict, config=None) -> dict:
    """Return suggested preprocessing settings from degradation flags.

    Priority when multiple flags conflict: faded_ink > bleed_through > low_contrast.
    high_noise overrides denoise independently (strongest denoise wins).
    Returns an empty dict when no flags are set.
    """
    settings: dict = {}

    if flags.get("faded_ink"):
        settings["clahe"] = 6.0
        settings["binarization"] = "OTSU"
    elif flags.get("bleed_through"):
        settings["binarization"] = "Adaptive"
        settings["denoise"] = 7
    elif flags.get("low_contrast"):
        settings["clahe"] = 4.0

    if flags.get("high_noise"):
        settings["denoise"] = 9  # overrides bleed_through denoise

    return settings


def preprocess_image(
    image: np.ndarray,
    config=None,
    source_dpi: int | None = None,
    profile=None,
) -> tuple[np.ndarray, dict]:
    """Run enabled preprocessing steps in canonical order.

    Steps: dpi_normalize → denoise → CLAHE → deskew → adaptive binarization

    Each step is wrapped in try/except. On failure: log the exception, keep
    the pre-step image, record status as 'failed' — never propagate.

    Args:
        image:      Input BGR image array.
        config:     Pipeline config object (reads preprocessing.* keys).
        source_dpi: Actual DPI of the source image. When below target_dpi,
                    the image is upscaled using INTER_CUBIC before any other
                    processing. Pass None to skip DPI normalization.

    Returns:
        (processed_image, metadata)  where metadata keys are step names and
        values are 'applied' | 'skipped' | 'failed', plus 'degradation_flags'
        and 'suggested_settings'.
    """
    meta: dict = {}
    current = image.copy()

    # Degradation detection on the original (pre-processing) image
    try:
        from ingest.document_loader import classify_page
        flags = classify_page(image, config)
        meta["degradation_flags"] = flags
        meta["suggested_settings"] = suggest_settings_from_degradation(flags, config)
    except Exception as exc:
        log.debug(f"degradation detection failed: {exc}")
        meta["degradation_flags"] = {}
        meta["suggested_settings"] = {}

    # Step 0: DPI normalization — upscale low-res scans before all other steps
    target_dpi = 300
    try:
        target_dpi = config.preprocessing.target_dpi
    except AttributeError:
        pass

    if source_dpi is not None and source_dpi < target_dpi:
        current, meta = _run_step(
            "dpi_normalize", current, meta,
            True,
            lambda img: _do_dpi_normalize(img, source_dpi, target_dpi),
        )
        log.debug(f"dpi_normalize source={source_dpi} target={target_dpi}")
    else:
        meta["dpi_normalize"] = "skipped"

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

    # Step 5: profile-driven fine adjustments (after general preprocessing)
    if profile is not None:
        current, meta = _run_step(
            "profile_adjustments", current, meta,
            True,
            lambda img: _do_profile_adjustments(img, profile),
        )
    else:
        meta["profile_adjustments"] = "skipped"

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


def _do_profile_adjustments(image: np.ndarray, profile) -> np.ndarray:
    from preprocessing.adjustments import apply_profile_adjustments
    return apply_profile_adjustments(image, profile.preprocessing)


def _do_dpi_normalize(image: np.ndarray, source_dpi: int, target_dpi: int) -> np.ndarray:
    import cv2
    scale = target_dpi / source_dpi
    new_h = int(round(image.shape[0] * scale))
    new_w = int(round(image.shape[1] * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


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
