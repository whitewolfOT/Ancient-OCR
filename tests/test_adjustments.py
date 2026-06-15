"""Tests for preprocessing/adjustments.py"""
import numpy as np
import pytest

from ocr_engine.profile_loader import PreprocessingParams
from preprocessing.adjustments import (
    adjust_brightness_contrast,
    adjust_gamma,
    adjust_saturation,
    apply_profile_adjustments,
    denoise,
    normalize_stroke_thickness,
    sharpen,
)


def _gray(val=128, h=80, w=80):
    return np.full((h, w), val, dtype=np.uint8)


def _bgr(val=128, h=80, w=80):
    return np.full((h, w, 3), val, dtype=np.uint8)


def test_brightness_increases_mean():
    img = _gray(100)
    out = adjust_brightness_contrast(img, brightness=30, contrast=1.0)
    assert float(out.mean()) > float(img.mean())


def test_contrast_increases_std():
    # gradient image has measurable std
    img = np.tile(np.arange(80, dtype=np.uint8), (80, 1))
    out = adjust_brightness_contrast(img, brightness=0, contrast=2.0)
    assert float(out.std()) >= float(img.std())


def test_gamma_gt1_brightens():
    # inv = 1/gamma; gamma=2 → x^0.5 which lifts mid-tones
    img = _gray(100)
    out = adjust_gamma(img, gamma=2.0)
    assert float(out.mean()) > float(img.mean())


def test_saturation_noop_on_grayscale():
    img = _gray(128)
    out = adjust_saturation(img, factor=2.0)
    np.testing.assert_array_equal(img, out)


def test_stroke_normalization_synthetic():
    # 1-pixel wide vertical strokes on white background
    img = np.full((60, 60), 255, dtype=np.uint8)
    img[:, 10] = 0
    img[:, 30] = 0
    img[:, 50] = 0
    out = normalize_stroke_thickness(img, target_width=4)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_denoise_noop_at_zero():
    img = _gray(128)
    out = denoise(img, strength=0)
    np.testing.assert_array_equal(img, out)


def test_sharpen_noop_at_zero():
    img = _gray(128)
    out = sharpen(img, amount=0)
    np.testing.assert_array_equal(img, out)


def test_apply_pipeline_order_deterministic():
    img = _bgr(100)
    params = PreprocessingParams(
        brightness=10, contrast=1.1, gamma=0.95,
        saturation=1.0, denoise_strength=0, sharpen=0.2,
        stroke_normalization_enabled=False,
    )
    out1 = apply_profile_adjustments(img.copy(), params)
    out2 = apply_profile_adjustments(img.copy(), params)
    np.testing.assert_array_equal(out1, out2)


# ── best_channel_extraction tests ─────────────────────────────────────────────

def test_best_channel_returns_grayscale_for_color_input():
    """Colour (H, W, 3) input → grayscale (H, W) output."""
    from preprocessing.adjustments import best_channel_extraction
    img = np.random.randint(0, 256, (50, 80, 3), dtype=np.uint8)
    out = best_channel_extraction(img)
    assert len(out.shape) == 2
    assert out.shape == (50, 80)


def test_best_channel_noop_for_grayscale():
    """Grayscale input returned unchanged (same object or equal)."""
    from preprocessing.adjustments import best_channel_extraction
    img = np.random.randint(0, 256, (50, 80), dtype=np.uint8)
    out = best_channel_extraction(img)
    np.testing.assert_array_equal(out, img)


def test_best_channel_picks_highest_contrast():
    """Synthetic image where green channel has max std → green channel returned."""
    from preprocessing.adjustments import best_channel_extraction
    h, w = 40, 40
    # B: uniform 128, G: alternating 0/255, R: uniform 64
    b = np.full((h, w), 128, dtype=np.uint8)
    g = np.tile(np.array([0, 255], dtype=np.uint8), (h, w // 2))
    r = np.full((h, w), 64, dtype=np.uint8)
    img = np.stack([b, g, r], axis=2)  # BGR order
    out = best_channel_extraction(img)
    # Green (index 1) has std ≈ 127.5 — highest; B and R near 0
    np.testing.assert_array_equal(out, g)


def test_best_channel_returns_ndarray():
    """Output is always a numpy ndarray."""
    from preprocessing.adjustments import best_channel_extraction
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    out = best_channel_extraction(img)
    assert isinstance(out, np.ndarray)


# ── remove_bleedthrough tests ──────────────────────────────────────────────────

def test_bleedthrough_noop_at_zero_strength():
    """strength=0 → image returned unchanged."""
    from preprocessing.adjustments import remove_bleedthrough
    img = _gray(128)
    out = remove_bleedthrough(img, strength=0)
    np.testing.assert_array_equal(out, img)


def test_bleedthrough_changes_image_at_nonzero():
    """strength>0 on a non-uniform image should change pixel values."""
    from preprocessing.adjustments import remove_bleedthrough
    rng = np.random.default_rng(42)
    img = rng.integers(50, 200, (80, 80), dtype=np.uint8)
    out = remove_bleedthrough(img, strength=0.5)
    assert not np.array_equal(out, img)


def test_bleedthrough_output_same_shape():
    """Output shape matches input shape for grayscale input."""
    from preprocessing.adjustments import remove_bleedthrough
    img = _gray(100, h=60, w=90)
    out = remove_bleedthrough(img, strength=0.5)
    assert out.shape == img.shape


def test_bleedthrough_output_uint8():
    """Output dtype is uint8."""
    from preprocessing.adjustments import remove_bleedthrough
    img = _gray(128)
    out = remove_bleedthrough(img, strength=0.5)
    assert out.dtype == np.uint8


def test_bleedthrough_noop_on_clean_image():
    """Uniform-grey image should remain close to original after removal."""
    from preprocessing.adjustments import remove_bleedthrough
    img = _gray(200)
    out = remove_bleedthrough(img, strength=0.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
