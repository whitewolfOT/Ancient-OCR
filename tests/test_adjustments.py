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
