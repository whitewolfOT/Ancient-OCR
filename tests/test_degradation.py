"""Tests for degradation detection in classify_page() and suggest_settings_from_degradation()."""
from __future__ import annotations

import numpy as np
import pytest


def _make_image(h: int = 100, w: int = 100, val: int = 128, channels: int = 3) -> np.ndarray:
    if channels == 1:
        return np.full((h, w), val, dtype=np.uint8)
    return np.full((h, w, channels), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# classify_page — new degradation flags
# ---------------------------------------------------------------------------

class TestClassifyPage:
    def test_clean_image_no_flags(self):
        from ingest.document_loader import classify_page
        # Black text on white: strong contrast, low median (lots of white → ~200?),
        # create a truly high-contrast image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:90, 10:90] = 255  # big white area with black border
        result = classify_page(img)
        assert "low_contrast" in result
        assert "faded_ink" in result
        assert "high_noise" in result
        assert "bleed_through" in result

    def test_faded_ink_flag(self):
        from ingest.document_loader import classify_page
        # All-white image → median well above 200 → faded_ink = True
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        result = classify_page(img)
        assert result["faded_ink"] is True

    def test_no_faded_ink_on_dark_image(self):
        from ingest.document_loader import classify_page
        # Dark image → median well below 200 → faded_ink = False
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        result = classify_page(img)
        assert result["faded_ink"] is False

    def test_low_contrast_flag(self):
        from ingest.document_loader import classify_page
        # Uniform grey — p5 and p95 are identical → range = 0 < 100
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = classify_page(img)
        assert result["low_contrast"] is True

    def test_no_low_contrast_on_high_contrast_image(self):
        from ingest.document_loader import classify_page
        # Half black, half white → p5 near 0, p95 near 255, range >> 100
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, 100:] = 255
        result = classify_page(img)
        assert result["low_contrast"] is False

    def test_high_noise_flag(self):
        from ingest.document_loader import classify_page
        # Random noise → very high Laplacian variance
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        result = classify_page(img)
        assert result["high_noise"] is True

    def test_no_high_noise_on_smooth_image(self):
        from ingest.document_loader import classify_page
        # Smooth gradient → Laplacian variance near 0
        row = np.linspace(0, 255, 200, dtype=np.uint8)
        img = np.tile(row, (200, 1))
        img = np.stack([img, img, img], axis=2)
        result = classify_page(img)
        assert result["high_noise"] is False

    def test_existing_flags_preserved(self):
        from ingest.document_loader import classify_page
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = classify_page(img)
        assert "is_low_res" in result
        assert "likely_skewed" in result
        assert "multi_column" in result
        assert "text_density" in result


# ---------------------------------------------------------------------------
# suggest_settings_from_degradation
# ---------------------------------------------------------------------------

class TestSuggestSettings:
    def test_no_flags_returns_empty(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        assert suggest_settings_from_degradation({}) == {}

    def test_low_contrast_suggests_higher_clahe(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        result = suggest_settings_from_degradation({"low_contrast": True})
        assert result.get("clahe", 0) >= 4.0

    def test_faded_ink_suggests_high_clahe_and_otsu(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        result = suggest_settings_from_degradation({"faded_ink": True})
        assert result.get("clahe", 0) >= 6.0
        assert result.get("binarization") == "OTSU"

    def test_high_noise_suggests_strong_denoise(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        result = suggest_settings_from_degradation({"high_noise": True})
        assert result.get("denoise", 0) >= 7

    def test_bleed_through_suggests_adaptive_and_denoise(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        result = suggest_settings_from_degradation({"bleed_through": True})
        assert result.get("binarization") == "Adaptive"
        assert "denoise" in result

    def test_faded_ink_dominates_low_contrast_for_binarization(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        result = suggest_settings_from_degradation({"faded_ink": True, "low_contrast": True})
        assert result.get("binarization") == "OTSU"
        assert result.get("clahe", 0) >= 6.0

    def test_high_noise_stacks_with_other_flags(self):
        from preprocessing.image_pipeline import suggest_settings_from_degradation
        result = suggest_settings_from_degradation({"low_contrast": True, "high_noise": True})
        assert result.get("clahe", 0) >= 4.0
        assert result.get("denoise", 0) >= 7
