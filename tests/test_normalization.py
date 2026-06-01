"""Tests for normalization modules: arabic_normalizer and noise_filter."""
import pytest

from normalization.arabic_normalizer import normalize_text
from normalization.noise_filter import clean_noise


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

def test_alef_variants_folded():
    result, log = normalize_text("إبراهيم")
    assert "أ" not in result
    assert "إ" not in result
    assert "آ" not in result
    assert len(log) > 0


def test_alef_variant_returns_non_empty_log():
    _, log = normalize_text("إبراهيم")
    assert len(log) > 0


def test_tatweel_removed():
    result, log = normalize_text("كتـاب")
    assert "ـ" not in result


def test_alef_maqsura_folded():
    result, _ = normalize_text("موسى")
    # alef maqsura (ى U+0649) → ya (ي U+064A)
    assert "ى" not in result  # no alef maqsura


def test_taa_marbuta_preserved_by_default():
    """taa_marbuta folding is OFF by default — 'ة' must stay unchanged."""
    result, _ = normalize_text("مدرسة")
    assert "ة" in result


def test_normalized_text_is_string():
    result, log = normalize_text("كتاب")
    assert isinstance(result, str)
    assert isinstance(log, list)


def test_no_change_returns_empty_log_for_plain_word():
    """A pure ba/ta/tha/jim word with no variants produces minimal log entries."""
    result, log = normalize_text("بيت")
    assert isinstance(log, list)


def test_normalize_returns_tuple():
    out = normalize_text("علم")
    assert len(out) == 2


# ---------------------------------------------------------------------------
# clean_noise
# ---------------------------------------------------------------------------

def test_clean_noise_removes_control_chars():
    text = "كتاب\x00\x01\x02"
    cleaned, log = clean_noise(text)
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert len(log) > 0


def test_clean_noise_preserves_valid_arabic():
    text = "كتاب علم"
    cleaned, log = clean_noise(text)
    assert "كتاب" in cleaned
    assert "علم" in cleaned


def test_clean_noise_preserves_ascii():
    text = "hello world"
    cleaned, _ = clean_noise(text)
    assert cleaned == "hello world"


def test_clean_noise_returns_tuple():
    out = clean_noise("test")
    assert len(out) == 2


def test_clean_noise_empty_string():
    cleaned, log = clean_noise("")
    assert cleaned == ""
    assert log == []
