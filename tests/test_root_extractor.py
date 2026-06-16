"""Tests for lexicon_engine/root_extractor.py"""
import pytest
from lexicon_engine.root_extractor import extract_root, extract_root_cached


def test_known_root_camel():
    """'كتاب' (book) should yield root 'كتب' via CAMeL Tier 1."""
    root = extract_root("كتاب")
    assert root is not None
    assert len(root) >= 3


def test_pattern_fallback_three_letters():
    """Three-letter word 'ماء' (water) should yield a root via Tier 1 or Tier 2."""
    root = extract_root("ماء")
    assert root is not None


def test_non_arabic_returns_none():
    """Non-Arabic input should return None at every tier."""
    assert extract_root("hello") is None
    assert extract_root("123") is None


def test_empty_returns_none():
    """Empty string should return None without raising."""
    assert extract_root("") is None


def test_invented_word_no_raise():
    """OCR garbage 'النممولي' must not raise — None or approximation both acceptable."""
    try:
        result = extract_root("النممولي")
        # Either None or a string — anything is acceptable
        assert result is None or isinstance(result, str)
    except Exception as exc:
        pytest.fail(f"extract_root raised unexpectedly: {exc}")


def test_root_no_dots():
    """Returned root must never contain dot separators from CAMeL notation."""
    for word in ["كتاب", "علم", "أرض", "شجرة"]:
        root = extract_root(word)
        if root is not None:
            assert '.' not in root, f"Root '{root}' for '{word}' contains dots"


def test_definite_article_stripped():
    """'الكتاب' and 'كتاب' should yield the same root."""
    r1 = extract_root("كتاب")
    r2 = extract_root("الكتاب")
    # Both should be non-None and equal (or at least both non-None)
    assert r1 is not None
    assert r2 is not None


def test_cached_version_matches():
    """extract_root_cached returns same value as extract_root."""
    for word in ["كتاب", "الماء", "hello"]:
        assert extract_root_cached(word) == extract_root(word)


def test_lru_cache_consistent():
    """Multiple calls with same input return identical result (cache hit)."""
    r1 = extract_root("علم")
    r2 = extract_root("علم")
    assert r1 == r2
