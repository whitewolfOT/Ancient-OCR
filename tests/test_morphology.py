"""Tests for morphology modules: root_extractor, camel_adapter, pattern_analyzer."""
import pytest

from morphology.root_extractor import extract_root, RootCandidate
from morphology.camel_adapter import analyze, is_available
from morphology.pattern_analyzer import detect_pattern


# ---------------------------------------------------------------------------
# root_extractor.extract_root
# ---------------------------------------------------------------------------

def test_extract_root_ktb_contains_ktb():
    candidates = extract_root("كتب")
    roots = [c.root for c in candidates]
    assert "كتب" in roots


def test_extract_root_returns_non_empty_list():
    candidates = extract_root("مكتبة")
    assert len(candidates) >= 1


def test_extract_root_returns_list_of_root_candidates():
    candidates = extract_root("كتاب")
    assert all(isinstance(c, RootCandidate) for c in candidates)


def test_extract_root_sorted_by_confidence_desc():
    candidates = extract_root("كتاب")
    confs = [c.confidence for c in candidates]
    assert confs == sorted(confs, reverse=True)


def test_extract_root_any_arabic_word():
    candidates = extract_root("علماء")
    assert len(candidates) >= 1
    for c in candidates:
        assert isinstance(c.root, str)
        assert len(c.root) >= 2


def test_extract_root_short_word():
    candidates = extract_root("في")
    assert len(candidates) >= 1


def test_extract_root_with_prefix():
    candidates = extract_root("الكتاب")
    roots = [c.root for c in candidates]
    # Should strip ال and find كتب or كتاب
    assert any(len(r) >= 2 for r in roots)


# ---------------------------------------------------------------------------
# camel_adapter.analyze
# ---------------------------------------------------------------------------

def test_camel_analyze_returns_none_when_unavailable():
    """camel-tools not installed in test env; analyze must return None gracefully."""
    result = analyze("كتاب")
    # Either None (not installed) or a dict (installed)
    assert result is None or isinstance(result, dict)


def test_camel_is_available_returns_bool():
    result = is_available()
    assert isinstance(result, bool)


def test_camel_analyze_does_not_raise():
    try:
        analyze("مكتبة")
    except Exception:
        pytest.fail("analyze() raised an exception")


# ---------------------------------------------------------------------------
# pattern_analyzer.detect_pattern
# ---------------------------------------------------------------------------

def test_detect_pattern_returns_dict():
    result = detect_pattern("كتاب")
    assert isinstance(result, dict)


def test_detect_pattern_has_required_keys():
    result = detect_pattern("كتاب")
    assert "pattern" in result
    assert "confidence" in result
    assert "wazn" in result


def test_detect_pattern_confidence_in_range():
    result = detect_pattern("كتاب")
    assert 0.0 <= result["confidence"] <= 1.0


def test_detect_pattern_with_root():
    result = detect_pattern("كتاب", root="كتب")
    assert isinstance(result, dict)
    assert "confidence" in result


def test_detect_pattern_unknown_word():
    result = detect_pattern("zzz")
    # Must not raise; confidence may be 0
    assert isinstance(result, dict)
    assert result["confidence"] >= 0.0
