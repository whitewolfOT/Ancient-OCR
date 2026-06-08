"""Tests for align/openiti.py — all mocked, no Passim/Java required."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from align.openiti import AlignmentResult, align_page, _rapidfuzz_align, _load_corpus


# ---------------------------------------------------------------------------
# AlignmentResult dataclass
# ---------------------------------------------------------------------------

def test_alignment_result_fields():
    r = AlignmentResult(
        corrected_text="abc", ocr_text="abc", confidence=0.9,
        method="rapidfuzz", accepted=True, match_start=0, match_end=3,
    )
    assert r.method == "rapidfuzz"
    assert r.accepted is True
    assert r.confidence == 0.9


# ---------------------------------------------------------------------------
# _load_corpus
# ---------------------------------------------------------------------------

def test_load_corpus_missing_returns_empty():
    with patch("align.openiti.Path") as MockPath:
        mock_p = MagicMock()
        mock_p.exists.return_value = False
        MockPath.return_value = mock_p
        result = _load_corpus()
    assert result == ""


def test_load_corpus_reads_file():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "filaha.txt"
        f.write_text("النص العربي", encoding="utf-8")
        with patch("align.openiti.Path", return_value=f):
            result = _load_corpus()
    assert "النص" in result


# ---------------------------------------------------------------------------
# align_page — disabled by default
# ---------------------------------------------------------------------------

def test_align_page_disabled_returns_none():
    result = align_page("بعض النص", config=None)
    assert result is None


def test_align_page_empty_text_returns_none():
    result = align_page("", config=None)
    assert result is None


def test_align_page_whitespace_only_returns_none():
    result = align_page("   ", config=None)
    assert result is None


# ---------------------------------------------------------------------------
# align_page — enabled, no corpus → returns None
# ---------------------------------------------------------------------------

def test_align_page_enabled_no_corpus_returns_none():
    cfg = MagicMock()
    cfg.align.passim.enabled = True
    cfg.align.passim.threshold = 0.85
    cfg.align.passim.window_chars = 2000
    cfg.align.passim.fallback = "rapidfuzz"
    cfg.align.passim.corpus_path = "/nonexistent/path.txt"

    with patch("align.openiti._load_corpus", return_value=""):
        result = align_page("test text", config=cfg)
    assert result is None


# ---------------------------------------------------------------------------
# _rapidfuzz_align
# ---------------------------------------------------------------------------

def test_rapidfuzz_align_returns_alignment_result():
    pytest.importorskip("rapidfuzz")
    reference = "الزراعة والبستنة في الأندلس" * 20
    result = _rapidfuzz_align("الزراعة", reference, window_chars=100, threshold=0.5)
    assert isinstance(result, AlignmentResult)
    assert result.method == "rapidfuzz"
    assert 0.0 <= result.confidence <= 1.0


def test_rapidfuzz_align_empty_reference():
    pytest.importorskip("rapidfuzz")
    result = _rapidfuzz_align("test", "", window_chars=100, threshold=0.85)
    assert result.confidence == 0.0
    assert result.accepted is False
    assert result.corrected_text == "test"


def test_rapidfuzz_align_high_confidence_accepted():
    pytest.importorskip("rapidfuzz")
    text = "الزراعة العربية"
    reference = "   " + text + "   " * 50
    result = _rapidfuzz_align(text, reference * 5, window_chars=500, threshold=0.5)
    assert result.accepted is True


def test_rapidfuzz_align_low_confidence_not_accepted():
    pytest.importorskip("rapidfuzz")
    result = _rapidfuzz_align("xyz123", "الزراعة العربية" * 10, window_chars=100, threshold=0.99)
    assert result.accepted is False
    assert result.corrected_text == "xyz123"


# ---------------------------------------------------------------------------
# align_page — enabled with mocked corpus + rapidfuzz
# ---------------------------------------------------------------------------

def test_align_page_returns_result_when_enabled_with_corpus():
    pytest.importorskip("rapidfuzz")
    cfg = MagicMock()
    cfg.align.passim.enabled = True
    cfg.align.passim.threshold = 0.5
    cfg.align.passim.window_chars = 200
    cfg.align.passim.fallback = "rapidfuzz"
    cfg.align.passim.corpus_path = None

    reference = "الزراعة والبستنة في الأندلس " * 30
    with patch("align.openiti._load_corpus", return_value=reference), \
         patch("align.openiti._passim_align", return_value=None):
        result = align_page("الزراعة", config=cfg)
    assert result is not None
    assert isinstance(result, AlignmentResult)
    assert result.method == "rapidfuzz"
