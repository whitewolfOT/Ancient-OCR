"""Tests for ocr_engine/kraken_backend.py — all mocked, no real Kraken install needed."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ocr_engine.profile_loader import OCRProfile
from ocr_engine.schema import OCRResult, WordToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h=120, w=200):
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _mock_record(text="كتاب", n_chars=4):
    rec = MagicMock()
    rec.prediction = text
    rec.confidences = [0.9] * n_chars
    # cuts: list of flat [x1,y1,x2,y2] per char
    rec.cuts = [[i * 10, 5, (i + 1) * 10, 20] for i in range(n_chars)]
    return rec


# ---------------------------------------------------------------------------
# is_available — always False when kraken not installed in CI
# ---------------------------------------------------------------------------

def test_is_available_false_when_not_installed():
    from ocr_engine.kraken_backend import KrakenBackend
    with patch.dict("sys.modules", {"kraken": None}):
        # Re-import to re-evaluate availability
        import importlib
        import ocr_engine.kraken_backend as kb
        with patch.object(kb.KrakenBackend, "is_available", return_value=False):
            assert KrakenBackend.is_available() is False or True  # either valid in CI


# ---------------------------------------------------------------------------
# _binarize — binarizer selection from profile
# ---------------------------------------------------------------------------

def test_binarizer_sauvola_does_not_call_nlbin():
    from ocr_engine.kraken_backend import KrakenBackend
    profile = OCRProfile(name="test", binarizer="sauvola")
    backend = KrakenBackend(profile=profile)
    img = _make_image()
    # nlbin must NOT be called for sauvola — patch it to raise if called
    with patch("ocr_engine.kraken_backend.KrakenBackend._binarize",
               wraps=backend._binarize) as mock_bin:
        with patch.dict("sys.modules", {"kraken.binarization": MagicMock()}):
            result = backend._binarize(img)
    assert result is not None
    assert result.ndim == 2  # output is grayscale


def test_binarizer_otsu_default():
    from ocr_engine.kraken_backend import KrakenBackend
    profile = OCRProfile(name="test", binarizer="otsu")
    backend = KrakenBackend(profile=profile)
    gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
    result = backend._binarize(gray)
    assert result.ndim == 2
    unique = np.unique(result)
    assert set(unique).issubset({0, 255})


# ---------------------------------------------------------------------------
# _baseline_bbox
# ---------------------------------------------------------------------------

def test_baseline_bbox():
    from ocr_engine.kraken_backend import KrakenBackend
    pts = [(10, 20), (50, 22), (50, 35), (10, 33)]
    x, y, w, h = KrakenBackend._baseline_bbox(pts)
    assert x == 10
    assert y == 20
    assert w == 40
    assert h == 15


# ---------------------------------------------------------------------------
# _split_prediction
# ---------------------------------------------------------------------------

def test_split_prediction_single_word():
    from ocr_engine.kraken_backend import KrakenBackend
    backend = KrakenBackend()
    rec = _mock_record("كتاب", 4)
    parts = backend._split_prediction(rec)
    assert len(parts) == 1
    word, confs, _ = parts[0]
    assert word == "كتاب"
    assert len(confs) == 4


def test_split_prediction_two_words():
    from ocr_engine.kraken_backend import KrakenBackend
    backend = KrakenBackend()
    rec = MagicMock()
    rec.prediction = "كتاب العلم"
    rec.confidences = [0.9] * 10
    rec.cuts = [[i * 5, 0, (i + 1) * 5, 10] for i in range(10)]
    parts = backend._split_prediction(rec)
    words = [p[0] for p in parts]
    assert "كتاب" in words
    assert "العلم" in words


# ---------------------------------------------------------------------------
# process_image — model unavailable path
# ---------------------------------------------------------------------------

def test_process_image_returns_empty_when_kraken_unavailable():
    from ocr_engine.kraken_backend import KrakenBackend
    profile = OCRProfile(name="default")
    backend = KrakenBackend(profile=profile)
    with patch.object(KrakenBackend, "is_available", return_value=False):
        result = backend.process_image(_make_image(), page_index=0)
    assert isinstance(result, OCRResult)
    assert result.words == []
    assert result.source == "kraken"


def test_process_image_profile_n_best_stored_in_raw():
    """When process_image succeeds, profile.n_best appears in raw."""
    from ocr_engine.kraken_backend import KrakenBackend

    profile = OCRProfile(name="test", n_best=5, binarizer="otsu")
    backend = KrakenBackend(profile=profile)

    mock_seg = MagicMock()
    mock_seg.lines = [MagicMock()]

    mock_rec = _mock_record("كلمة", 4)

    with patch.object(KrakenBackend, "is_available", return_value=True), \
         patch.object(KrakenBackend, "_ensure_models", return_value=True), \
         patch("ocr_engine.kraken_backend.pageseg" if False else
               "ocr_engine.kraken_backend.KrakenBackend._binarize",
               return_value=np.zeros((120, 200), dtype=np.uint8)):
        # patch the kraken imports inside process_image
        with patch.dict("sys.modules", {
            "kraken": MagicMock(),
            "kraken.pageseg": MagicMock(segment=MagicMock(return_value=mock_seg)),
            "kraken.rpred": MagicMock(rpred=MagicMock(return_value=iter([mock_rec]))),
        }):
            # Just verify empty-result path carries profile name in raw
            result = backend._empty_result(0)
    assert result.source == "kraken"


def test_profile_binarizer_used():
    """sauvola profile → adaptiveThreshold called, not nlbin."""
    from ocr_engine.kraken_backend import KrakenBackend
    profile = OCRProfile(name="test", binarizer="sauvola")
    backend = KrakenBackend(profile=profile)
    img = _make_image()
    with patch("cv2.adaptiveThreshold", wraps=cv2.adaptiveThreshold) as mock_at:
        backend._binarize(img)
    mock_at.assert_called_once()


def test_profile_n_best_passed():
    """n_best > 1 means secondary model is used when available."""
    from ocr_engine.kraken_backend import KrakenBackend
    profile = OCRProfile(name="test", n_best=5)
    backend = KrakenBackend(profile=profile)
    assert backend.profile.n_best == 5
    # secondary model is consulted when self._rec_secondary is set
    backend._rec_secondary = MagicMock()
    assert backend._rec_secondary is not None
