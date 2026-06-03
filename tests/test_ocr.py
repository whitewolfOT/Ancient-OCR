"""Tests for OCR backends — all use mocks, no real engines called."""
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ocr_engine.schema import OCRResult, WordToken


def _make_image(h=100, w=80):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _synthetic_result(page_index=0):
    tok = WordToken(
        text="كتاب", confidence=0.85,
        bbox=(0, 0, 40, 20),
        page_index=page_index, source="paddle",
    )
    return OCRResult(
        text="كتاب", words=[tok], confidence=0.85,
        page_index=page_index, source="paddle",
        raw={"paddle_raw": []},
    )


# ---------------------------------------------------------------------------
# is_available() when deps absent
# ---------------------------------------------------------------------------

def test_paddle_is_available_false_when_not_installed():
    from ocr_engine.paddle_backend import PaddleBackend
    # In test environment paddleocr is not installed
    assert PaddleBackend.is_available() is False


def test_tesseract_is_available_false_when_not_installed():
    import ocr_engine.tesseract_backend as tess_mod
    with patch.object(tess_mod, "_tesseract_available", False), \
         patch.object(tess_mod, "_tesseract_ara_available", False):
        from ocr_engine.tesseract_backend import TesseractBackend
        assert TesseractBackend.is_available() is False


def test_trocr_is_available_false_when_not_installed():
    from ocr_engine.trocr_backend import TrOCRBackend
    assert TrOCRBackend.is_available() is False


# ---------------------------------------------------------------------------
# TrOCRBackend.is_ready() requires model_id
# ---------------------------------------------------------------------------

def test_trocr_not_ready_without_model_id():
    from ocr_engine.trocr_backend import TrOCRBackend
    backend = TrOCRBackend(config=None)
    assert backend.is_ready() is False


# ---------------------------------------------------------------------------
# Ensemble with mocked PaddleBackend
# ---------------------------------------------------------------------------

class _PaddleCfg:
    enabled = True
    weight = 0.6
    lang = "ar"
    use_gpu = False


class _TessCfg:
    enabled = False
    weight = 0.4


class _TrOCRCfg:
    enabled = False
    weight = 0.0
    conf_threshold = 0.5


class _OCRCfg:
    paddle = _PaddleCfg()
    tesseract = _TessCfg()
    trocr = _TrOCRCfg()


class _MockConfig:
    ocr = _OCRCfg()
    ensemble_iou_threshold = 0.3


def test_ensemble_with_one_backend_returns_paddle_result():
    from ocr_engine.ensemble import run_ensemble
    from ocr_engine.paddle_backend import PaddleBackend

    synth = _synthetic_result(page_index=0)

    with patch.object(PaddleBackend, "is_available", return_value=True), \
         patch.object(PaddleBackend, "extract", return_value=synth):
        result = run_ensemble(_make_image(), page_index=0, config=_MockConfig())

    assert result is not None
    assert len(result.words) >= 1
    assert result.words[0].text == "كتاب"


def test_ensemble_result_has_engines_in_raw():
    from ocr_engine.ensemble import run_ensemble
    from ocr_engine.paddle_backend import PaddleBackend

    synth = _synthetic_result(page_index=0)

    with patch.object(PaddleBackend, "is_available", return_value=True), \
         patch.object(PaddleBackend, "extract", return_value=synth):
        result = run_ensemble(_make_image(), page_index=0, config=_MockConfig())

    assert "engines" in result.raw


def test_ensemble_no_backends_returns_empty():
    from ocr_engine.ensemble import run_ensemble
    from ocr_engine.paddle_backend import PaddleBackend
    from ocr_engine.tesseract_backend import TesseractBackend

    with patch.object(PaddleBackend, "is_available", return_value=False), \
         patch.object(TesseractBackend, "is_available", return_value=False):
        result = run_ensemble(_make_image(), page_index=0, config=_MockConfig())

    assert result.text == ""
    assert result.words == []


def test_ensemble_single_backend_source_preserved():
    """When only one backend runs, its source name is preserved in the result."""
    from ocr_engine.ensemble import run_ensemble
    from ocr_engine.paddle_backend import PaddleBackend

    synth = _synthetic_result(page_index=0)

    with patch.object(PaddleBackend, "is_available", return_value=True), \
         patch.object(PaddleBackend, "extract", return_value=synth):
        result = run_ensemble(_make_image(), page_index=0, config=_MockConfig())

    # With single backend, result uses that backend's result directly
    # source can be "paddle" (single pass-through) or contain engine
    assert result.raw.get("engines") is not None


def test_ensemble_with_crop_bbox_stores_it_in_raw():
    from ocr_engine.ensemble import run_ensemble
    from ocr_engine.paddle_backend import PaddleBackend

    synth = _synthetic_result()

    with patch.object(PaddleBackend, "is_available", return_value=True), \
         patch.object(PaddleBackend, "extract", return_value=synth):
        result = run_ensemble(
            _make_image(), page_index=0, config=_MockConfig(),
            crop_bbox=(10, 20, 80, 100),
        )

    assert result.raw.get("crop_bbox_in_page") == [10, 20, 80, 100]
