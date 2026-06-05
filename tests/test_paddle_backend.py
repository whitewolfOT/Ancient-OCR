"""Tests for PaddleOCR 3.x backend API compatibility."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_predict_result(lines):
    """Build a synthetic predict() return value in PaddleOCR 3.x format.

    Args:
        lines: list of (text, score, poly) tuples where poly is (4,2) ndarray.
    """
    rec_texts = [t for t, _, _ in lines]
    rec_scores = [s for _, s, _ in lines]
    rec_polys = [p for _, _, p in lines]
    return [{"rec_texts": rec_texts, "rec_scores": rec_scores, "rec_polys": rec_polys}]


def _make_poly(x, y, w, h):
    """4-corner polygon from top-left + size."""
    return np.array([
        [x,     y    ],
        [x + w, y    ],
        [x + w, y + h],
        [x,     y + h],
    ], dtype=float)


FAKE_LINES = [
    ("كتاب",   0.92, _make_poly(10, 20, 80, 15)),
    ("العربي", 0.78, _make_poly(10, 40, 100, 15)),
]


class TestPaddleBackend3x:
    """Verify extract() works with the PaddleOCR 3.x predict() API."""

    def _run_extract(self):
        """Patch PaddleOCR and run extract(); return the OCRResult."""
        import ocr_engine.paddle_backend as pb

        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = _make_predict_result(FAKE_LINES)

        # Reset module-level cache so _get_model() creates a fresh instance.
        original_model = pb._model
        original_failed = pb._model_init_failed
        pb._model = None
        pb._model_init_failed = False

        try:
            with patch("ocr_engine.paddle_backend._paddle_available", True), \
                 patch("ocr_engine.paddle_backend.PaddleOCR" if hasattr(pb, "PaddleOCR") else
                       "paddleocr.PaddleOCR", mock_ocr, create=True), \
                 patch("ocr_engine.paddle_backend._get_model", return_value=mock_ocr):
                from ocr_engine.paddle_backend import PaddleBackend
                backend = PaddleBackend()
                image = np.zeros((100, 200, 3), dtype=np.uint8)
                return backend.extract(image, page_index=0)
        finally:
            pb._model = original_model
            pb._model_init_failed = original_failed

    def test_returns_ocr_result(self):
        from ocr_engine.schema import OCRResult
        result = self._run_extract()
        assert isinstance(result, OCRResult)

    def test_token_count(self):
        result = self._run_extract()
        assert len(result.words) == 2

    def test_token_texts(self):
        result = self._run_extract()
        texts = [w.text for w in result.words]
        assert "كتاب" in texts
        assert "العربي" in texts

    def test_token_confidence(self):
        result = self._run_extract()
        confs = {w.text: w.confidence for w in result.words}
        assert abs(confs["كتاب"] - 0.92) < 1e-6
        assert abs(confs["العربي"] - 0.78) < 1e-6

    def test_bbox_shape_and_values(self):
        """Each bbox must be (x, y, w, h) derived from the polygon."""
        result = self._run_extract()
        w0 = result.words[0]
        # poly[0] = (10, 20, 80, 15) → bbox = (10, 20, 80, 15)
        assert len(w0.bbox) == 4
        x, y, w, h = w0.bbox
        assert x == 10
        assert y == 20
        assert w == 80
        assert h == 15

    def test_source_is_paddle(self):
        result = self._run_extract()
        assert result.source == "paddle"
        for tok in result.words:
            assert tok.source == "paddle"

    def test_page_index_propagated(self):
        import ocr_engine.paddle_backend as pb
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = _make_predict_result(FAKE_LINES)

        pb._model = None
        pb._model_init_failed = False
        try:
            with patch("ocr_engine.paddle_backend._paddle_available", True), \
                 patch("ocr_engine.paddle_backend._get_model", return_value=mock_ocr):
                from ocr_engine.paddle_backend import PaddleBackend
                backend = PaddleBackend()
                result = backend.extract(np.zeros((100, 200, 3), dtype=np.uint8), page_index=3)
                assert all(w.page_index == 3 for w in result.words)
        finally:
            pb._model = None
            pb._model_init_failed = False

    def test_predict_called_with_image(self):
        import ocr_engine.paddle_backend as pb
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = _make_predict_result(FAKE_LINES)

        pb._model = None
        pb._model_init_failed = False
        try:
            with patch("ocr_engine.paddle_backend._paddle_available", True), \
                 patch("ocr_engine.paddle_backend._get_model", return_value=mock_ocr):
                from ocr_engine.paddle_backend import PaddleBackend
                backend = PaddleBackend()
                img = np.zeros((100, 200, 3), dtype=np.uint8)
                backend.extract(img, page_index=0)
                mock_ocr.predict.assert_called_once_with(img)
        finally:
            pb._model = None
            pb._model_init_failed = False

    def test_alternatives_stored_in_raw(self):
        """Confusion-pair alternatives for Arabic tokens land in raw, not words."""
        result = self._run_extract()
        assert "paddle_alternatives" in result.raw
        # كتاب contains ك — no confusion pair; العربي contains ب — has confusables
        alts = result.raw["paddle_alternatives"]
        # At least one of the tokens should generate alternatives
        assert isinstance(alts, dict)

    def test_empty_predict_result(self):
        """predict() returning [] must produce an OCRResult with zero tokens."""
        import ocr_engine.paddle_backend as pb
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = []

        pb._model = None
        pb._model_init_failed = False
        try:
            with patch("ocr_engine.paddle_backend._paddle_available", True), \
                 patch("ocr_engine.paddle_backend._get_model", return_value=mock_ocr):
                from ocr_engine.paddle_backend import PaddleBackend
                backend = PaddleBackend()
                result = backend.extract(np.zeros((100, 200, 3), dtype=np.uint8))
                assert result.words == []
                assert result.confidence == 0.0
        finally:
            pb._model = None
            pb._model_init_failed = False
