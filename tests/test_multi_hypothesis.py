"""Tests for multi-hypothesis OCR alternatives via Arabic confusion pairs."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _generate_alternatives in paddle_backend
# ---------------------------------------------------------------------------

class TestGenerateAlternatives:
    def test_ba_generates_confusable_alternatives(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        alts = _generate_alternatives("بسم", primary_conf=0.8, top_n=10)
        alt_texts = [a["text"] for a in alts]
        # ب confuses with ت ث ن ي → expect at least one of these
        assert any(t.startswith(c) for t in alt_texts for c in ["ت", "ث", "ن", "ي"]), \
            f"Expected ب-confusion alternatives, got: {alt_texts}"

    def test_alternatives_have_lower_confidence(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        primary_conf = 0.9
        alts = _generate_alternatives("بسم", primary_conf=primary_conf, top_n=5)
        assert len(alts) > 0
        for alt in alts:
            assert alt["confidence"] < primary_conf

    def test_alternatives_confidence_ratio(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        alts = _generate_alternatives("بسم", primary_conf=1.0, top_n=5)
        for alt in alts:
            assert abs(alt["confidence"] - 0.85) < 1e-6

    def test_alternatives_source_is_paddle_alt(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        alts = _generate_alternatives("بسم", primary_conf=0.8, top_n=5)
        for alt in alts:
            assert alt["source"] == "paddle_alt"

    def test_top_n_limit_respected(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        alts = _generate_alternatives("بسم", primary_conf=0.8, top_n=2)
        assert len(alts) <= 2

    def test_no_alternatives_for_non_confusable_token(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        # Pure Latin — no Arabic confusion pairs apply
        alts = _generate_alternatives("abc", primary_conf=0.9, top_n=3)
        assert alts == []

    def test_primary_text_not_in_alternatives(self):
        from ocr_engine.paddle_backend import _generate_alternatives
        alts = _generate_alternatives("بسم", primary_conf=0.8, top_n=10)
        assert all(a["text"] != "بسم" for a in alts)


# ---------------------------------------------------------------------------
# OCRResult.raw carries paddle_alternatives
# ---------------------------------------------------------------------------

class TestPaddleBackendRaw:
    def test_ocr_result_raw_contains_paddle_alternatives(self):
        from ocr_engine.paddle_backend import PaddleBackend
        import numpy as np

        backend = PaddleBackend()
        mock_raw = [{"rec_texts": ["بسم"], "rec_scores": [0.8], "rec_polys": [
            np.array([[0,0],[10,0],[10,10],[0,10]])
        ]}]

        with patch.object(backend, "_config", None):
            with patch("ocr_engine.paddle_backend._get_model") as mock_model:
                mock_model.return_value = MagicMock(predict=MagicMock(return_value=mock_raw))
                result = backend.extract(np.zeros((50, 50, 3), dtype=np.uint8))

        assert "paddle_alternatives" in result.raw
        assert "بسم" in result.raw["paddle_alternatives"]
        alts = result.raw["paddle_alternatives"]["بسم"]
        assert len(alts) > 0
        assert all(a["confidence"] < 0.8 for a in alts)


# ---------------------------------------------------------------------------
# candidate_generator uses ocr_alternatives
# ---------------------------------------------------------------------------

class TestCandidateGeneratorOCRAlternatives:
    def _make_token(self, text: str):
        from ocr_engine.schema import WordToken
        return WordToken(text=text, confidence=0.5, bbox=(0, 0, 10, 10),
                         page_index=0, source="paddle")

    def test_ocr_alternatives_appear_as_candidates(self, tmp_path, monkeypatch):
        """Alternatives passed in appear as ocr_alternative candidates."""
        from lexicon_engine import candidate_generator

        token = self._make_token("بسم")
        alts = [{"text": "تسم", "confidence": 0.68, "source": "paddle_alt"}]

        with patch("lexicon_engine.query_engine.query", return_value=[]):
            with patch("lexicon_ingestion.index_builder.get_index") as mock_idx:
                mock_idx.return_value = MagicMock(all_lemmas=[], by_root={})
                with patch("lexicon_engine.query_engine._trigram_narrow", return_value=[]):
                    cands = candidate_generator.generate(
                        token, None, config=None, ocr_alternatives=alts
                    )

        reasons = [c.reason for c in cands]
        assert "ocr_alternative" in reasons, f"Expected ocr_alternative, got: {reasons}"

    def test_ocr_alternative_confidence_below_primary(self, tmp_path):
        from lexicon_engine import candidate_generator
        from ocr_engine.paddle_backend import _generate_alternatives

        token = self._make_token("بسم")
        alts = _generate_alternatives("بسم", primary_conf=0.9, top_n=3)

        with patch("lexicon_engine.query_engine.query", return_value=[]):
            with patch("lexicon_ingestion.index_builder.get_index") as mock_idx:
                mock_idx.return_value = MagicMock(all_lemmas=[], by_root={})
                with patch("lexicon_engine.query_engine._trigram_narrow", return_value=[]):
                    cands = candidate_generator.generate(
                        token, None, config=None, ocr_alternatives=alts
                    )

        alt_cands = [c for c in cands if c.reason == "ocr_alternative"]
        assert len(alt_cands) > 0
        for ac in alt_cands:
            assert ac.text != "بسم"

    def test_no_ocr_alternatives_is_backward_compatible(self, tmp_path):
        """generate() with no ocr_alternatives arg works identically to before."""
        from lexicon_engine import candidate_generator
        token = self._make_token("بسم")

        with patch("lexicon_engine.query_engine.query", return_value=[]):
            with patch("lexicon_ingestion.index_builder.get_index") as mock_idx:
                mock_idx.return_value = MagicMock(all_lemmas=[], by_root={})
                with patch("lexicon_engine.query_engine._trigram_narrow", return_value=[]):
                    cands = candidate_generator.generate(token, None, config=None)

        assert isinstance(cands, list)
        assert all(c.reason != "ocr_alternative" for c in cands)
