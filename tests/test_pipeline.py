"""End-to-end integration tests for main.process_file and related exports.

All OCR backends are mocked. A synthetic 200x400 BGR white image is used.
The fixture lexicon is loaded into a temp DB so the pipeline has lexicon data.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from ocr_engine.schema import OCRResult, WordToken


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(h=200, w=400):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _synthetic_result(page_index=0):
    tok = WordToken(
        text="كتاب", confidence=0.8,
        bbox=(10, 10, 50, 20),
        page_index=page_index, source="paddle",
    )
    return OCRResult(
        text="كتاب", words=[tok], confidence=0.8,
        page_index=page_index, source="paddle",
        raw={},
    )


class _LexCfg:
    def __init__(self, db_path):
        _dp = db_path
        class _idx:
            approximate_match_threshold = 0.8
        class _lex:
            db_path = _dp
            index = _idx()
            sources = ["_fixture"]
        self.lexicon = _lex()


@pytest.fixture(scope="module")
def temp_db_path():
    """Temp SQLite DB pre-populated with _fixture lexicon."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    cfg = _LexCfg(path)

    from lexicon_ingestion.sources import get_source
    from lexicon_ingestion.parser import parse_source
    import lexicon_ingestion.storage as st
    import lexicon_ingestion.index_builder as ib

    source = get_source("_fixture")
    entries = parse_source(source)
    st.save_entries(entries, "_fixture", cfg)

    ib._index_singleton = None
    ib.get_index(cfg, force_rebuild=True)

    yield path

    ib._index_singleton = None
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# run_pipeline directly (bypasses file I/O)
# ---------------------------------------------------------------------------

def _make_pages():
    return [{"image": _make_image(), "page_index": 0}]


def _run_with_mock_ocr(pages, mode, db_path):
    from main import run_pipeline
    from ocr_engine.ensemble import run_ensemble

    synth = _synthetic_result(page_index=0)

    cfg = _LexCfg(db_path)

    # Patch run_ensemble to return our synthetic result
    with patch("main.run_pipeline.__code__", run_pipeline.__code__), \
         patch("ocr_engine.ensemble.run_ensemble", return_value=synth):
        result = run_pipeline(pages, mode=mode, cfg=cfg)
    return result


# ---------------------------------------------------------------------------
# Mode: clean
# ---------------------------------------------------------------------------

def test_clean_mode_has_text_key(temp_db_path):
    from main import run_pipeline
    from ocr_engine import ensemble as ens_mod

    synth = _synthetic_result()
    pages = _make_pages()

    with patch.object(ens_mod, "run_ensemble", return_value=synth):
        result = run_pipeline(pages, mode="clean", cfg=_LexCfg(temp_db_path))

    assert "text" in result


def test_clean_mode_returns_dict(temp_db_path):
    from main import run_pipeline
    from ocr_engine import ensemble as ens_mod

    synth = _synthetic_result()
    pages = _make_pages()

    with patch.object(ens_mod, "run_ensemble", return_value=synth):
        result = run_pipeline(pages, mode="clean", cfg=_LexCfg(temp_db_path))

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Mode: annotated
# ---------------------------------------------------------------------------

def test_annotated_mode_has_tokens_key(temp_db_path):
    from main import run_pipeline
    from ocr_engine import ensemble as ens_mod

    synth = _synthetic_result()
    pages = _make_pages()

    with patch.object(ens_mod, "run_ensemble", return_value=synth):
        result = run_pipeline(pages, mode="annotated", cfg=_LexCfg(temp_db_path))

    assert "tokens" in result


# ---------------------------------------------------------------------------
# Mode: debug
# ---------------------------------------------------------------------------

def test_debug_mode_has_tokens_and_raw_ocr(temp_db_path):
    from main import run_pipeline
    from ocr_engine import ensemble as ens_mod

    synth = _synthetic_result()
    pages = _make_pages()

    with patch.object(ens_mod, "run_ensemble", return_value=synth):
        result = run_pipeline(pages, mode="debug", cfg=_LexCfg(temp_db_path))

    assert "tokens" in result
    assert "raw_ocr" in result


# ---------------------------------------------------------------------------
# Mode: invalid
# ---------------------------------------------------------------------------

def test_invalid_mode_raises():
    from main import run_pipeline
    from ocr_engine import ensemble as ens_mod

    synth = _synthetic_result()
    pages = _make_pages()

    with pytest.raises((ValueError, Exception)):
        with patch.object(ens_mod, "run_ensemble", return_value=synth):
            run_pipeline(pages, mode="invalid_mode", cfg=None)


# ---------------------------------------------------------------------------
# review_export
# ---------------------------------------------------------------------------

def test_review_export_uncertain_tokens_appear_in_queue():
    from output.review_export import export_review_queue
    from confidence_engine.state import TokenState, Candidate

    ts_uncertain = TokenState(
        original="كتاب", normalized="كتاب",
        normalization_log=[],
        candidates=[Candidate(text="كتاب", reason="identity", score=0.55)],
        selected="كتاب", confidence=0.55,
        sources=[], decision="uncertain",
        reason_code="conf_uncertain_55",
        bbox=(0, 0, 10, 10), page_index=0,
    )
    ts_accept = TokenState(
        original="علم", normalized="علم",
        normalization_log=[],
        candidates=[Candidate(text="علم", reason="identity", score=0.95)],
        selected="علم", confidence=0.95,
        sources=[], decision="accept",
        reason_code="conf_accept_95",
        bbox=(20, 0, 10, 10), page_index=0,
    )

    queue = export_review_queue(
        [ts_uncertain, ts_accept],
        source_image_pages=None,
        source_file="test.pdf",
    )
    assert queue["review_count"] == 1
    assert queue["items"][0]["decision"] == "uncertain"


def test_review_export_review_required_included():
    from output.review_export import export_review_queue
    from confidence_engine.state import TokenState, Candidate

    ts = TokenState(
        original="xxx", normalized="xxx",
        normalization_log=[],
        candidates=[],
        selected="xxx", confidence=0.3,
        sources=[], decision="review_required",
        reason_code="conf_review_required_30",
        bbox=(0, 0, 10, 10), page_index=0,
    )
    queue = export_review_queue([ts], source_image_pages=None, source_file="f.pdf")
    assert queue["review_count"] == 1


def test_review_export_accept_tokens_excluded():
    from output.review_export import export_review_queue
    from confidence_engine.state import TokenState, Candidate

    ts_accept = TokenState(
        original="abc", normalized="abc",
        normalization_log=[],
        candidates=[],
        selected="abc", confidence=0.95,
        sources=[], decision="accept",
        reason_code="conf_accept_95",
    )
    queue = export_review_queue([ts_accept], source_image_pages=None, source_file="f.pdf")
    assert queue["review_count"] == 0


# ---------------------------------------------------------------------------
# Morphology round-trip
# ---------------------------------------------------------------------------

def test_morphology_roundtrip_rule_based_to_lexicon(temp_db_path):
    """Rule-based root extraction of كتب feeds directly into lexicon index lookup."""
    from morphology.root_extractor import extract_root
    from lexicon_ingestion.index_builder import get_index
    import lexicon_ingestion.index_builder as ib

    cfg = _LexCfg(temp_db_path)
    candidates = extract_root("كتب", cfg)
    assert candidates, "extract_root returned empty list"
    top_root = candidates[0].root

    ib._index_singleton = None
    idx = get_index(cfg, force_rebuild=True)
    entries = idx.by_root.get(top_root, [])
    assert len(entries) > 0, f"No lexicon entries found for root {top_root!r}"


# ---------------------------------------------------------------------------
# Tier 3 false positive documentation
# ---------------------------------------------------------------------------

def test_tier3_high_confidence_ocr_accepted_unconditionally(temp_db_path):
    """High OCR confidence (>=0.90) bypasses lexicon + morphology.

    Arabic tokens that survive the noise filter and carry OCR confidence >=0.90
    are accepted as-is with reason_code='ocr_confident'.
    """
    from main import run_pipeline
    from ocr_engine import ensemble as ens_mod
    from ocr_engine.schema import OCRResult, WordToken

    # Valid Arabic multi-consonant token — passes noise filter, confidence >=0.90 → Tier 3
    high_conf_tok = WordToken(
        text="كتاب", confidence=0.95,
        bbox=(0, 0, 40, 20),
        page_index=0, source="paddle",
    )
    ocr_result = OCRResult(
        text="كتاب", words=[high_conf_tok], confidence=0.95,
        page_index=0, source="paddle", raw={},
    )
    pages = [{"image": _make_image(), "page_index": 0}]
    cfg = _LexCfg(temp_db_path)

    with patch.object(ens_mod, "run_ensemble", return_value=ocr_result):
        result = run_pipeline(pages, mode="annotated", cfg=cfg)

    tokens = [t for t in result.get("tokens", []) if t["decision"] != "reject"]
    assert tokens, "Expected at least one non-rejected token in annotated output"
    tok = tokens[0]
    assert tok["decision"] == "accept", f"Expected accept, got {tok['decision']!r}"
    assert tok["reason_code"] == "ocr_confident", f"Expected ocr_confident, got {tok['reason_code']!r}"
