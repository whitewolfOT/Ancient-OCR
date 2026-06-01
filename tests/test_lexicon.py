"""Tests for lexicon pipeline: ingestion, query, candidate generation, ranking, context scoring.

Calls ingest_source("_fixture") to populate the DB before lexicon-dependent tests.
Uses a temporary DB path so tests don't corrupt the real index.
"""
import os
import tempfile
import pytest

# ---------------------------------------------------------------------------
# Helpers: isolated temp DB config
# ---------------------------------------------------------------------------

class _LexCfg:
    """Minimal config stub that redirects the lexicon DB to a temp file."""
    def __init__(self, db_path):
        class _index:
            path = db_path
            approximate_match_threshold = 0.8
        class _lexicon:
            index = _index()
        self.lexicon = _lexicon()


def _temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# ---------------------------------------------------------------------------
# Fixture: ingest _fixture source into an isolated DB once per module
# ---------------------------------------------------------------------------

import pytest

@pytest.fixture(scope="module")
def populated_cfg():
    """Return a config pointing to a temp DB that has _fixture loaded."""
    db_path = _temp_db()
    cfg = _LexCfg(db_path)

    # Reset index singleton so it rebuilds against our temp DB
    import lexicon_ingestion.index_builder as ib
    import lexicon_ingestion.storage as st

    # Force ingestion into temp DB
    from lexicon_ingestion.sources import get_source
    from lexicon_ingestion.parser import parse_source
    source = get_source("_fixture")
    entries = parse_source(source)
    st.save_entries(entries, "_fixture", cfg)

    # Rebuild index against temp DB
    ib._index_singleton = None
    ib.get_index(cfg, force_rebuild=True)

    yield cfg

    # Cleanup
    ib._index_singleton = None
    try:
        os.unlink(db_path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# query_engine
# ---------------------------------------------------------------------------

def test_query_known_word_returns_entries(populated_cfg):
    from lexicon_engine.query_engine import query
    results = query("كتاب", config=populated_cfg)
    assert len(results) >= 1


def test_query_unknown_word_returns_empty(populated_cfg):
    from lexicon_engine.query_engine import query
    results = query("zzz_unknown_xyz", config=populated_cfg)
    assert results == []


def test_query_results_are_lexicon_entries(populated_cfg):
    from lexicon_engine.query_engine import query
    from confidence_engine.state import LexiconEntry
    results = query("كتاب", config=populated_cfg)
    for r in results:
        assert isinstance(r, LexiconEntry)


# ---------------------------------------------------------------------------
# candidate_generator
# ---------------------------------------------------------------------------

def test_generate_returns_list(populated_cfg):
    from lexicon_engine.candidate_generator import generate
    from ocr_engine.schema import WordToken

    token = WordToken(
        text="كتاب", confidence=0.8,
        bbox=(0, 0, 10, 10), page_index=0, source="paddle",
    )
    candidates = generate(token, None, config=populated_cfg)
    assert isinstance(candidates, list)


def test_generate_has_identity_candidate(populated_cfg):
    from lexicon_engine.candidate_generator import generate
    from ocr_engine.schema import WordToken

    token = WordToken(
        text="كتاب", confidence=0.8,
        bbox=(0, 0, 10, 10), page_index=0, source="paddle",
    )
    candidates = generate(token, None, config=populated_cfg)
    reasons = [c.reason for c in candidates]
    assert "identity" in reasons


def test_generate_identity_candidate_reason(populated_cfg):
    from lexicon_engine.candidate_generator import generate
    from ocr_engine.schema import WordToken

    token = WordToken(
        text="كتاب", confidence=0.8,
        bbox=(0, 0, 10, 10), page_index=0, source="paddle",
    )
    candidates = generate(token, None, config=populated_cfg)
    identity = [c for c in candidates if c.reason == "identity"]
    assert len(identity) >= 1
    assert identity[0].text == "كتاب"


# ---------------------------------------------------------------------------
# ranker — identity wins tiebreak over spelling_variant at same score
# ---------------------------------------------------------------------------

def test_ranking_identity_wins_tiebreak():
    from lexicon_engine.ranker import rank
    from confidence_engine.state import Candidate

    identity = Candidate(text="كتاب", reason="identity", score=0.75)
    variant = Candidate(text="كتب", reason="spelling_variant", score=0.75)

    result = rank([variant, identity])
    assert result.best.reason == "identity"


def test_ranking_higher_score_wins():
    from lexicon_engine.ranker import rank
    from confidence_engine.state import Candidate

    low = Candidate(text="a", reason="identity", score=0.5)
    high = Candidate(text="b", reason="spelling_variant", score=0.9)

    result = rank([low, high])
    assert result.best.text == "b"


def test_ranking_empty_returns_empty():
    from lexicon_engine.ranker import rank
    result = rank([])
    assert result.best is None
    assert result.ranked == []
    assert result.selected_text == ""


# ---------------------------------------------------------------------------
# context_scorer — empty context returns fallback 0.5
# ---------------------------------------------------------------------------

def test_context_score_empty_context_returns_fallback():
    from lexicon_engine.context_scorer import context_score
    score = context_score("كتاب", [], [])
    assert score == 0.5


def test_context_score_in_range():
    from lexicon_engine.context_scorer import context_score
    score = context_score("كتاب", ["قرأ"], ["الطالب"])
    assert 0.0 <= score <= 1.0


def test_context_score_never_raises():
    from lexicon_engine.context_scorer import context_score
    try:
        context_score("xyznotreal", ["a"], ["b"])
    except Exception:
        pytest.fail("context_score raised an exception")
