"""Tests for review_export.py Kraken baseline coordinate support."""
from __future__ import annotations

import numpy as np
import pytest

from confidence_engine.state import TokenState, build_token_state, Candidate
from output.review_export import export_review_queue, to_review_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts(decision="review_required", bbox=(10, 20, 30, 15),
             page_index=0, baseline=None, line_id=None):
    candidates = [
        Candidate(text="كتاب", reason="identity", score=0.4),
    ]
    ts = TokenState(
        original="كتب",
        normalized="كتب",
        normalization_log=[],
        candidates=candidates,
        selected="كتب",
        confidence=0.4,
        sources=[],
        decision=decision,
        reason_code="low_confidence",
        bbox=bbox,
        page_index=page_index,
        line_id=line_id,
        baseline=baseline,
    )
    return ts


# ---------------------------------------------------------------------------
# TokenState baseline fields
# ---------------------------------------------------------------------------

def test_token_state_has_baseline_field():
    ts = _make_ts(baseline=[(10, 20), (50, 22)])
    assert ts.baseline == [(10, 20), (50, 22)]


def test_token_state_has_line_id_field():
    ts = _make_ts(line_id="kraken-line-001")
    assert ts.line_id == "kraken-line-001"


def test_token_state_baseline_defaults_none():
    ts = _make_ts()
    assert ts.baseline is None
    assert ts.line_id is None


# ---------------------------------------------------------------------------
# export_review_queue — baseline included in items
# ---------------------------------------------------------------------------

def test_export_includes_baseline_when_present():
    ts = _make_ts(
        baseline=[(10, 20), (50, 22), (50, 30), (10, 28)],
        line_id="kraken-line-007",
    )
    queue = export_review_queue([ts], None, "test.pdf")
    assert queue["review_count"] == 1
    item = queue["items"][0]
    assert item["baseline"] is not None
    assert item["baseline"][0] == [10, 20]
    assert item["line_id"] == "kraken-line-007"


def test_export_baseline_none_when_absent():
    ts = _make_ts()
    queue = export_review_queue([ts], None, "test.pdf")
    item = queue["items"][0]
    assert item["baseline"] is None
    assert item["line_id"] is None


def test_export_accepted_tokens_excluded():
    ts_accept = _make_ts(decision="accept")
    ts_review = _make_ts(decision="review_required")
    queue = export_review_queue([ts_accept, ts_review], None, "test.pdf")
    assert queue["review_count"] == 1
    assert queue["items"][0]["decision"] == "review_required"


# ---------------------------------------------------------------------------
# Crop with baseline overlay — smoke test (no file I/O assertion needed)
# ---------------------------------------------------------------------------

def test_crop_with_baseline_does_not_crash():
    """_extract_crop with baseline draws overlay without raising."""
    import tempfile, os
    from output.review_export import _extract_crop

    page = np.full((200, 300, 3), 200, dtype=np.uint8)
    ts = _make_ts(
        bbox=(10, 20, 80, 30),
        page_index=0,
        baseline=[(10, 25), (50, 26), (90, 25)],
    )
    with tempfile.TemporaryDirectory() as d:
        path = _extract_crop(ts, [page], d, "tok_001")
    assert path is not None
    assert path.endswith(".png")


def test_crop_without_baseline_works():
    import tempfile
    from output.review_export import _extract_crop

    page = np.full((200, 300, 3), 200, dtype=np.uint8)
    ts = _make_ts(bbox=(0, 0, 50, 40), page_index=0)
    with tempfile.TemporaryDirectory() as d:
        path = _extract_crop(ts, [page], d, "tok_002")
    assert path is not None


def test_crop_no_image_pages_returns_none():
    from output.review_export import _extract_crop
    ts = _make_ts(bbox=(0, 0, 10, 10))
    assert _extract_crop(ts, None, "/tmp", "tok") is None


# ---------------------------------------------------------------------------
# to_review_json — baseline serialises correctly
# ---------------------------------------------------------------------------

def test_review_json_contains_baseline():
    ts = _make_ts(baseline=[(5, 10), (30, 12)], line_id="ln-42")
    queue = export_review_queue([ts], None, "doc.pdf")
    j = to_review_json(queue)
    assert '"baseline"' in j
    assert '"line_id"' in j
    assert "ln-42" in j
