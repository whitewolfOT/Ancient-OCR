"""Tests for CER/WER functions in training/feedback_store.py."""
from __future__ import annotations

import pytest

from training.feedback_store import _edit_distance, _cer, _wer, cer_wer, submit, stats
from confidence_engine.state import FeedbackEntry


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

def test_edit_distance_equal():
    assert _edit_distance(list("abc"), list("abc")) == 0


def test_edit_distance_empty_vs_nonempty():
    assert _edit_distance([], list("abc")) == 3
    assert _edit_distance(list("abc"), []) == 3


def test_edit_distance_both_empty():
    assert _edit_distance([], []) == 0


def test_edit_distance_single_sub():
    assert _edit_distance(list("cat"), list("bat")) == 1


def test_edit_distance_insertion():
    assert _edit_distance(list("cat"), list("cats")) == 1


def test_edit_distance_deletion():
    assert _edit_distance(list("cats"), list("cat")) == 1


# ---------------------------------------------------------------------------
# _cer
# ---------------------------------------------------------------------------

def test_cer_identical():
    assert _cer("كتاب", "كتاب") == pytest.approx(0.0)


def test_cer_empty_ref_empty_hyp():
    assert _cer("", "") == pytest.approx(0.0)


def test_cer_empty_ref_nonempty_hyp():
    assert _cer("", "كتاب") == pytest.approx(1.0)


def test_cer_nonempty_ref_empty_hyp():
    # 4 deletions / 4 chars = 1.0
    assert _cer("كتاب", "") == pytest.approx(1.0)


def test_cer_one_substitution():
    # "كتاب" vs "كتان" — 1 sub out of 4 chars
    assert _cer("كتاب", "كتان") == pytest.approx(0.25)


def test_cer_clamped_to_one():
    # hyp much longer than ref — clamped to 1.0
    assert _cer("ب", "ب" * 100) <= 1.0


# ---------------------------------------------------------------------------
# _wer
# ---------------------------------------------------------------------------

def test_wer_identical():
    assert _wer("hello world", "hello world") == pytest.approx(0.0)


def test_wer_one_substitution():
    # "hello world" vs "hello there" — 1 word sub / 2 words = 0.5
    assert _wer("hello world", "hello there") == pytest.approx(0.5)


def test_wer_empty_ref_empty_hyp():
    assert _wer("", "") == pytest.approx(0.0)


def test_wer_empty_ref_nonempty_hyp():
    assert _wer("", "word") == pytest.approx(1.0)


def test_wer_arabic():
    # "كتاب العلم" vs "كتاب" — 1 deletion / 2 = 0.5
    assert _wer("كتاب العلم", "كتاب") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# cer_wer() — integration with feedback store
# ---------------------------------------------------------------------------

def test_cer_wer_empty_store(tmp_path, monkeypatch):
    monkeypatch.setattr("training.feedback_store._db_path", lambda *_: str(tmp_path / "fb.db"))
    result = cer_wer()
    assert result == {"cer": 0.0, "wer": 0.0, "sample_size": 0}


def test_cer_wer_with_entries(tmp_path, monkeypatch):
    import uuid
    from datetime import datetime, timezone

    db = str(tmp_path / "fb.db")
    monkeypatch.setattr("training.feedback_store._db_path", lambda *_: db)

    for pred, gt in [("كتب", "كتاب"), ("علم", "علم"), ("بيت", "بيت")]:
        entry = FeedbackEntry(
            id=str(uuid.uuid4()), image_path="", bbox=(0, 0, 10, 10),
            page_index=0, predicted=pred, ground_truth=gt,
            source_file="test.pdf",
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )
        submit(entry)

    result = cer_wer()
    assert result["sample_size"] == 3
    assert 0.0 <= result["cer"] <= 1.0
    assert 0.0 <= result["wer"] <= 1.0
    # "علم" and "بيت" are perfect → average CER < 0.5
    assert result["cer"] < 0.5


# ---------------------------------------------------------------------------
# stats() — includes cer and wer keys
# ---------------------------------------------------------------------------

def test_stats_includes_cer_wer_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("training.feedback_store._db_path", lambda *_: str(tmp_path / "fb.db"))
    result = stats()
    assert "cer" in result
    assert "wer" in result


def test_stats_cer_wer_with_data(tmp_path, monkeypatch):
    import uuid
    from datetime import datetime, timezone

    db = str(tmp_path / "fb.db")
    monkeypatch.setattr("training.feedback_store._db_path", lambda *_: db)

    entry = FeedbackEntry(
        id=str(uuid.uuid4()), image_path="", bbox=(0, 0, 10, 10),
        page_index=0, predicted="كتب", ground_truth="كتاب",
        source_file="test.pdf",
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )
    submit(entry)

    result = stats()
    assert result["cer"] > 0.0
    assert result["wer"] > 0.0
    assert result["total"] == 1
