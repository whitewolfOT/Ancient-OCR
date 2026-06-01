"""Tests for all 6 shared Pydantic data contracts."""
import pytest
import json

from ocr_engine.schema import WordToken, OCRResult
from confidence_engine.state import (
    LexiconEntry, Candidate, TokenState, FeedbackEntry
)


# ---------------------------------------------------------------------------
# WordToken
# ---------------------------------------------------------------------------

def test_word_token_instantiation():
    tok = WordToken(
        text="كتاب",
        confidence=0.9,
        bbox=(10, 20, 30, 40),
        page_index=0,
        source="paddle",
    )
    assert tok.text == "كتاب"
    assert tok.confidence == 0.9
    assert tok.bbox == (10, 20, 30, 40)
    assert tok.page_index == 0
    assert tok.source == "paddle"
    assert tok.region_id is None


def test_word_token_region_id():
    tok = WordToken(
        text="word", confidence=0.5, bbox=(0, 0, 5, 5),
        page_index=1, source="tesseract", region_id="r1",
    )
    assert tok.region_id == "r1"


def test_word_token_roundtrip():
    tok = WordToken(
        text="كتاب", confidence=0.75, bbox=(0, 0, 10, 10),
        page_index=0, source="ensemble",
    )
    serialized = tok.model_dump_json()
    data = json.loads(serialized)
    tok2 = WordToken(**data)
    assert tok2.text == tok.text
    assert tok2.confidence == tok.confidence
    assert tok2.bbox == tok.bbox


def test_word_token_rejects_bad_confidence():
    with pytest.raises(Exception):
        WordToken(text="x", confidence="not_a_float", bbox=(0, 0, 1, 1),
                  page_index=0, source="paddle")


def test_word_token_rejects_bad_bbox():
    with pytest.raises(Exception):
        WordToken(text="x", confidence=0.5, bbox="not_a_tuple",
                  page_index=0, source="paddle")


# ---------------------------------------------------------------------------
# OCRResult
# ---------------------------------------------------------------------------

def test_ocr_result_instantiation():
    tok = WordToken(text="abc", confidence=0.8, bbox=(0, 0, 5, 5),
                    page_index=0, source="paddle")
    result = OCRResult(
        text="abc", words=[tok], confidence=0.8,
        page_index=0, source="paddle",
    )
    assert result.text == "abc"
    assert len(result.words) == 1
    assert result.raw == {}


def test_ocr_result_roundtrip():
    result = OCRResult(
        text="hello", words=[], confidence=0.5,
        page_index=2, source="tesseract", raw={"extra": 1},
    )
    data = json.loads(result.model_dump_json())
    result2 = OCRResult(**data)
    assert result2.page_index == 2
    assert result2.raw == {"extra": 1}


def test_ocr_result_rejects_bad_words():
    with pytest.raises(Exception):
        OCRResult(text="x", words="not_a_list", confidence=0.5,
                  page_index=0, source="paddle")


# ---------------------------------------------------------------------------
# LexiconEntry
# ---------------------------------------------------------------------------

def test_lexicon_entry_instantiation():
    entry = LexiconEntry(
        lemma="كتاب", root="كتب", pattern="فِعَال",
        gloss="book", source="_fixture", era="classical",
        priority=10,
    )
    assert entry.lemma == "كتاب"
    assert entry.era == "classical"
    assert entry.examples == []
    assert entry.domain is None


def test_lexicon_entry_roundtrip():
    entry = LexiconEntry(
        lemma="علم", root="علم", gloss="knowledge",
        source="_fixture", era="classical", priority=9,
        examples=["طلب العلم"],
    )
    data = json.loads(entry.model_dump_json())
    entry2 = LexiconEntry(**data)
    assert entry2.examples == ["طلب العلم"]


def test_lexicon_entry_rejects_bad_priority():
    with pytest.raises(Exception):
        LexiconEntry(
            lemma="x", gloss="x", source="x", era="classical",
            priority="not_int",
        )


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

def test_candidate_instantiation():
    cand = Candidate(text="كتاب", reason="identity")
    assert cand.text == "كتاب"
    assert cand.reason == "identity"
    assert cand.lexicon_entries == []
    assert cand.features == {}
    assert cand.score is None


def test_candidate_with_entries():
    entry = LexiconEntry(
        lemma="كتاب", gloss="book", source="_fixture",
        era="classical", priority=10,
    )
    cand = Candidate(
        text="كتاب", reason="identity",
        lexicon_entries=[entry],
        features={"lexicon_score": 0.9},
        score=0.85,
    )
    assert len(cand.lexicon_entries) == 1
    assert cand.score == 0.85


def test_candidate_roundtrip():
    cand = Candidate(text="خير", reason="spelling_variant", score=0.6)
    data = json.loads(cand.model_dump_json())
    cand2 = Candidate(**data)
    assert cand2.text == "خير"
    assert cand2.score == 0.6


def test_candidate_rejects_bad_text():
    with pytest.raises(Exception):
        Candidate(text=123, reason="identity")


# ---------------------------------------------------------------------------
# TokenState
# ---------------------------------------------------------------------------

def test_token_state_instantiation():
    ts = TokenState(
        original="كتاب",
        normalized="كتاب",
        normalization_log=[],
        candidates=[],
        selected="كتاب",
        confidence=0.9,
        sources=[],
        decision="accept",
        reason_code="conf_accept_90",
    )
    assert ts.original == "كتاب"
    assert ts.decision == "accept"
    assert ts.bbox is None
    assert ts.page_index == 0


def test_token_state_with_bbox():
    ts = TokenState(
        original="test", normalized="test",
        normalization_log=[{"step": "alef", "before": "أ", "after": "ا", "rule": "alef_variant_fold"}],
        candidates=[],
        selected="test", confidence=0.75,
        sources=["_fixture"], decision="accept_with_note",
        reason_code="conf_accept_with_note_75",
        bbox=(5, 10, 20, 15), page_index=2,
    )
    assert ts.bbox == (5, 10, 20, 15)
    assert ts.page_index == 2


def test_token_state_roundtrip():
    ts = TokenState(
        original="x", normalized="x",
        normalization_log=[],
        candidates=[],
        selected="x", confidence=0.5,
        sources=[], decision="uncertain",
        reason_code="conf_uncertain_50",
    )
    data = json.loads(ts.model_dump_json())
    ts2 = TokenState(**data)
    assert ts2.decision == "uncertain"


def test_token_state_rejects_bad_confidence():
    with pytest.raises(Exception):
        TokenState(
            original="x", normalized="x",
            normalization_log=[], candidates=[],
            selected="x", confidence="bad",
            sources=[], decision="accept",
            reason_code="x",
        )


# ---------------------------------------------------------------------------
# FeedbackEntry
# ---------------------------------------------------------------------------

def test_feedback_entry_instantiation():
    fe = FeedbackEntry(
        id="uuid-001",
        image_path="/tmp/crop.png",
        bbox=(0, 0, 20, 10),
        page_index=0,
        predicted="كتاب",
        ground_truth="كتب",
        source_file="doc.pdf",
        submitted_at="2026-06-01T10:00:00Z",
    )
    assert fe.predicted == "كتاب"
    assert fe.applied is False


def test_feedback_entry_applied_flag():
    fe = FeedbackEntry(
        id="uuid-002",
        image_path="/tmp/crop2.png",
        bbox=(1, 2, 3, 4),
        page_index=1,
        predicted="x",
        ground_truth="y",
        source_file="doc.pdf",
        submitted_at="2026-06-01T11:00:00Z",
        applied=True,
    )
    assert fe.applied is True


def test_feedback_entry_roundtrip():
    fe = FeedbackEntry(
        id="uuid-003",
        image_path="/tmp/crop3.png",
        bbox=(0, 0, 5, 5),
        page_index=0,
        predicted="a",
        ground_truth="b",
        source_file="x.pdf",
        submitted_at="2026-06-01T12:00:00Z",
    )
    data = json.loads(fe.model_dump_json())
    fe2 = FeedbackEntry(**data)
    assert fe2.id == "uuid-003"
    assert fe2.bbox == (0, 0, 5, 5)


def test_feedback_entry_rejects_missing_field():
    with pytest.raises(Exception):
        FeedbackEntry(
            id="x",
            image_path="/tmp/x.png",
            # missing bbox
            page_index=0,
            predicted="a",
            ground_truth="b",
            source_file="f",
            submitted_at="2026-01-01T00:00:00Z",
        )
