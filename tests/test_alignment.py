"""Tests for alignment modules: string_similarity, bbox_alignment, token_matcher."""
import pytest

from alignment.string_similarity import similarity
from alignment.bbox_alignment import iou, align_by_bbox
from alignment.token_matcher import match_tokens, TokenCluster
from ocr_engine.schema import WordToken


def _token(text, x=0, y=0, w=10, h=10, conf=0.8, source="paddle"):
    return WordToken(
        text=text, confidence=conf,
        bbox=(x, y, w, h),
        page_index=0, source=source,
    )


# ---------------------------------------------------------------------------
# string_similarity.similarity
# ---------------------------------------------------------------------------

def test_similarity_identical():
    assert similarity("كتاب", "كتاب") == 1.0


def test_similarity_both_empty():
    assert similarity("", "") == 1.0


def test_similarity_one_empty():
    assert similarity("", "a") == 0.0


def test_similarity_reversed_empty():
    assert similarity("a", "") == 0.0


def test_similarity_different_strings():
    s = similarity("كتاب", "علم")
    assert 0.0 <= s <= 1.0


def test_similarity_near_match():
    # single char difference — should be > 0.5
    s = similarity("كتاب", "كتب")
    assert s > 0.5


# ---------------------------------------------------------------------------
# bbox_alignment.iou
# ---------------------------------------------------------------------------

def test_iou_identical_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_non_overlapping():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_partial_overlap():
    v = iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert 0.0 < v < 1.0


def test_iou_one_inside_other():
    # (0,0,20,20) contains (5,5,5,5) — intersection = 25, union = 400+25-25 = 400
    v = iou((0, 0, 20, 20), (5, 5, 5, 5))
    assert 0.0 < v < 1.0


# ---------------------------------------------------------------------------
# bbox_alignment.align_by_bbox
# ---------------------------------------------------------------------------

def test_align_two_matching_token_lists():
    ta1 = _token("ا", x=0, y=0, w=10, h=10)
    ta2 = _token("ب", x=50, y=0, w=10, h=10)
    tb1 = _token("ا", x=0, y=0, w=10, h=10)
    tb2 = _token("ب", x=50, y=0, w=10, h=10)

    pairs = align_by_bbox([ta1, ta2], [tb1, tb2])
    matched = [(a, b) for a, b in pairs if a is not None and b is not None]
    assert len(matched) == 2


def test_align_empty_tokens_a():
    tb = _token("x", x=0, y=0)
    pairs = align_by_bbox([], [tb])
    assert len(pairs) == 1
    assert pairs[0][0] is None
    assert pairs[0][1] is tb


def test_align_empty_tokens_b():
    ta = _token("x", x=0, y=0)
    pairs = align_by_bbox([ta], [])
    assert len(pairs) == 1
    assert pairs[0][0] is ta
    assert pairs[0][1] is None


def test_align_no_overlap_produces_unmatched():
    ta = _token("a", x=0, y=0, w=10, h=10)
    tb = _token("b", x=100, y=100, w=10, h=10)
    pairs = align_by_bbox([ta], [tb])
    matched = [(a, b) for a, b in pairs if a is not None and b is not None]
    assert len(matched) == 0


# ---------------------------------------------------------------------------
# token_matcher.match_tokens
# ---------------------------------------------------------------------------

def test_match_tokens_two_lists_produces_clusters():
    list_a = [_token("كتاب", x=0, y=0, w=20, h=15),
               _token("علم", x=50, y=0, w=20, h=15)]
    list_b = [_token("كتاب", x=0, y=0, w=20, h=15),
               _token("علم", x=50, y=0, w=20, h=15)]

    clusters = match_tokens([list_a, list_b])
    assert len(clusters) == 2


def test_match_tokens_agreement_in_range():
    list_a = [_token("كتاب", x=0, y=0, w=20, h=15)]
    list_b = [_token("كتاب", x=0, y=0, w=20, h=15)]
    clusters = match_tokens([list_a, list_b])
    for cluster in clusters:
        assert 0.0 <= cluster.agreement <= 1.0


def test_match_tokens_single_list_returns_one_cluster_per_token():
    tokens = [_token("a", x=i * 20, y=0) for i in range(3)]
    clusters = match_tokens([tokens])
    assert len(clusters) == 3


def test_match_tokens_empty_lists():
    clusters = match_tokens([[], []])
    assert clusters == []


def test_match_tokens_clusters_are_token_clusters():
    list_a = [_token("x", x=0, y=0)]
    clusters = match_tokens([list_a])
    assert all(isinstance(c, TokenCluster) for c in clusters)
