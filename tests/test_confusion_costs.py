"""Tests for weighted confusion costs in candidate_generator.py."""
from __future__ import annotations

import pytest

from lexicon_engine.candidate_generator import (
    _load_confusion_costs,
    _build_confusion_pair_map,
    _confusion_similarity,
    _CONFUSION_CLASSES,
)


# ---------------------------------------------------------------------------
# _load_confusion_costs
# ---------------------------------------------------------------------------

def test_load_confusion_costs_no_config_returns_defaults():
    costs = _load_confusion_costs(None)
    assert "default" in costs
    assert costs["default"] == 0.85
    for k in _CONFUSION_CLASSES:
        assert k in costs


def test_load_confusion_costs_reads_from_config():
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.lexicon.confusion_costs.dot_pairs = 0.78
    cfg.lexicon.confusion_costs.body_pairs = 0.74
    cfg.lexicon.confusion_costs.qaf_faa = 0.77
    cfg.lexicon.confusion_costs.emphatic_pairs = 0.81
    cfg.lexicon.confusion_costs.tail_pairs = 0.84
    cfg.lexicon.confusion_costs.taa_haa = 0.68
    cfg.lexicon.confusion_costs.default = 0.85
    costs = _load_confusion_costs(cfg)
    assert costs["dot_pairs"] == pytest.approx(0.78)
    assert costs["taa_haa"] == pytest.approx(0.68)
    assert costs["default"] == pytest.approx(0.85)


def test_load_confusion_costs_bad_config_falls_back():
    """AttributeError on config access → returns safe defaults."""
    class _Bad:
        @property
        def lexicon(self):
            raise AttributeError("no lexicon")
    costs = _load_confusion_costs(_Bad())
    assert "default" in costs


# ---------------------------------------------------------------------------
# _build_confusion_pair_map
# ---------------------------------------------------------------------------

def test_pair_map_contains_dot_pair():
    costs = {"dot_pairs": 0.78, "body_pairs": 0.74, "qaf_faa": 0.77,
             "emphatic_pairs": 0.81, "tail_pairs": 0.84, "taa_haa": 0.68, "default": 0.85}
    pair_map = _build_confusion_pair_map(costs)
    # ب and ن are both in dot_pairs
    assert ("ب", "ن") in pair_map
    assert ("ن", "ب") in pair_map
    assert pair_map[("ب", "ن")] == pytest.approx(0.78)


def test_pair_map_symmetric():
    costs = {k: 0.75 for k in _CONFUSION_CLASSES}
    pair_map = _build_confusion_pair_map(costs)
    for (a, b), v in list(pair_map.items()):
        assert pair_map.get((b, a)) == v


def test_pair_map_zero_for_non_confused_pair():
    costs = {k: 0.8 for k in _CONFUSION_CLASSES}
    pair_map = _build_confusion_pair_map(costs)
    # ك and ف are not in the same confusion class
    assert pair_map.get(("ك", "ف"), 0.0) == 0.0


# ---------------------------------------------------------------------------
# _confusion_similarity
# ---------------------------------------------------------------------------

def test_confusion_similarity_identical():
    assert _confusion_similarity("كتاب", "كتاب", {}) == 1.0


def test_confusion_similarity_empty():
    assert _confusion_similarity("", "", {}) == 1.0
    assert _confusion_similarity("", "ب", {}) == 0.0
    assert _confusion_similarity("ب", "", {}) == 0.0


def test_confusion_similarity_no_pair_map_matches_standard():
    """Without confusion pairs, should behave like standard Levenshtein similarity."""
    # "كتاب" vs "كتان": 1 substitution out of 4 chars → raw similarity = 0.75
    sim = _confusion_similarity("كتاب", "كتان", {})
    assert sim == pytest.approx(0.75)


def test_confusion_similarity_confusion_pair_boosts_score():
    """ب↔ن is a dot_pair (weight 0.78) → confusion sim > raw sim."""
    costs = {k: 0.0 for k in _CONFUSION_CLASSES}
    costs["dot_pairs"] = 0.78
    pair_map = _build_confusion_pair_map(costs)
    raw_sim = _confusion_similarity("كتاب", "كتان", {})     # no confusion map
    conf_sim = _confusion_similarity("كتاب", "كتان", pair_map)
    assert conf_sim > raw_sim
    # sub_cost = 1 - 0.78 = 0.22 → dist = 0.22 → sim = 1 - 0.22/4 ≈ 0.945
    assert conf_sim == pytest.approx(1.0 - 0.22 / 4, abs=0.01)


def test_confusion_similarity_non_confusion_pair_unchanged():
    """ك↔ف is not in any confusion class — no boost."""
    costs = {k: 0.78 for k in _CONFUSION_CLASSES}
    pair_map = _build_confusion_pair_map(costs)
    sim_with = _confusion_similarity("كتاب", "فتاب", pair_map)
    sim_without = _confusion_similarity("كتاب", "فتاب", {})
    assert sim_with == pytest.approx(sim_without)


def test_confusion_similarity_threshold_accepts_dot_pair():
    """With dot_pairs=0.78, ب↔ن should exceed default threshold 0.85."""
    costs = _load_confusion_costs(None)
    costs["dot_pairs"] = 0.78
    pair_map = _build_confusion_pair_map(costs)
    sv_threshold = costs["default"]   # 0.85
    csim = _confusion_similarity("كتاب", "كتان", pair_map)
    assert csim >= sv_threshold, f"expected {csim:.3f} >= {sv_threshold}"


def test_confusion_similarity_random_substitution_below_threshold():
    """A non-confusion substitution should remain below threshold."""
    costs = _load_confusion_costs(None)
    pair_map = _build_confusion_pair_map(costs)
    sv_threshold = costs["default"]  # 0.85
    # 2 random substitutions in a 4-char word → raw sim = 0.5
    csim = _confusion_similarity("كتاب", "علمن", pair_map)
    assert csim < sv_threshold
