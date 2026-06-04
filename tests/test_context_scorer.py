"""Tests for build_quranic_bigrams() and ngram model Quranic corpus merge."""
from __future__ import annotations

from pathlib import Path
import pytest


def _write_morphology(tmp_path: Path, lines: list[str]) -> Path:
    f = tmp_path / "quran-morphology.txt"
    f.write_text("\n".join(lines), encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# build_quranic_bigrams — unit tests
# ---------------------------------------------------------------------------

def test_basic_bigrams_from_two_verses(tmp_path):
    lines = [
        # verse 1:1 — two content words
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
        "1:1:2:1\tاللَّهِ\tN\tPN|ROOT:أله|LEM:اللَّه|GEN",
        # verse 2:1 — one content word (no bigram within verse)
        "2:1:1:1\tكَتَبَ\tV\tROOT:كتب|LEM:كَتَبَ|PERF",
    ]
    path = _write_morphology(tmp_path, lines)
    from lexicon_engine.context_scorer import build_quranic_bigrams
    result = build_quranic_bigrams(str(path))

    bigrams = result["bigrams"]
    unigrams = result["unigrams"]

    assert ("اسْم", "اللَّه") in bigrams
    assert bigrams[("اسْم", "اللَّه")] == 1
    assert "اسْم" in unigrams
    assert "اللَّه" in unigrams
    assert "كَتَبَ" in unigrams
    # No bigram within verse 2:1 (only one token)
    assert ("اللَّه", "كَتَبَ") not in bigrams


def test_particle_rows_skipped(tmp_path):
    lines = [
        "1:1:1:1\tبِ\tP\tP|PREF|LEM:ب",        # P — skip
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
        "1:1:2:1\tاللَّهِ\tN\tPN|ROOT:أله|LEM:اللَّه|GEN",
    ]
    path = _write_morphology(tmp_path, lines)
    from lexicon_engine.context_scorer import build_quranic_bigrams
    result = build_quranic_bigrams(str(path))

    bigrams = result["bigrams"]
    # ب skipped — bigram should be اسْم→اللَّه, not ب→اسْم
    assert ("اسْم", "اللَّه") in bigrams
    assert not any("ب" in str(k) for k in bigrams)


def test_missing_lem_rows_skipped(tmp_path):
    lines = [
        "1:1:1:1\tكَتَبَ\tV\tROOT:كتب|PERF",     # no LEM — skip
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
        "1:1:2:1\tاللَّهِ\tN\tPN|ROOT:أله|LEM:اللَّه|GEN",
    ]
    path = _write_morphology(tmp_path, lines)
    from lexicon_engine.context_scorer import build_quranic_bigrams
    result = build_quranic_bigrams(str(path))

    bigrams = result["bigrams"]
    assert ("اسْم", "اللَّه") in bigrams
    assert len(bigrams) == 1


def test_repeated_bigram_across_verses_increments_count(tmp_path):
    lines = [
        # Same bigram appears in 3 different verses
        "1:1:1:1\tرَبِّ\tN\tROOT:ربب|LEM:رَبّ|M|GEN",
        "1:1:1:2\tالْعَالَمِينَ\tN\tROOT:علم|LEM:عالَم|MP|GEN",
        "2:2:1:1\tرَبِّ\tN\tROOT:ربب|LEM:رَبّ|M|GEN",
        "2:2:1:2\tالْعَالَمِينَ\tN\tROOT:علم|LEM:عالَم|MP|GEN",
        "3:3:1:1\tرَبِّ\tN\tROOT:ربب|LEM:رَبّ|M|GEN",
        "3:3:1:2\tالْعَالَمِينَ\tN\tROOT:علم|LEM:عالَم|MP|GEN",
    ]
    path = _write_morphology(tmp_path, lines)
    from lexicon_engine.context_scorer import build_quranic_bigrams
    result = build_quranic_bigrams(str(path))

    assert result["bigrams"][("رَبّ", "عالَم")] == 3
    assert result["unigrams"]["رَبّ"] == 3
    assert result["unigrams"]["عالَم"] == 3


def test_missing_file_returns_empty(tmp_path):
    from lexicon_engine.context_scorer import build_quranic_bigrams
    result = build_quranic_bigrams(str(tmp_path / "nonexistent.txt"))
    assert result == {"bigrams": {}, "unigrams": {}}


# ---------------------------------------------------------------------------
# Integration: bigrams flow into _get_ngram_model and context_score
# ---------------------------------------------------------------------------

def test_known_bigram_lifts_context_score(tmp_path):
    """Three-tier scoring: bigram hit > known-unigram > unknown word.

    The morphology file creates:
      - unigrams: {"رَبّ": 2, "فغثص": 2}
      - bigrams:  {("رَبّ", "فغثص"): 2}

    So "فغثص" is a known unigram (it appears as a LEM in the corpus).
    The three distinguishable outcomes are:
      - known bigram context ("رَبّ" → "فغثص") → above 0.6
      - known unigram, unknown context ("xyz" → "فغثص") → exactly 0.6
      - truly unknown candidate, no bigram → 0.5 fallback (tested separately)
    """
    synthetic_lemma = "فغثص"
    lines = [
        "1:1:1:1\tرَبِّ\tN\tROOT:ربب|LEM:رَبّ|M|GEN",
        f"1:1:1:2\tمَثَلٌ\tN\tROOT:مثل|LEM:{synthetic_lemma}|M|NOM",
        "2:2:1:1\tرَبِّ\tN\tROOT:ربب|LEM:رَبّ|M|GEN",
        f"2:2:1:2\tمَثَلٌ\tN\tROOT:مثل|LEM:{synthetic_lemma}|M|NOM",
    ]
    morph_path = _write_morphology(tmp_path, lines)

    import lexicon_engine.context_scorer as cs_mod
    cs_mod._ngram_model = None

    class FakeLexiconCfg:
        quranic_morphology_path = str(morph_path)

    class FakeCfg:
        lexicon = FakeLexiconCfg()
        class context_scorer:
            backend = "ngram"
            fallback_score = 0.5
            min_bigram_count = 1

    from lexicon_engine.context_scorer import context_score

    # Bigram tier: known pair in context → strictly above unigram tier (0.6)
    score_bigram = context_score(synthetic_lemma, left_context=["رَبّ"], right_context=[], config=FakeCfg())
    # Unigram tier: candidate known, context unknown → exactly 0.6
    score_unigram = context_score(synthetic_lemma, left_context=["xyz"], right_context=[], config=FakeCfg())

    assert score_bigram > score_unigram          # bigram tier > unigram tier
    assert score_bigram > 0.6                    # above unigram tier
    assert score_unigram == 0.6                  # exactly unigram tier

    cs_mod._ngram_model = None


def test_ngram_score_three_tiers_direct():
    """Directly probe _ngram_score with a hand-crafted model.

    This confirms the full three-tier hierarchy including the unknown-word case,
    without going through file loading or the lexicon DB.
    """
    import lexicon_engine.context_scorer as cs_mod

    mock_model = {
        "unigrams": {"رَبّ": 10, "عالَم": 5},
        "bigrams": {("رَبّ", "عالَم"): 4},
        "total": 15,
        "vocab": 2,
    }
    original = cs_mod._ngram_model
    cs_mod._ngram_model = mock_model

    try:
        # Tier 1: bigram hit
        score_bi = cs_mod._ngram_score("عالَم", ["رَبّ"], [], config=None)
        # Tier 2: known unigram, no bigram context
        score_uni = cs_mod._ngram_score("عالَم", ["xyz"], [], config=None)
        # Tier 3: unknown candidate, no bigram
        score_unk = cs_mod._ngram_score("مجهول", ["رَبّ"], [], config=None)
    finally:
        cs_mod._ngram_model = original

    assert score_bi > score_uni          # bigram > unigram
    assert score_uni > score_unk         # unigram > fallback
    assert score_bi > 0.6               # above unigram tier
    assert score_uni == 0.6             # exactly unigram tier
    assert score_unk == 0.5             # exactly fallback
