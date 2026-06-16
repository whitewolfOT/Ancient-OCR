"""Tests for lexicon_engine/ralm_matrices.py"""
import json
import pytest
from pathlib import Path

from lexicon_engine.ralm_matrices import (
    build_w_base,
    load_m_domain,
    save_m_domain,
    update_m_domain,
    compute_affinity,
)

LANES_PATH = Path("data/lexicons/lanes")


# ── W_base ────────────────────────────────────────────────────────────────────

def test_w_base_loads_or_builds(tmp_path):
    """build_w_base on the real lanes dir produces a non-empty dict."""
    out = tmp_path / "w_base.json"
    result = build_w_base(LANES_PATH, out)
    assert isinstance(result, dict)
    assert len(result) > 0
    assert out.exists()


def test_w_base_values_in_range(tmp_path):
    """All w_base values in (0, 1]."""
    out = tmp_path / "w_base.json"
    result = build_w_base(LANES_PATH, out)
    for root, val in result.items():
        assert 0 < val <= 1.0, f"Root '{root}' has value {val} outside (0,1]"


def test_w_base_max_is_one(tmp_path):
    """Min-max normalization: highest-frequency root scores 1.0."""
    out = tmp_path / "w_base.json"
    result = build_w_base(LANES_PATH, out)
    assert max(result.values()) == pytest.approx(1.0)


def test_w_base_persisted(tmp_path):
    """JSON output is valid and matches returned dict."""
    out = tmp_path / "w_base.json"
    result = build_w_base(LANES_PATH, out)
    loaded = json.loads(out.read_text(encoding='utf-8'))
    assert loaded == result


def test_w_base_empty_dir(tmp_path):
    """Empty directory → empty dict (no crash)."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = build_w_base(empty_dir, tmp_path / "out.json")
    assert result == {}


# ── M_domain ─────────────────────────────────────────────────────────────────

def test_load_m_domain_missing(tmp_path):
    """Missing file returns empty dict."""
    result = load_m_domain(tmp_path / "nonexistent.json")
    assert result == {}


def test_save_and_load_roundtrip(tmp_path):
    """save_m_domain then load_m_domain returns equal dict."""
    path = tmp_path / "m.json"
    matrix = {"كتب": {"أرض": 0.5, "ماء": 0.5}}
    save_m_domain(matrix, path)
    loaded = load_m_domain(path)
    assert loaded == matrix


def test_update_below_min_trust_ignored():
    """Trust score below min_trust → matrix unchanged."""
    matrix = {}
    result = update_m_domain("ماء", ["أرض"], user_trust_score=0.1,
                              matrix=matrix, min_trust=0.3)
    assert result == {}


def test_update_adds_entry():
    """High-trust correction adds an entry for the corrected root."""
    matrix = {}
    result = update_m_domain("ماء", ["أرض"], user_trust_score=0.9,
                              matrix=matrix, learning_rate=0.1, min_trust=0.3)
    assert len(result) > 0


def test_update_row_normalized():
    """After update, row values sum to 1.0."""
    matrix = {}
    result = update_m_domain("كتاب", ["علم", "أرض"], user_trust_score=0.8,
                              matrix=matrix, learning_rate=0.1, min_trust=0.3)
    for row in result.values():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-6)


def test_update_no_context_no_change():
    """If all context words yield root=None, matrix unchanged."""
    matrix = {}
    result = update_m_domain("كتاب", ["123", "abc"], user_trust_score=0.9,
                              matrix=matrix, learning_rate=0.1, min_trust=0.3)
    assert result == {}


def test_update_unknown_corrected_word():
    """If corrected word has no root, matrix unchanged."""
    matrix = {}
    result = update_m_domain("النممولي", ["كتاب"], user_trust_score=0.9,
                              matrix=matrix, learning_rate=0.1, min_trust=0.3)
    # النممولي has no valid root → no update
    assert result == {}


# ── compute_affinity ──────────────────────────────────────────────────────────

def test_affinity_invented_word_returns_epsilon():
    """Invented OCR garbage with no root → epsilon (≤ 0.05)."""
    score = compute_affinity("النممولي", [], w_base={}, m_domain={})
    assert score <= 0.05


def test_affinity_known_word_no_context():
    """Known word in w_base with no context → uses W_base value."""
    w_base = {"كتب": 0.8}
    score = compute_affinity("كتاب", [], w_base=w_base, m_domain={})
    assert score > 0.05


def test_affinity_neutral_when_no_domain():
    """When M_domain has no entry for root, m_score defaults to 1.0."""
    w_base = {"مأي": 0.6}   # root of ماء in some analyses
    score_no_domain = compute_affinity("ماء", ["أرض"], w_base=w_base, m_domain={})
    score_empty_ctx = compute_affinity("ماء", [], w_base=w_base, m_domain={})
    # Both should produce the same result (M_domain neutral = 1.0)
    assert score_no_domain == pytest.approx(score_empty_ctx, abs=1e-6)


def test_affinity_clamped_to_one():
    """Score never exceeds 1.0."""
    w_base = {"كتب": 1.0}
    m_domain = {"كتب": {"أرض": 1.0}}
    score = compute_affinity("كتاب", ["أرض"], w_base=w_base, m_domain=m_domain)
    assert score <= 1.0


def test_affinity_returns_float():
    """Return type is always float."""
    score = compute_affinity("كتاب", [], w_base={}, m_domain={})
    assert isinstance(score, float)
