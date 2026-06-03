"""Tests for _parse_quranic_corpus_tsv() using synthetic TSV fixtures only."""
from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from lexicon_ingestion.sources import SourceConfig


def _source(path: str) -> SourceConfig:
    return SourceConfig(
        name="quranic_corpus",
        enabled=True,
        priority=10,
        era="classical",
        license="research_open",
        path=path,
        parser_adapter="quranic_corpus_tsv",
        domain="quranic",
        book_name="Quranic Arabic Corpus v0.4",
    )


def _write_tsv(tmp_path: Path, lines: list[str]) -> SourceConfig:
    """Write tab-separated lines to a fixture file and return a SourceConfig."""
    tsv = tmp_path / "quranic-corpus-morphology-0.4.txt"
    tsv.write_text("\n".join(lines), encoding="utf-8")
    return _source(str(tmp_path))


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_stem_row_parses_lemma_root_pattern(tmp_path):
    src = _write_tsv(tmp_path, [
        "# comment",
        "(1:1:1:2)\tاسْمِ\tSTEM\tROOT:سمو|LEM:اِسْم|POS:N",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    e = entries[0]
    assert e.lemma == "اِسْم"
    assert e.root == "سمو"
    assert e.pattern == "N"
    assert e.examples == ["1:1:1"]
    assert e.source == "quranic_corpus"
    assert e.era == "classical"
    assert e.domain == "quranic"
    assert e.gloss == ""


def test_verb_stem_parses_correctly(tmp_path):
    src = _write_tsv(tmp_path, [
        "(2:1:1:1)\tخَلَقَ\tSTEM\tROOT:خلق|LEM:خَلَقَ|POS:V",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert entries[0].root == "خلق"
    assert entries[0].pattern == "V"


# ---------------------------------------------------------------------------
# Row filtering
# ---------------------------------------------------------------------------

def test_prefix_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "(1:1:1:1)\tبِ\tPREFIX\tPREFIX|B",
        "(1:1:1:2)\tاسْمِ\tSTEM\tROOT:سمو|LEM:اِسْم|POS:N",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert entries[0].lemma == "اِسْم"


def test_suffix_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "(1:1:1:3)\tهُ\tSUFFIX\tSUFFIX|PRON:3MS",
        "(1:1:1:2)\tاسْمِ\tSTEM\tROOT:سمو|LEM:اِسْم|POS:N",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1


def test_missing_root_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "(1:1:1:1)\tوَ\tSTEM\tLEM:وَ|POS:CONJ",          # no ROOT
        "(1:1:1:2)\tاسْمِ\tSTEM\tROOT:سمو|LEM:اِسْم|POS:N",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert entries[0].lemma == "اِسْم"


def test_missing_lem_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "(1:1:1:1)\tكَتَبَ\tSTEM\tROOT:كتب|POS:V",        # no LEM
        "(1:1:1:2)\tاسْمِ\tSTEM\tROOT:سمو|LEM:اِسْم|POS:N",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Deduplication and examples
# ---------------------------------------------------------------------------

def test_same_lemma_root_deduplicates_to_one_entry(tmp_path):
    src = _write_tsv(tmp_path, [
        "(3:1:1:1)\tخَلَقَ\tSTEM\tROOT:خلق|LEM:خَلَقَ|POS:V",
        "(3:5:3:1)\tخَلَقَ\tSTEM\tROOT:خلق|LEM:خَلَقَ|POS:V",
        "(7:2:1:1)\tخَلَقَ\tSTEM\tROOT:خلق|LEM:خَلَقَ|POS:V",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert len(entries[0].examples) == 3
    assert "3:1:1" in entries[0].examples
    assert "3:5:3" in entries[0].examples
    assert "7:2:1" in entries[0].examples


def test_examples_capped_at_five(tmp_path):
    rows = [
        f"({i}:1:1:1)\tكَتَبَ\tSTEM\tROOT:كتب|LEM:كَتَبَ|POS:V"
        for i in range(1, 9)   # 8 occurrences
    ]
    src = _write_tsv(tmp_path, rows)
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert len(entries[0].examples) == 5


# ---------------------------------------------------------------------------
# Full 10-row fixture (matches the real synthetic file in data/lexicons/)
# ---------------------------------------------------------------------------

def test_ten_row_fixture_produces_five_entries(tmp_path):
    rows = [
        "# QURANIC ARABIC CORPUS - MORPHOLOGICAL ANNOTATION",
        "(1:1:1:1)\tبِ\tPREFIX\tPREFIX|B",
        "(1:1:1:2)\tاسْمِ\tSTEM\tROOT:سمو|LEM:اِسْم|POS:N",
        "(1:1:1:3)\tاللَّهِ\tSTEM\tROOT:اله|LEM:اَللّٰه|POS:PN",
        "(2:1:1:1)\tالرَّحْمَٰنِ\tSTEM\tROOT:رحم|LEM:رَحْمَٰن|POS:ADJ",
        "(3:1:1:1)\tخَلَقَ\tSTEM\tROOT:خلق|LEM:خَلَقَ|POS:V",
        "(3:5:3:1)\tخَلَقَ\tSTEM\tROOT:خلق|LEM:خَلَقَ|POS:V",
        "(4:1:1:1)\tوَ\tSTEM\tLEM:وَ|POS:CONJ",            # no ROOT → skip
        "(5:1:1:1)\tكَتَبَ\tSTEM\tROOT:كتب|LEM:كَتَبَ|POS:V",
        "(6:1:1:1)\tكَتَبَ\tSTEM\tROOT:كتب|LEM:كَتَبَ|POS:V",
        "(7:1:1:1)\tكَتَبَ\tSTEM\tROOT:كتب|LEM:كَتَبَ|POS:V",
    ]
    src = _write_tsv(tmp_path, rows)
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)

    assert len(entries) == 5

    by_lemma = {e.lemma: e for e in entries}
    assert "اِسْم" in by_lemma
    assert "خَلَقَ" in by_lemma
    assert "كَتَبَ" in by_lemma
    assert "وَ" not in by_lemma              # no ROOT → skipped

    assert len(by_lemma["خَلَقَ"].examples) == 2
    assert len(by_lemma["كَتَبَ"].examples) == 3


# ---------------------------------------------------------------------------
# Missing data path
# ---------------------------------------------------------------------------

def test_missing_directory_returns_empty(tmp_path):
    src = _source(str(tmp_path / "nonexistent_dir"))
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    assert _parse_quranic_corpus_tsv(src) == []


def test_empty_directory_returns_empty(tmp_path):
    src = _source(str(tmp_path))   # dir exists but no .txt file
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    assert _parse_quranic_corpus_tsv(src) == []
