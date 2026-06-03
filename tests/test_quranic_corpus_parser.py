"""Tests for _parse_quranic_corpus_tsv() using synthetic fixtures in new format.

New file format (mustafa0x/quran-morphology):
  LOCATION   FORM   TAG   FEATURES
  1:1:1:2    اسْمِ  N     ROOT:سمو|LEM:اسْم|M|GEN

TAG is POS code: N (noun/adj/pn), V (verb), P (particle — skip).
Location has no parens. source.path is the file path, not a directory.
"""
from __future__ import annotations

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
        book_name="Quranic Arabic Corpus (mustafa0x/quran-morphology)",
    )


def _write_tsv(tmp_path: Path, lines: list[str]) -> SourceConfig:
    """Write tab-separated lines directly to quran-morphology.txt, return SourceConfig."""
    tsv = tmp_path / "quran-morphology.txt"
    tsv.write_text("\n".join(lines), encoding="utf-8")
    return _source(str(tsv))


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_noun_row_parses_lemma_root_pattern(tmp_path):
    src = _write_tsv(tmp_path, [
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    e = entries[0]
    assert e.lemma == "اسْم"
    assert e.root == "سمو"
    assert e.pattern == "N"
    assert e.examples == ["1:1:1"]
    assert e.source == "quranic_corpus"
    assert e.era == "classical"
    assert e.domain == "quranic"
    assert e.gloss == ""


def test_verb_row_parses_correctly(tmp_path):
    src = _write_tsv(tmp_path, [
        "1:5:2:1\tنَعْبُدُ\tV\tIMPF|VF:1|ROOT:عبد|LEM:عَبَدَ|1P|MOOD:IND",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert entries[0].root == "عبد"
    assert entries[0].pattern == "V"


# ---------------------------------------------------------------------------
# Row filtering
# ---------------------------------------------------------------------------

def test_particle_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "1:1:1:1\tبِ\tP\tP|PREF|LEM:ب",
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert entries[0].lemma == "اسْم"


def test_missing_root_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "1:5:3:1\tوَ\tN\tLEM:وَ|CONJ",              # no ROOT
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert entries[0].lemma == "اسْم"


def test_missing_lem_row_is_skipped(tmp_path):
    src = _write_tsv(tmp_path, [
        "1:5:1:1\tكَتَبَ\tV\tROOT:كتب|PERF",         # no LEM
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
    ])
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Deduplication and examples
# ---------------------------------------------------------------------------

def test_same_lemma_root_deduplicates_to_one_entry(tmp_path):
    src = _write_tsv(tmp_path, [
        "3:1:1:1\tخَلَقَ\tV\tROOT:خلق|LEM:خَلَقَ|PERF",
        "3:5:3:1\tخَلَقَ\tV\tROOT:خلق|LEM:خَلَقَ|PERF",
        "7:2:1:1\tخَلَقَ\tV\tROOT:خلق|LEM:خَلَقَ|PERF",
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
        f"{i}:1:1:1\tكَتَبَ\tV\tROOT:كتب|LEM:كَتَبَ|PERF"
        for i in range(1, 9)   # 8 occurrences
    ]
    src = _write_tsv(tmp_path, rows)
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)
    assert len(entries) == 1
    assert len(entries[0].examples) == 5


# ---------------------------------------------------------------------------
# Multi-row fixture
# ---------------------------------------------------------------------------

def test_mixed_fixture_count_and_dedup(tmp_path):
    rows = [
        "1:1:1:1\tبِ\tP\tP|PREF|LEM:ب",              # P → skip
        "1:1:1:2\tاسْمِ\tN\tROOT:سمو|LEM:اسْم|M|GEN",
        "1:1:2:1\tاللَّهِ\tN\tPN|ROOT:أله|LEM:اللَّه|GEN",
        "1:1:3:2\tرَّحْمَٰنِ\tN\tROOT:رحم|LEM:رَحْمٰن|MS|GEN|ADJ",
        "3:1:1:1\tخَلَقَ\tV\tROOT:خلق|LEM:خَلَقَ|PERF",
        "3:5:3:1\tخَلَقَ\tV\tROOT:خلق|LEM:خَلَقَ|PERF",  # dup of above
        "4:1:1:1\tوَ\tN\tLEM:وَ|CONJ",                # no ROOT → skip
        "5:1:1:1\tكَتَبَ\tV\tROOT:كتب|LEM:كَتَبَ|PERF",
    ]
    src = _write_tsv(tmp_path, rows)
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    entries = _parse_quranic_corpus_tsv(src)

    # P skipped (1), no-ROOT skipped (1), خلق deduped → 5 unique entries
    assert len(entries) == 5

    by_lemma = {e.lemma: e for e in entries}
    assert "اسْم" in by_lemma
    assert "اللَّه" in by_lemma
    assert "رَحْمٰن" in by_lemma
    assert "خَلَقَ" in by_lemma
    assert "كَتَبَ" in by_lemma
    assert "وَ" not in by_lemma

    assert len(by_lemma["خَلَقَ"].examples) == 2


# ---------------------------------------------------------------------------
# Missing data path
# ---------------------------------------------------------------------------

def test_missing_file_returns_empty(tmp_path):
    src = _source(str(tmp_path / "nonexistent.txt"))
    from lexicon_ingestion.parser import _parse_quranic_corpus_tsv
    assert _parse_quranic_corpus_tsv(src) == []
