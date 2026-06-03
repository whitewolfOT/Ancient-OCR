"""Tests for _parse_khorsi_sql() using synthetic SQL fixtures only."""
from __future__ import annotations

from pathlib import Path

import pytest

from lexicon_ingestion.sources import SourceConfig


def _source(path: str) -> SourceConfig:
    return SourceConfig(
        name="khorsi_roots",
        enabled=True,
        priority=5,
        era="classical",
        license="cc_by_sa_3",
        path=path,
        parser_adapter="khorsi_sql",
        domain=None,
        book_name="Arabic Roots and Derivatives (from Taj al-Arus)",
    )


def _write_sql(tmp_path: Path, value_lines: list[str]) -> SourceConfig:
    """Write a minimal MySQL dump with the given VALUES lines."""
    header = (
        "-- phpMyAdmin SQL Dump\n"
        "/*!40101 SET NAMES utf8 */;\n"
        "CREATE TABLE IF NOT EXISTS `KhorsiCorpus` (\n"
        "  `id` int(11) NOT NULL AUTO_INCREMENT,\n"
        "  `root` varchar(10) COLLATE cp1256_bin NOT NULL,\n"
        "  `word` varchar(40) COLLATE cp1256_bin NOT NULL,\n"
        "  `unvowelword` varchar(30) COLLATE cp1256_bin NOT NULL,\n"
        "  `nonormstem` varchar(20) COLLATE cp1256_bin NOT NULL\n"
        ") ENGINE=MyISAM DEFAULT CHARSET=cp1256;\n"
        "INSERT INTO `KhorsiCorpus` (`id`, `root`, `word`, `unvowelword`, `nonormstem`) VALUES\n"
    )
    body = ",\n".join(value_lines) + ";\n"
    sql_file = tmp_path / "KhorsiCorpus.sql"
    sql_file.write_text(header + body, encoding="utf-8")
    return _source(str(sql_file))


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_basic_row_parses_root_lemma_pattern(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'كتب', 'كَتَبَ', 'كتب', 'كتب')",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1
    e = entries[0]
    assert e.root == "كتب"
    assert e.lemma == "كَتَبَ"
    assert e.pattern == "كتب"
    assert e.gloss == ""
    assert e.source == "khorsi_roots"
    assert e.era == "classical"
    assert e.examples == []


def test_arabic_strings_decode_correctly(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'أبأ', 'الأَبَاءَةُ', 'الأباءة', 'ءباء')",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1
    assert entries[0].root == "أبأ"
    assert entries[0].lemma == "الأَبَاءَةُ"
    assert entries[0].pattern == "ءباء"


# ---------------------------------------------------------------------------
# NULL and empty pattern handling
# ---------------------------------------------------------------------------

def test_null_nonormstem_gives_pattern_none(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'رجل', 'الرَّجُلُ', 'الرجل', NULL)",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1
    assert entries[0].pattern is None


def test_empty_nonormstem_gives_pattern_none(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'علم', 'عَلِمَ', 'علم', '')",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1
    assert entries[0].pattern is None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_lemma_root_deduplicates(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'كتب', 'كَتَبَ', 'كتب', 'كتب')",
        "(2, 'كتب', 'كَتَبَ', 'كتب', 'كتب')",  # exact duplicate
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# Row filtering
# ---------------------------------------------------------------------------

def test_empty_root_row_is_skipped(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, '', 'كَتَبَ', 'كتب', 'كتب')",
        "(2, 'علم', 'عَلِمَ', 'علم', '')",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1
    assert entries[0].root == "علم"


def test_empty_word_row_is_skipped(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'كتب', '', 'كتب', 'كتب')",
        "(2, 'علم', 'عَلِمَ', 'علم', '')",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)
    assert len(entries) == 1
    assert entries[0].root == "علم"


# ---------------------------------------------------------------------------
# Full 5-row fixture
# ---------------------------------------------------------------------------

def test_five_row_fixture_count_and_dedup(tmp_path):
    src = _write_sql(tmp_path, [
        "(1, 'كتب', 'كَتَبَ', 'كتب', 'كتب')",
        "(2, 'كتب', 'كَتَبَ', 'كتب', 'كتب')",    # duplicate of row 1
        "(3, 'رجل', 'الرَّجُلُ', 'الرجل', NULL)",
        "(4, 'علم', 'عَلِمَ', 'علم', '')",
        "(5, 'بيت', 'البَيْتُ', 'البيت', 'بيت')",
    ])
    from lexicon_ingestion.parser import _parse_khorsi_sql
    entries = _parse_khorsi_sql(src)

    assert len(entries) == 4          # row 2 deduped

    by_root = {e.root: e for e in entries}
    assert "كتب" in by_root
    assert "رجل" in by_root
    assert "علم" in by_root
    assert "بيت" in by_root

    assert by_root["كتب"].pattern == "كتب"
    assert by_root["رجل"].pattern is None
    assert by_root["علم"].pattern is None
    assert by_root["بيت"].pattern == "بيت"


# ---------------------------------------------------------------------------
# Missing data path
# ---------------------------------------------------------------------------

def test_missing_sql_file_returns_empty(tmp_path):
    src = _source(str(tmp_path / "nonexistent.sql"))
    from lexicon_ingestion.parser import _parse_khorsi_sql
    assert _parse_khorsi_sql(src) == []
