"""Tests for _parse_arabic_wordnet_lmf() using a synthetic LMF fixture."""
from __future__ import annotations

from pathlib import Path

import pytest

from lexicon_ingestion.sources import SourceConfig


def _source(path: str) -> SourceConfig:
    return SourceConfig(
        name="arabic_wordnet",
        enabled=False,
        priority=8,
        era="classical",
        license="cc_by_4",
        path=path,
        parser_adapter="arabic_wordnet_lmf",
        domain=None,
        book_name="Arabic WordNet (Global WordNet Association)",
    )


def _write_lmf(tmp_path: Path, lexical_entries: list[str], synsets: list[str] = None) -> SourceConfig:
    """Write a minimal Global WordNet LMF XML file and return a SourceConfig."""
    synset_block = "\n".join(synsets or [])
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<LexicalResource>\n'
        '  <Lexicon id="arb" language="arb">\n'
        + "\n".join(lexical_entries)
        + "\n"
        '  </Lexicon>\n'
        + synset_block + "\n"
        '</LexicalResource>\n'
    )
    wn_file = tmp_path / "wn.xml"
    wn_file.write_text(xml, encoding="utf-8")
    return _source(str(wn_file))


# ---------------------------------------------------------------------------
# Basic parsing
# ---------------------------------------------------------------------------

def test_basic_entry_parses_lemma_and_gloss(tmp_path):
    src = _write_lmf(tmp_path, [
        '    <LexicalEntry id="arb-كتاب-n">',
        '      <Lemma writtenForm="كتاب"/>',
        '      <Sense id="s1" synset="arb-01"/>',
        '    </LexicalEntry>',
    ], synsets=[
        '  <Synset id="arb-01">',
        '    <Definition>a book or written work</Definition>',
        '  </Synset>',
    ])
    from lexicon_ingestion.parser import _parse_arabic_wordnet_lmf
    entries = _parse_arabic_wordnet_lmf(src)
    assert len(entries) == 1
    e = entries[0]
    assert e.lemma == "كتاب"
    assert e.gloss == "a book or written work"
    assert e.source == "arabic_wordnet"
    assert e.era == "classical"
    assert e.examples == []
    assert e.priority == 8


def test_entry_with_no_synset_gloss_falls_back_to_empty(tmp_path):
    src = _write_lmf(tmp_path, [
        '    <LexicalEntry id="arb-قلم-n">',
        '      <Lemma writtenForm="قلم"/>',
        '      <Sense id="s2" synset="arb-99"/>',
        '    </LexicalEntry>',
    ])
    from lexicon_ingestion.parser import _parse_arabic_wordnet_lmf
    entries = _parse_arabic_wordnet_lmf(src)
    assert len(entries) == 1
    assert entries[0].lemma == "قلم"
    assert entries[0].gloss == ""


# ---------------------------------------------------------------------------
# Row filtering — non-Arabic script skipped
# ---------------------------------------------------------------------------

def test_non_arabic_lemma_is_skipped(tmp_path):
    src = _write_lmf(tmp_path, [
        '    <LexicalEntry id="arb-book-n">',
        '      <Lemma writtenForm="book"/>',
        '      <Sense id="s3" synset="arb-01"/>',
        '    </LexicalEntry>',
        '    <LexicalEntry id="arb-كتاب-n">',
        '      <Lemma writtenForm="كتاب"/>',
        '      <Sense id="s4" synset="arb-01"/>',
        '    </LexicalEntry>',
    ], synsets=[
        '  <Synset id="arb-01">',
        '    <Definition>a book</Definition>',
        '  </Synset>',
    ])
    from lexicon_ingestion.parser import _parse_arabic_wordnet_lmf
    entries = _parse_arabic_wordnet_lmf(src)
    assert len(entries) == 1
    assert entries[0].lemma == "كتاب"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_lemma_deduplicates(tmp_path):
    src = _write_lmf(tmp_path, [
        '    <LexicalEntry id="arb-كتاب-n-1">',
        '      <Lemma writtenForm="كتاب"/>',
        '      <Sense id="s5" synset="arb-01"/>',
        '    </LexicalEntry>',
        '    <LexicalEntry id="arb-كتاب-n-2">',
        '      <Lemma writtenForm="كتاب"/>',
        '      <Sense id="s6" synset="arb-02"/>',
        '    </LexicalEntry>',
    ], synsets=[
        '  <Synset id="arb-01"><Definition>sense one</Definition></Synset>',
        '  <Synset id="arb-02"><Definition>sense two</Definition></Synset>',
    ])
    from lexicon_ingestion.parser import _parse_arabic_wordnet_lmf
    entries = _parse_arabic_wordnet_lmf(src)
    assert len(entries) == 1
    assert entries[0].lemma == "كتاب"


# ---------------------------------------------------------------------------
# Multiple entries, inline Definition fallback
# ---------------------------------------------------------------------------

def test_inline_definition_fallback(tmp_path):
    src = _write_lmf(tmp_path, [
        '    <LexicalEntry id="arb-علم-n">',
        '      <Lemma writtenForm="عِلْم"/>',
        '      <Definition>knowledge or science</Definition>',
        '    </LexicalEntry>',
    ])
    from lexicon_ingestion.parser import _parse_arabic_wordnet_lmf
    entries = _parse_arabic_wordnet_lmf(src)
    assert len(entries) == 1
    assert entries[0].gloss == "knowledge or science"


# ---------------------------------------------------------------------------
# Missing data path
# ---------------------------------------------------------------------------

def test_missing_file_returns_empty(tmp_path):
    src = _source(str(tmp_path / "nonexistent.xml"))
    from lexicon_ingestion.parser import _parse_arabic_wordnet_lmf
    assert _parse_arabic_wordnet_lmf(src) == []
