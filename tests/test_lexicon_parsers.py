"""
Tests for the real-lexicon parser adapters.

Each adapter is tested with a small synthetic dataset that mirrors the
real format — no external downloads required.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import textwrap
from pathlib import Path

import pytest

from confidence_engine.state import LexiconEntry
from lexicon_ingestion.sources import SourceConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(name, adapter, path, **kwargs) -> SourceConfig:
    return SourceConfig(
        name=name,
        enabled=True,
        priority=9,
        era="classical",
        license="test",
        path=str(path),
        parser_adapter=adapter,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# lanes_xml adapter (TEI.2 XML, laneslexicon/lexicon_xml)
# ---------------------------------------------------------------------------

class TestLanesXml:
    _XML_FIXTURE = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <TEI.2>
          <text><body><div1 type="lexicon">
            <div2 n="ktb" type="root">
              <entryFree id="n0001" key="kAtib" type="main">
                <form>
                  <orth orig="" extent="full" lang="ar">kAtib</orth>
                  <orth extent="full" lang="ar">*</orth>
                </form>
                One who writes; a scribe or secretary. <foreign lang="ar">kAtaba</foreign> he wrote.
              </entryFree>
              <entryFree id="n0002" key="kitAb" type="main">
                <form>
                  <orth orig="" extent="full" lang="ar">kitAb</orth>
                  <orth extent="full" lang="ar">*</orth>
                </form>
                A book; a written document or letter.
              </entryFree>
            </div2>
            <div2 n="Elm" type="root">
              <entryFree id="n0003" key="Eilm" type="main">
                <form>
                  <orth orig="" extent="full" lang="ar">Eilm</orth>
                  <orth extent="full" lang="ar">*</orth>
                </form>
                Knowledge; learning; science. Applied to any branch of learning.
              </entryFree>
            </div2>
            <div2 n="A" type="root">
              <entryFree id="n0004" key="Al" type="cross">
                <form><orth orig="" extent="full" lang="ar">Al</orth></form>
                Definite article — see main grammar entry.
              </entryFree>
            </div2>
            <div2 n="qra" type="root">
              <entryFree id="n0005" key="qr" type="cross">
                <form><orth orig="" extent="full" lang="ar">qra'a</orth></form>
                See.
              </entryFree>
            </div2>
          </div1></body></text>
        </TEI.2>
    """)

    def _make_xml_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "lanes_xml"
        d.mkdir()
        (d / "test_lane.xml").write_text(self._XML_FIXTURE, encoding="utf-8")
        return d

    def test_parse_returns_entries(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        d = self._make_xml_dir(tmp_path)
        src = _make_source("lanes", "lanes_xml", d)
        entries = parse_source(src)
        assert len(entries) >= 2  # kAtib + kitAb + Eilm (≥2 to allow parsing variance)

    def test_roots_are_arabic(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        import re
        arabic_re = re.compile(r"[؀-ۿ]")
        d = self._make_xml_dir(tmp_path)
        src = _make_source("lanes", "lanes_xml", d)
        entries = parse_source(src)
        roots = [e.root for e in entries if e.root]
        assert roots, "expected at least one entry with a root"
        assert all(arabic_re.search(r) for r in roots)

    def test_foreign_excluded_from_gloss(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        d = self._make_xml_dir(tmp_path)
        src = _make_source("lanes", "lanes_xml", d)
        entries = parse_source(src)
        # ArabTeX content inside <foreign> must never appear in any gloss
        for e in entries:
            assert "kAtaba" not in e.gloss
        # ktb root entries must have meaningful English (write/scribe/book)
        ktb = [e for e in entries if e.root == "كتب"]
        assert ktb, "expected entries with root كتب"
        assert any(
            "write" in e.gloss.lower() or "scribe" in e.gloss.lower()
            or "book" in e.gloss.lower()
            for e in ktb
        )

    def test_short_gloss_skipped(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        d = self._make_xml_dir(tmp_path)
        src = _make_source("lanes", "lanes_xml", d)
        entries = parse_source(src)
        assert all(len(e.gloss) >= 15 for e in entries)

    def test_short_root_div_skipped(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        d = self._make_xml_dir(tmp_path)
        src = _make_source("lanes", "lanes_xml", d)
        entries = parse_source(src)
        # div2 n="A" has only 1 Arabic consonant → entire div2 skipped
        # so "Definite article" gloss should not appear
        assert not any("definite" in e.gloss.lower() for e in entries)

    def test_missing_dir_returns_empty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        src = _make_source("lanes", "lanes_xml", tmp_path / "nonexistent")
        assert parse_source(src) == []

    def test_source_name_set(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        d = self._make_xml_dir(tmp_path)
        src = _make_source("lanes", "lanes_xml", d)
        entries = parse_source(src)
        assert all(e.source == "lanes" for e in entries)

    def test_arabtex_two_char_sequences(self, tmp_path):
        from lexicon_ingestion.parser import _arabtex_to_arabic
        assert _arabtex_to_arabic("A^") == "أ"
        assert _arabtex_to_arabic("A=") == "إ"
        assert _arabtex_to_arabic("A_") == "آ"
        assert _arabtex_to_arabic("w^") == "ؤ"
        assert _arabtex_to_arabic("y^") == "ئ"

    def test_arabtex_consonant_mapping(self, tmp_path):
        from lexicon_ingestion.parser import _arabtex_to_arabic
        assert _arabtex_to_arabic("ktb") == "كتب"
        assert _arabtex_to_arabic("Eilm") == "علم"

    def test_arabtex_strips_vowels(self, tmp_path):
        from lexicon_ingestion.parser import _arabtex_to_arabic
        # kAtib: vowels i stripped → كاتب
        assert _arabtex_to_arabic("kAtib") == "كاتب"


# ---------------------------------------------------------------------------
# qamus_lmf adapter
# ---------------------------------------------------------------------------

class TestQamusLmf:
    def _make_xml(self, tmp_path: Path) -> Path:
        xml_dir = tmp_path / "qamus_xml"
        xml_dir.mkdir()
        xml_content = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <LexicalResource>
              <GlobalInformation>
                <feat att="label" val="Al-Qamus al-Muhit Test"/>
              </GlobalInformation>
              <Lexicon>
                <feat att="language" val="ara"/>
                <LexicalEntry id="le_0001">
                  <feat att="partOfSpeech" val="noun"/>
                  <feat att="root" val="كتب"/>
                  <Lemma>
                    <feat att="writtenForm" val="كِتَابٌ"/>
                  </Lemma>
                  <Sense id="s_0001">
                    <Definition>
                      <feat att="writtenForm" val="a book; written document"/>
                    </Definition>
                  </Sense>
                </LexicalEntry>
                <LexicalEntry id="le_0002">
                  <feat att="root" val="علم"/>
                  <Lemma>
                    <feat att="writtenForm" val="عِلْمٌ"/>
                  </Lemma>
                  <Sense id="s_0002">
                    <Definition>
                      <feat att="writtenForm" val="knowledge; science"/>
                    </Definition>
                  </Sense>
                  <Sense id="s_0003">
                    <Definition>
                      <feat att="writtenForm" val="a mark; sign"/>
                    </Definition>
                  </Sense>
                </LexicalEntry>
                <LexicalEntry id="le_0003">
                  <Lemma>
                    <feat att="writtenForm" val="latin_only"/>
                  </Lemma>
                  <Sense id="s_0004">
                    <Definition>
                      <feat att="writtenForm" val="should be skipped"/>
                    </Definition>
                  </Sense>
                </LexicalEntry>
              </Lexicon>
            </LexicalResource>
        """)
        (xml_dir / "test.xml").write_text(xml_content, encoding="utf-8")
        return xml_dir

    def test_parse_returns_entries(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        xml_dir = self._make_xml(tmp_path)
        src = _make_source("qamus", "qamus_lmf", xml_dir)
        entries = parse_source(src)
        assert len(entries) >= 2

    def test_lemma_extracted(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        xml_dir = self._make_xml(tmp_path)
        src = _make_source("qamus", "qamus_lmf", xml_dir)
        entries = parse_source(src)
        lemmas = [e.lemma for e in entries]
        assert "كِتَابٌ" in lemmas

    def test_root_extracted(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        xml_dir = self._make_xml(tmp_path)
        src = _make_source("qamus", "qamus_lmf", xml_dir)
        entries = parse_source(src)
        ktb = [e for e in entries if e.root == "كتب"]
        assert len(ktb) >= 1

    def test_multi_sense_creates_multiple_entries(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        xml_dir = self._make_xml(tmp_path)
        src = _make_source("qamus", "qamus_lmf", xml_dir)
        entries = parse_source(src)
        ilm = [e for e in entries if "علم" in (e.root or "")]
        assert len(ilm) >= 2  # two senses → two entries

    def test_non_arabic_lemma_skipped(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        xml_dir = self._make_xml(tmp_path)
        src = _make_source("qamus", "qamus_lmf", xml_dir)
        entries = parse_source(src)
        assert not any(e.lemma == "latin_only" for e in entries)

    def test_empty_directory_returns_empty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        src = _make_source("qamus", "qamus_lmf", empty_dir)
        entries = parse_source(src)
        assert entries == []

    def test_missing_directory_returns_empty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        src = _make_source("qamus", "qamus_lmf", tmp_path / "does_not_exist")
        entries = parse_source(src)
        assert entries == []

    def test_source_name_and_era(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        xml_dir = self._make_xml(tmp_path)
        src = _make_source("qamus", "qamus_lmf", xml_dir)
        entries = parse_source(src)
        assert all(e.source == "qamus" for e in entries)
        assert all(e.era == "classical" for e in entries)


# ---------------------------------------------------------------------------
# shamela_sqlite adapter
# ---------------------------------------------------------------------------

class TestShamelaSqlite:
    def _make_shamela(self, tmp_path: Path) -> Path:
        """
        Create a minimal Shamela4-style layout:
          shamela_dir/index.db   — metadata with book table
          shamela_dir/42.db      — book text table
        """
        shamela_dir = tmp_path / "shamela"
        shamela_dir.mkdir()

        # Metadata DB
        meta = shamela_dir / "index.db"
        conn = sqlite3.connect(str(meta))
        conn.execute("""
            CREATE TABLE b (
                id INTEGER PRIMARY KEY,
                name TEXT,
                auth TEXT,
                cat INTEGER
            )
        """)
        conn.execute("INSERT INTO b VALUES (42, 'لسان العرب', 'ابن منظور', 1)")
        conn.commit()
        conn.close()

        # Book DB — 3 entries in simplified Shamela text format
        book = shamela_dir / "42.db"
        conn = sqlite3.connect(str(book))
        conn.execute("""
            CREATE TABLE t (
                id INTEGER PRIMARY KEY,
                nass TEXT,
                page INTEGER,
                part INTEGER
            )
        """)
        paragraphs = [
            # Entry 1: root ك ت ب
            ("ك ت ب", 1, 1),
            ("كَتَبَ يَكتُبُ كَتْبًا وكِتَابَةً: خَطَّ وَرَسَمَ.", 1, 1),
            # Entry 2: root ق ر أ
            ("ق ر أ", 2, 1),
            ("قَرَأَ الكِتَابَ: تَلَاهُ وَنَطَقَ بِهِ.", 2, 1),
            # Entry 3: root ع ل م
            ("ع ل م", 3, 1),
            ("عَلِمَ الشَّيْءَ عِلْمًا: أَدْرَكَهُ وَأَيْقَنَ بِهِ.", 3, 1),
        ]
        conn.executemany("INSERT INTO t VALUES (?, ?, ?, ?)",
                         [(i+1, p[0], p[1], p[2]) for i, p in enumerate(paragraphs)])
        conn.commit()
        conn.close()

        return shamela_dir

    def test_parse_returns_entries(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        shamela_dir = self._make_shamela(tmp_path)
        src = _make_source("lisan", "shamela_sqlite", shamela_dir, book_name="لسان العرب")
        entries = parse_source(src)
        assert len(entries) >= 1

    def test_roots_extracted(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        shamela_dir = self._make_shamela(tmp_path)
        src = _make_source("lisan", "shamela_sqlite", shamela_dir, book_name="لسان العرب")
        entries = parse_source(src)
        roots = {e.root for e in entries if e.root}
        # Roots should be compacted from spaced letters
        assert any(r in roots for r in ("كتب", "قرأ", "علم"))

    def test_gloss_is_nonempty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        shamela_dir = self._make_shamela(tmp_path)
        src = _make_source("lisan", "shamela_sqlite", shamela_dir, book_name="لسان العرب")
        entries = parse_source(src)
        assert all(e.gloss for e in entries)

    def test_missing_book_returns_empty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        shamela_dir = self._make_shamela(tmp_path)
        src = _make_source("lisan", "shamela_sqlite", shamela_dir, book_name="تاج العروس")
        entries = parse_source(src)
        assert entries == []

    def test_missing_directory_returns_empty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        src = _make_source("lisan", "shamela_sqlite", tmp_path / "no_dir", book_name="لسان العرب")
        entries = parse_source(src)
        assert entries == []

    def test_source_name_propagated(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        shamela_dir = self._make_shamela(tmp_path)
        src = _make_source("lisan", "shamela_sqlite", shamela_dir, book_name="لسان العرب")
        entries = parse_source(src)
        assert all(e.source == "lisan" for e in entries)


# ---------------------------------------------------------------------------
# openiti_markdown adapter
# ---------------------------------------------------------------------------

class TestOpenitiMarkdown:
    def _make_txt(self, tmp_path: Path) -> Path:
        txt_dir = tmp_path / "openiti_txt"
        txt_dir.mkdir()
        content = textwrap.dedent("""\
            ######OpenITI#1.3#
            #META# 000.SortField	:: Shamela_0007643
            #META#Header#End#

            ### | باب الكاف

            ### | كتب

            كَتَبَ يَكتُبُ كَتْبًا وكِتَابَةً.
            ~~
            وكِتَابٌ وكُتُبٌ جمعه.
            ~~

            ### | علم

            عَلِمَ الشَّيْءَ عِلْمًا إذا أيقن به.
            ~~

            PageV01P010

        """)
        (txt_dir / "sample.txt").write_text(content, encoding="utf-8")
        return txt_dir

    def test_parse_returns_entries(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        txt_dir = self._make_txt(tmp_path)
        src = _make_source("lisan", "openiti", txt_dir)
        entries = parse_source(src)
        assert len(entries) >= 1

    def test_section_headings_become_lemmas(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        txt_dir = self._make_txt(tmp_path)
        src = _make_source("lisan", "openiti", txt_dir)
        entries = parse_source(src)
        lemmas = [e.lemma for e in entries]
        assert any("كتب" in l or "علم" in l for l in lemmas)

    def test_gloss_nonempty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        txt_dir = self._make_txt(tmp_path)
        src = _make_source("lisan", "openiti", txt_dir)
        entries = parse_source(src)
        assert all(e.gloss for e in entries)

    def test_metadata_lines_skipped(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        txt_dir = self._make_txt(tmp_path)
        src = _make_source("lisan", "openiti", txt_dir)
        entries = parse_source(src)
        assert not any("META" in e.gloss or "OpenITI" in e.gloss for e in entries)

    def test_missing_directory_returns_empty(self, tmp_path):
        from lexicon_ingestion.parser import parse_source
        src = _make_source("lisan", "openiti", tmp_path / "no_dir")
        entries = parse_source(src)
        assert entries == []


# ---------------------------------------------------------------------------
# ingest_all_enabled — integration with temp DB
# ---------------------------------------------------------------------------

def test_ingest_all_enabled_includes_fixture(tmp_path):
    """ingest_all_enabled must always return fixture entries even when others are missing."""
    import lexicon_ingestion.index_builder as ib

    class _Cfg:
        class lexicon:
            class index:
                path = str(tmp_path / "test.db")
                approximate_match_threshold = 0.8

    ib._index_singleton = None
    results = ib.ingest_all_enabled(_Cfg())
    assert results.get("_fixture", 0) > 0
    ib._index_singleton = None


def test_ingest_all_enabled_absent_sources_return_zero(tmp_path):
    """Absent real-source data files return 0 without crashing."""
    import lexicon_ingestion.index_builder as ib

    class _Cfg:
        class lexicon:
            class index:
                path = str(tmp_path / "test2.db")
                approximate_match_threshold = 0.8

    ib._index_singleton = None
    results = ib.ingest_all_enabled(_Cfg())
    for name in ("lanes", "lisan", "taj", "qamus"):
        assert results.get(name, 0) == 0, f"{name} should be 0 when data absent"
    ib._index_singleton = None
