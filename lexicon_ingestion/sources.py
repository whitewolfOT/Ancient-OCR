"""Declarative registry of lexicon sources."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceConfig:
    name: str
    enabled: bool
    priority: int
    era: str                    # "classical" | "modern"
    license: str
    path: str | None
    parser_adapter: str
    domain: str | None = None
    book_name: str | None = None   # for Shamela: Arabic book name to locate in the DB


SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="_fixture",
        enabled=True,
        priority=1,              # synthetic test data — always lowest real-source rank
        era="classical",
        license="synthetic",
        path="data/lexicons/_fixture/fixture.jsonl",
        parser_adapter="fixture",
    ),
    SourceConfig(
        name="quranic_corpus",
        enabled=True,
        priority=10,             # research-grade peer-reviewed Quranic morphology
        era="classical",
        license="research_open",
        path="data/lexicons/quranic_corpus/quran-morphology.txt",
        parser_adapter="quranic_corpus_tsv",
        domain="quranic",
        book_name="Quranic Arabic Corpus (mustafa0x/quran-morphology)",
    ),
    SourceConfig(
        name="lanes",
        enabled=True,
        priority=9,
        era="classical",
        license="public_domain",
        path="data/lexicons/lanes/",
        parser_adapter="lanes_xml",
    ),
    SourceConfig(
        name="arabic_wordnet",
        enabled=False,       # data needs manual download from globalwordnet.org (403 in container)
        priority=8,
        era="classical",
        license="cc_by_4",
        path="data/lexicons/arabic_wordnet/wn.xml",
        parser_adapter="arabic_wordnet_lmf",
        domain=None,
        book_name="Arabic WordNet (Global WordNet Association)",
    ),
    SourceConfig(
        name="lisan",
        enabled=False,       # requires Shamela4 bulk dump — enable only after build step
        priority=8,
        era="classical",
        license="openiti",
        path="data/lexicons/shamela/",
        parser_adapter="shamela_sqlite",
        book_name="لسان العرب",
    ),
    SourceConfig(
        name="taj",
        enabled=False,       # requires Shamela4 bulk dump — enable only after build step
        priority=7,
        era="classical",
        license="openiti",
        path="data/lexicons/shamela/",
        parser_adapter="shamela_sqlite",
        book_name="تاج العروس",
    ),
    SourceConfig(
        name="qamus",
        enabled=True,
        priority=6,
        era="classical",
        license="cc_by_sa_4",
        path="data/lexicons/qamus/",
        parser_adapter="qamus_lmf",
    ),
    SourceConfig(
        name="khorsi_roots",
        enabled=True,
        priority=5,              # supplementary root signal; unvalidated 2013 extraction
        era="classical",
        license="cc_by_sa_3",
        path="data/lexicons/khorsi/KhorsiCorpus.sql",
        parser_adapter="khorsi_sql",
        domain=None,
        book_name="Arabic Roots and Derivatives (from Taj al-Arus)",
    ),
    SourceConfig(
        name="wordnet",
        enabled=False,
        priority=3,
        era="modern",
        license="research",
        path="data/lexicons/wordnet/",
        parser_adapter="wordnet",
    ),
    SourceConfig(
        name="almaany",
        enabled=False,       # ToS prohibits scraping — must never be enabled
        priority=0,
        era="modern",
        license="tos_prohibited",
        path=None,
        parser_adapter="almaany_disabled",
    ),
]


def get_source(name: str) -> SourceConfig | None:
    for s in SOURCES:
        if s.name == name:
            return s
    return None


def enabled_sources() -> list[SourceConfig]:
    return [s for s in SOURCES if s.enabled]
