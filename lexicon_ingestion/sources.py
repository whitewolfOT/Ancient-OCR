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


SOURCES: list[SourceConfig] = [
    SourceConfig(
        name="_fixture",
        enabled=True,
        priority=10,
        era="classical",
        license="synthetic",
        path="data/lexicons/_fixture/fixture.jsonl",
        parser_adapter="fixture",
    ),
    SourceConfig(
        name="lanes",
        enabled=False,
        priority=9,
        era="classical",
        license="public_domain",
        path="data/lexicons/lanes/",
        parser_adapter="lanes",
    ),
    SourceConfig(
        name="lisan",
        enabled=False,
        priority=8,
        era="classical",
        license="openiti",
        path="data/lexicons/lisan/",
        parser_adapter="openiti",
    ),
    SourceConfig(
        name="taj",
        enabled=False,
        priority=7,
        era="classical",
        license="openiti",
        path="data/lexicons/taj/",
        parser_adapter="openiti",
    ),
    SourceConfig(
        name="qamus",
        enabled=False,
        priority=6,
        era="classical",
        license="openiti",
        path="data/lexicons/qamus/",
        parser_adapter="openiti",
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
