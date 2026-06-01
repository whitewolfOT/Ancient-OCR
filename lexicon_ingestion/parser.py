"""Per-source parser adapters → canonical LexiconEntry."""

from __future__ import annotations

import json
from pathlib import Path

from confidence_engine.state import LexiconEntry
from lexicon_ingestion.sources import SourceConfig
from utils.logging import get_logger

log = get_logger(__name__)


class DisabledSourceError(Exception):
    pass


def parse_source(source: SourceConfig) -> list[LexiconEntry]:
    """Dispatch to the correct per-source adapter."""
    adapter = source.parser_adapter
    if adapter == "fixture":
        return _parse_fixture(source)
    if adapter == "almaany_disabled":
        return _parse_almaany_disabled(source)
    if adapter == "openiti":
        return _parse_openiti(source)
    if adapter in ("lanes", "wordnet"):
        log.warning(f"parser adapter '{adapter}' not yet implemented; returning empty")
        return []
    log.warning(f"unknown parser adapter '{adapter}'; skipping source '{source.name}'")
    return []


# ---------------------------------------------------------------------------
# Fixture adapter
# ---------------------------------------------------------------------------

def _parse_fixture(source: SourceConfig) -> list[LexiconEntry]:
    path = Path(source.path)
    if not path.exists():
        log.warning(f"fixture path not found: {path}")
        return []
    entries: list[LexiconEntry] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
                entries.append(LexiconEntry(**raw))
            except Exception as exc:
                log.warning(f"fixture parse error line={lineno}: {exc}")
    log.debug(f"fixture loaded entries={len(entries)}")
    return entries


# ---------------------------------------------------------------------------
# OpenITI / Shamela shared adapter
# ---------------------------------------------------------------------------

def _parse_openiti(source: SourceConfig) -> list[LexiconEntry]:
    """Placeholder for OpenITI corpus adapter. Returns empty until data is present."""
    path = Path(source.path) if source.path else None
    if path is None or not path.exists():
        log.info(f"openiti source '{source.name}' path not found; skipping")
        return []
    # Real implementation: walk path for .json/.txt files and parse
    log.warning(f"openiti adapter for '{source.name}' not yet implemented")
    return []


# ---------------------------------------------------------------------------
# Disabled stub
# ---------------------------------------------------------------------------

def _parse_almaany_disabled(_source: SourceConfig) -> list[LexiconEntry]:
    raise DisabledSourceError(
        "Almaany scraping is prohibited by ToS. "
        "This source must never be enabled."
    )
