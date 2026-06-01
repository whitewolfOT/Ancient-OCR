"""Build in-memory indexes from the lexicon store for fast runtime lookup."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from confidence_engine.state import LexiconEntry
from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LexiconIndex:
    by_lemma: dict[str, list[LexiconEntry]] = field(default_factory=lambda: defaultdict(list))
    by_root: dict[str, list[LexiconEntry]] = field(default_factory=lambda: defaultdict(list))
    by_root_prefix: dict[str, list[LexiconEntry]] = field(default_factory=lambda: defaultdict(list))
    by_normalized: dict[str, list[LexiconEntry]] = field(default_factory=lambda: defaultdict(list))
    all_lemmas: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.all_lemmas) == 0


_index_singleton: LexiconIndex | None = None


def build_index(config=None) -> LexiconIndex:
    """Load all entries from storage and build in-memory indexes."""
    from lexicon_ingestion.storage import load_entries
    from normalization.arabic_normalizer import normalize_text

    entries = load_entries(config=config)
    idx = LexiconIndex()

    for entry in entries:
        idx.by_lemma[entry.lemma].append(entry)
        if entry.root:
            idx.by_root[entry.root].append(entry)
            prefix = entry.root[:2] if len(entry.root) >= 2 else entry.root
            idx.by_root_prefix[prefix].append(entry)
        # normalized form for fuzzy lookup
        norm, _ = normalize_text(entry.lemma)
        idx.by_normalized[norm].append(entry)

    idx.all_lemmas = list(idx.by_lemma.keys())
    log.debug(
        f"build_index entries={len(entries)} lemmas={len(idx.all_lemmas)}"
        f" roots={len(idx.by_root)}"
    )
    return idx


def get_index(config=None, force_rebuild: bool = False) -> LexiconIndex:
    """Return the singleton index, building it if needed."""
    global _index_singleton
    if _index_singleton is None or force_rebuild:
        _index_singleton = build_index(config)
    return _index_singleton


def ingest_source(source_name: str, config=None) -> int:
    """Parse a source and save it to storage. Returns entry count."""
    from lexicon_ingestion.sources import get_source
    from lexicon_ingestion.parser import parse_source
    from lexicon_ingestion.storage import save_entries

    source = get_source(source_name)
    if source is None:
        raise ValueError(f"Unknown source: {source_name}")
    if not source.enabled:
        log.warning(f"source '{source_name}' is disabled; skipping ingestion")
        return 0

    entries = parse_source(source)
    saved = save_entries(entries, source_name, config)
    global _index_singleton
    _index_singleton = None  # invalidate cache
    return saved
