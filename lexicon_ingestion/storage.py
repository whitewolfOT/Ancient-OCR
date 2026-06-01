"""SQLite-backed durable store for lexicon entries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from confidence_engine.state import LexiconEntry
from utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma       TEXT NOT NULL,
    root        TEXT,
    pattern     TEXT,
    gloss       TEXT NOT NULL,
    source      TEXT NOT NULL,
    era         TEXT NOT NULL,
    domain      TEXT,
    examples    TEXT NOT NULL DEFAULT '[]',
    priority    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_lemma  ON entries(lemma);
CREATE INDEX IF NOT EXISTS idx_root   ON entries(root);
CREATE INDEX IF NOT EXISTS idx_source ON entries(source);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _db_path(config=None) -> str:
    if config is not None:
        try:
            return config.lexicon.index.path
        except AttributeError:
            pass
    return "data/lexicons/index.db"


def save_entries(entries: list[LexiconEntry], source_name: str, config=None) -> int:
    """Upsert entries for the given source. Returns count inserted."""
    db = _db_path(config)
    with _connect(db) as conn:
        conn.execute("DELETE FROM entries WHERE source = ?", (source_name,))
        rows = [
            (e.lemma, e.root, e.pattern, e.gloss, e.source, e.era,
             e.domain, json.dumps(e.examples, ensure_ascii=False), e.priority)
            for e in entries
        ]
        conn.executemany(
            "INSERT INTO entries (lemma,root,pattern,gloss,source,era,domain,examples,priority)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    log.debug(f"save_entries source={source_name} count={len(entries)}")
    return len(entries)


def load_entries(
    source: str | None = None,
    era: str | None = None,
    enabled_only: bool = True,
    config=None,
) -> list[LexiconEntry]:
    """Load entries from the store, optionally filtered by source or era."""
    db = _db_path(config)
    if not Path(db).exists():
        return []

    clauses: list[str] = []
    params: list = []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if era:
        clauses.append("era = ?")
        params.append(era)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM entries {where} ORDER BY priority DESC, id ASC"

    entries: list[LexiconEntry] = []
    try:
        with _connect(db) as conn:
            for row in conn.execute(sql, params):
                entries.append(LexiconEntry(
                    lemma=row["lemma"],
                    root=row["root"],
                    pattern=row["pattern"],
                    gloss=row["gloss"],
                    source=row["source"],
                    era=row["era"],
                    domain=row["domain"],
                    examples=json.loads(row["examples"]),
                    priority=row["priority"],
                ))
    except Exception as exc:
        log.warning(f"load_entries failed: {exc}")
    return entries


def clear(source: str | None = None, config=None) -> None:
    """Delete entries; if source given delete only that source."""
    db = _db_path(config)
    if not Path(db).exists():
        return
    with _connect(db) as conn:
        if source:
            conn.execute("DELETE FROM entries WHERE source = ?", (source,))
        else:
            conn.execute("DELETE FROM entries")
        conn.commit()
    log.debug(f"clear source={source or 'ALL'}")
