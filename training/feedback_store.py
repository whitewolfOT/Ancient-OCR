"""SQLite store for user-submitted ground-truth corrections."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# CER / WER helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein distance on arbitrary element lists."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[m]


def _cer(ref: str, hyp: str) -> float:
    """Character Error Rate = char edit distance / max(1, len(ref))."""
    if not ref:
        return 0.0 if not hyp else 1.0
    return min(1.0, _edit_distance(list(ref), list(hyp)) / len(ref))


def _wer(ref: str, hyp: str) -> float:
    """Word Error Rate = word edit distance / max(1, len(ref.split()))."""
    ref_w = ref.split()
    hyp_w = hyp.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return min(1.0, _edit_distance(ref_w, hyp_w) / len(ref_w))


def cer_wer(config=None) -> dict:
    """Return mean CER, WER and sample size over all feedback entries."""
    entries = load(config=config)
    if not entries:
        return {"cer": 0.0, "wer": 0.0, "sample_size": 0}
    cer_vals = [_cer(e.ground_truth, e.predicted) for e in entries]
    wer_vals = [_wer(e.ground_truth, e.predicted) for e in entries]
    return {
        "cer": round(sum(cer_vals) / len(cer_vals), 4),
        "wer": round(sum(wer_vals) / len(wer_vals), 4),
        "sample_size": len(entries),
    }

from confidence_engine.state import FeedbackEntry
from utils.logging import get_logger

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id           TEXT PRIMARY KEY,
    image_path   TEXT NOT NULL,
    bbox         TEXT NOT NULL,
    page_index   INTEGER NOT NULL,
    predicted    TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    source_file  TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    applied      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_applied ON feedback(applied);
CREATE INDEX IF NOT EXISTS idx_source  ON feedback(source_file);
"""


def _db_path(config=None) -> str:
    if config is not None:
        try:
            return config.training.feedback_db
        except AttributeError:
            pass
    return "data/feedback/feedback.db"


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def submit(entry: FeedbackEntry, config=None) -> str:
    """Store a feedback entry. Returns the assigned UUID."""
    if not entry.id:
        entry = entry.model_copy(update={"id": str(uuid.uuid4())})
    if not entry.submitted_at:
        entry = entry.model_copy(
            update={"submitted_at": datetime.now(timezone.utc).isoformat()}
        )

    db = _db_path(config)
    with _connect(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO feedback "
            "(id,image_path,bbox,page_index,predicted,ground_truth,source_file,submitted_at,applied)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                entry.id,
                entry.image_path,
                json.dumps(list(entry.bbox)),
                entry.page_index,
                entry.predicted,
                entry.ground_truth,
                entry.source_file,
                entry.submitted_at,
                int(entry.applied),
            ),
        )
        conn.commit()

    log.debug(f"submit feedback id={entry.id} predicted={entry.predicted!r} gt={entry.ground_truth!r}")
    return entry.id


def load(
    applied_only: bool = False,
    pending_only: bool = False,
    limit: int | None = None,
    config=None,
) -> list[FeedbackEntry]:
    """Load feedback entries from the store."""
    db = _db_path(config)
    if not Path(db).exists():
        return []

    clauses = []
    if applied_only:
        clauses.append("applied = 1")
    elif pending_only:
        clauses.append("applied = 0")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM feedback {where} ORDER BY submitted_at ASC"
    if limit:
        sql += f" LIMIT {int(limit)}"

    entries: list[FeedbackEntry] = []
    try:
        with _connect(db) as conn:
            for row in conn.execute(sql):
                entries.append(FeedbackEntry(
                    id=row["id"],
                    image_path=row["image_path"],
                    bbox=tuple(json.loads(row["bbox"])),
                    page_index=row["page_index"],
                    predicted=row["predicted"],
                    ground_truth=row["ground_truth"],
                    source_file=row["source_file"],
                    submitted_at=row["submitted_at"],
                    applied=bool(row["applied"]),
                ))
    except Exception as exc:
        log.warning(f"load feedback failed: {exc}")
    return entries


def mark_applied(ids: list[str], config=None) -> None:
    """Mark entries as applied after calibration."""
    if not ids:
        return
    db = _db_path(config)
    placeholders = ",".join("?" * len(ids))
    with _connect(db) as conn:
        conn.execute(
            f"UPDATE feedback SET applied=1 WHERE id IN ({placeholders})", ids
        )
        conn.commit()
    log.debug(f"mark_applied count={len(ids)}")


def stats(config=None) -> dict:
    """Return summary statistics about the feedback store, including CER/WER."""
    db = _db_path(config)
    _empty = {
        "total": 0, "applied": 0, "pending": 0,
        "by_source_file": {}, "error_rate": 0.0, "cer": 0.0, "wer": 0.0,
    }
    if not Path(db).exists():
        return _empty

    try:
        with _connect(db) as conn:
            total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
            applied = conn.execute("SELECT COUNT(*) FROM feedback WHERE applied=1").fetchone()[0]
            pending = total - applied

            rows = conn.execute(
                "SELECT source_file, COUNT(*) as cnt FROM feedback GROUP BY source_file"
            ).fetchall()
            by_source = {r["source_file"]: r["cnt"] for r in rows}

            wrong = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE predicted != ground_truth"
            ).fetchone()[0]
            error_rate = round(wrong / total, 4) if total else 0.0

            pairs = conn.execute(
                "SELECT predicted, ground_truth FROM feedback"
            ).fetchall()

        cer_vals = [_cer(r["ground_truth"], r["predicted"]) for r in pairs]
        wer_vals = [_wer(r["ground_truth"], r["predicted"]) for r in pairs]
        cer_avg = round(sum(cer_vals) / len(cer_vals), 4) if cer_vals else 0.0
        wer_avg = round(sum(wer_vals) / len(wer_vals), 4) if wer_vals else 0.0

    except Exception as exc:
        log.warning(f"stats failed: {exc}")
        return _empty

    return {
        "total": total,
        "applied": applied,
        "pending": pending,
        "by_source_file": by_source,
        "error_rate": error_rate,
        "cer": cer_avg,
        "wer": wer_avg,
    }
