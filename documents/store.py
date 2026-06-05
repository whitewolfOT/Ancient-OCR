"""SQLite store for document/page/cluster calibration state.

DB_PATH and DOCS_DIR are module-level so tests can monkeypatch them without
touching the real filesystem.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Module-level paths — monkeypatched in tests
DB_PATH: Path = Path("data/documents/documents.db")
DOCS_DIR: Path = Path("data/documents")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT,
    page_count  INT,
    upload_time TEXT,
    status      TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    page_id     TEXT PRIMARY KEY,
    doc_id      TEXT,
    page_num    INT,
    image_path  TEXT,
    cluster_id  TEXT,
    phash       TEXT,
    status      TEXT
);
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id            TEXT PRIMARY KEY,
    doc_id                TEXT,
    label                 TEXT,
    representative_page_id TEXT
);
CREATE TABLE IF NOT EXISTS page_settings (
    page_id           TEXT PRIMARY KEY,
    clahe             REAL,
    denoise           INT,
    deskew_threshold  REAL,
    binarization      TEXT,
    applied_at        TEXT
);
CREATE TABLE IF NOT EXISTS ground_truth (
    page_id      TEXT PRIMARY KEY,
    text         TEXT,
    submitted_at TEXT
);
CREATE TABLE IF NOT EXISTS ocr_results (
    page_id      TEXT PRIMARY KEY,
    result_json  TEXT,
    processed_at TEXT
);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def init_db() -> None:
    with _connect() as con:
        con.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def insert_document(
    doc_id: str,
    filename: str,
    page_count: int,
    upload_time: str,
    status: str = "ready",
) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO documents (doc_id, filename, page_count, upload_time, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, page_count, upload_time, status),
        )


def get_document(doc_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT doc_id, filename, page_count, upload_time, status "
            "FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(zip(["doc_id", "filename", "page_count", "upload_time", "status"], row))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def insert_page(
    page_id: str,
    doc_id: str,
    page_num: int,
    image_path: str,
    cluster_id: str,
    phash: str,
    status: str = "pending",
) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO pages "
            "(page_id, doc_id, page_num, image_path, cluster_id, phash, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (page_id, doc_id, page_num, image_path, cluster_id, phash, status),
        )


def get_page(page_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT page_id, doc_id, page_num, image_path, cluster_id, phash, status "
            "FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
    if row is None:
        return None
    cols = ["page_id", "doc_id", "page_num", "image_path", "cluster_id", "phash", "status"]
    return dict(zip(cols, row))


def get_pages_for_document(doc_id: str) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT page_id, doc_id, page_num, image_path, cluster_id, phash, status "
            "FROM pages WHERE doc_id = ? ORDER BY page_num",
            (doc_id,),
        ).fetchall()
    cols = ["page_id", "doc_id", "page_num", "image_path", "cluster_id", "phash", "status"]
    return [dict(zip(cols, r)) for r in rows]


def get_cluster_pages(cluster_id: str) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT page_id, doc_id, page_num, image_path, cluster_id, phash, status "
            "FROM pages WHERE cluster_id = ? ORDER BY page_num",
            (cluster_id,),
        ).fetchall()
    cols = ["page_id", "doc_id", "page_num", "image_path", "cluster_id", "phash", "status"]
    return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

def insert_cluster(
    cluster_id: str,
    doc_id: str,
    label: str,
    representative_page_id: str,
) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO clusters (cluster_id, doc_id, label, representative_page_id) "
            "VALUES (?, ?, ?, ?)",
            (cluster_id, doc_id, label, representative_page_id),
        )


def get_cluster(cluster_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT cluster_id, doc_id, label, representative_page_id "
            "FROM clusters WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(zip(["cluster_id", "doc_id", "label", "representative_page_id"], row))


def get_clusters_for_document(doc_id: str) -> list[dict]:
    with _connect() as con:
        rows = con.execute(
            "SELECT cluster_id, doc_id, label, representative_page_id "
            "FROM clusters WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
    cols = ["cluster_id", "doc_id", "label", "representative_page_id"]
    return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Page settings
# ---------------------------------------------------------------------------

def upsert_page_settings(
    page_id: str,
    clahe: float,
    denoise: int,
    deskew_threshold: float,
    binarization: str,
    applied_at: str,
) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO page_settings "
            "(page_id, clahe, denoise, deskew_threshold, binarization, applied_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(page_id) DO UPDATE SET "
            "clahe=excluded.clahe, denoise=excluded.denoise, "
            "deskew_threshold=excluded.deskew_threshold, "
            "binarization=excluded.binarization, applied_at=excluded.applied_at",
            (page_id, clahe, denoise, deskew_threshold, binarization, applied_at),
        )


def get_page_settings(page_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT page_id, clahe, denoise, deskew_threshold, binarization, applied_at "
            "FROM page_settings WHERE page_id = ?",
            (page_id,),
        ).fetchone()
    if row is None:
        return None
    cols = ["page_id", "clahe", "denoise", "deskew_threshold", "binarization", "applied_at"]
    return dict(zip(cols, row))


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def upsert_ground_truth(page_id: str, text: str, submitted_at: str) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO ground_truth (page_id, text, submitted_at) VALUES (?, ?, ?) "
            "ON CONFLICT(page_id) DO UPDATE SET text=excluded.text, submitted_at=excluded.submitted_at",
            (page_id, text, submitted_at),
        )


def get_ground_truth(page_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT page_id, text, submitted_at FROM ground_truth WHERE page_id = ?",
            (page_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(zip(["page_id", "text", "submitted_at"], row))


# ---------------------------------------------------------------------------
# OCR results
# ---------------------------------------------------------------------------

def upsert_ocr_result(page_id: str, result_json: str, processed_at: str) -> None:
    with _connect() as con:
        con.execute(
            "INSERT INTO ocr_results (page_id, result_json, processed_at) VALUES (?, ?, ?) "
            "ON CONFLICT(page_id) DO UPDATE SET "
            "result_json=excluded.result_json, processed_at=excluded.processed_at",
            (page_id, result_json, processed_at),
        )


def get_ocr_result(page_id: str) -> dict | None:
    with _connect() as con:
        row = con.execute(
            "SELECT page_id, result_json, processed_at FROM ocr_results WHERE page_id = ?",
            (page_id,),
        ).fetchone()
    if row is None:
        return None
    return dict(zip(["page_id", "result_json", "processed_at"], row))


def update_page_status(page_id: str, status: str) -> None:
    with _connect() as con:
        con.execute(
            "UPDATE pages SET status = ? WHERE page_id = ?",
            (status, page_id),
        )
