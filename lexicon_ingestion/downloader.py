"""Two-direction lexicon DB manager.

Build direction  (run once in CI / manually):
  build_lexicons_db()  — fetch raw sources, parse, write lexicons.db
  upload_to_hf()       — push lexicons.db to a HuggingFace dataset repo

Runtime direction  (called at server/CLI startup):
  ensure_lexicons_db() — check local copy, download from HF if absent/stale,
                         call progress_cb(fraction) during download,
                         return local path or None on graceful fallback

All build-side functions require huggingface_hub and requests (both listed in
requirements.txt). Runtime functions tolerate their absence gracefully.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

from utils.logging import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_DB_PATH = "data/lexicons/lexicons.db"
_ETAG_SUFFIX = ".etag"
_MIN_ENTRIES_FOR_UPLOAD = 1_000   # abort upload if build produces fewer entries

# ── Helpers ───────────────────────────────────────────────────────────────────


def _db_path(config=None) -> str:
    if config is not None:
        try:
            return config.lexicon.db_path
        except AttributeError:
            pass
    return _DEFAULT_DB_PATH


def _hf_repo(config=None) -> str:
    if config is not None:
        try:
            return config.lexicon.hf_repo_id
        except AttributeError:
            pass
    return os.environ.get("LEXICON_HF_REPO", "")


def _count_entries(db_path: str) -> tuple[int, list[str]]:
    """Return (total_entry_count, list_of_source_names) from a lexicons.db."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            sources = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT source FROM entries ORDER BY source"
                ).fetchall()
            ]
        return total, sources
    except Exception as exc:
        log.warning(f"_count_entries failed for {db_path}: {exc}")
        return 0, []


# ── Build direction ───────────────────────────────────────────────────────────


def fetch_lanes_raw(dest_dir: str) -> str | None:
    """Download TEI XML files from laneslexicon/lexicon_xml to dest_dir.

    Uses GitHub's archive API to get the full repo ZIP (no auth required
    for public repos).  Returns dest_dir containing the extracted .xml files,
    or None on failure.
    """
    try:
        import requests
    except ImportError:
        log.error("fetch_lanes_raw: 'requests' not installed"); return None

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    xml_files = list(dest.glob("*.xml"))
    if xml_files:
        log.info(f"fetch_lanes_raw: {len(xml_files)} XML files already present at {dest}")
        return str(dest)

    # GitHub archive API — no auth required for public repos; follows redirect
    archive_url = "https://api.github.com/repos/laneslexicon/lexicon_xml/zipball/HEAD"
    zip_dest = dest / "lexicon_xml_archive.zip"
    try:
        with requests.get(
            archive_url, stream=True, timeout=120,
            headers={"Accept": "application/vnd.github+json"},
            allow_redirects=True,
        ) as r:
            r.raise_for_status()
            with open(zip_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
    except Exception as exc:
        log.error(f"fetch_lanes_raw: GitHub archive download failed: {exc}")
        return None

    try:
        with zipfile.ZipFile(zip_dest) as zf:
            xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
            if not xml_names:
                log.error("fetch_lanes_raw: no .xml files in archive")
                return None
            for name in xml_names:
                zf.extract(name, dest)
        # Flatten nested dirs — repo ZIP nests under laneslexicon-lexicon_xml-<sha>/
        for xf in list(dest.glob("**/*.xml")):
            if xf.parent != dest:
                xf.rename(dest / xf.name)
        for d in sorted(dest.glob("*/"), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        zip_dest.unlink(missing_ok=True)
        xml_files = list(dest.glob("*.xml"))
        log.info(f"fetch_lanes_raw: extracted {len(xml_files)} TEI XML files to {dest}")
        return str(dest) if xml_files else None
    except Exception as exc:
        log.error(f"fetch_lanes_raw: extraction failed: {exc}")
        return None


def fetch_qamus_raw(dest_dir: str) -> str | None:
    """Attempt to download al-Qāmūs al-Muḥīṭ LMF XML from CLARIN.

    NOTE: The CLARIN distribution URL returns HTTP 403 from most network
    environments (confirmed by probe).  If the probe below fails, download
    the LMF ZIP manually from:

        https://clarin.eurac.edu/repository/handle/20.500.12124/23

    and extract the .xml files into dest_dir.
    """
    try:
        import requests
    except ImportError:
        log.error("fetch_qamus_raw: 'requests' not installed"); return None

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if list(dest.glob("*.xml")):
        log.info("fetch_qamus_raw: XML files already present"); return str(dest)

    clarin_url = (
        "https://clarin.eurac.edu/repository/xmlui/bitstream/handle/"
        "20.500.12124/23/AlQamusAlMuhit-LMF.zip"
    )

    # Probe before committing to a full download — URL frequently 403s
    try:
        probe = requests.head(clarin_url, timeout=10, allow_redirects=True)
        if probe.status_code == 403:
            log.error(
                "fetch_qamus_raw: CLARIN URL returned 403 Forbidden. "
                "Download manually from "
                "https://clarin.eurac.edu/repository/handle/20.500.12124/23 "
                f"and extract XML files into {dest}"
            )
            return None
        probe.raise_for_status()
    except Exception as exc:
        log.error(f"fetch_qamus_raw: URL probe failed: {exc}"); return None

    zip_dest = dest / "qamus_lmf.zip"
    try:
        with requests.get(clarin_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
    except Exception as exc:
        log.error(f"fetch_qamus_raw: download failed: {exc}"); return None

    try:
        with zipfile.ZipFile(zip_dest) as zf:
            for name in zf.namelist():
                if name.endswith(".xml"):
                    zf.extract(name, dest)
        xml_files = list(dest.glob("**/*.xml"))
        for xf in xml_files:
            if xf.parent != dest:
                xf.rename(dest / xf.name)
        zip_dest.unlink(missing_ok=True)
        log.info(f"fetch_qamus_raw: extracted {len(xml_files)} XML files to {dest}")
        return str(dest) if xml_files else None
    except Exception as exc:
        log.error(f"fetch_qamus_raw: extraction failed: {exc}"); return None


def build_lexicons_db(
    output_path: str | None = None,
    sources: list[str] | None = None,
    config=None,
) -> int:
    """Fetch raw sources, parse them, and write a unified ``lexicons.db``.

    Build path only. Runtime uses ensure_lexicons_db().

    Parameters
    ----------
    output_path : path for the output SQLite (defaults to ``_DEFAULT_DB_PATH``)
    sources     : subset of source names to build; None means all enabled
    config      : optional config object

    Returns
    -------
    int : total number of entries written

    Raises
    ------
    RuntimeError if the resulting DB has fewer than ``_MIN_ENTRIES_FOR_UPLOAD``
    entries, indicating a silent parser failure.
    """
    import json
    from lexicon_ingestion.sources import enabled_sources
    from lexicon_ingestion.parser import parse_source
    from lexicon_ingestion.storage import _SCHEMA

    out = Path(output_path or _DEFAULT_DB_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build into a temp file first; atomic rename on success
    with tempfile.NamedTemporaryFile(
        suffix=".db", dir=out.parent, delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        # Initialise schema in the temp DB
        with sqlite3.connect(tmp_path) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

        total = 0
        active_sources = [
            s for s in enabled_sources()
            if sources is None or s.name in sources
        ]

        for source in active_sources:
            # Download raw data for sources that support it
            if source.parser_adapter == "lanes_xml":
                lanes_dir = str(Path(source.path))
                if not list(Path(lanes_dir).glob("*.xml")):
                    fetch_lanes_raw(lanes_dir)
            elif source.parser_adapter == "qamus_lmf":
                qamus_dir = Path(source.path)
                if not qamus_dir.exists() or not list(qamus_dir.glob("*.xml")):
                    fetch_qamus_raw(source.path)

            try:
                entries = parse_source(source)
            except Exception as exc:
                log.warning(f"build_lexicons_db: {source.name} parse failed: {exc}")
                entries = []

            if not entries:
                log.info(f"build_lexicons_db: {source.name} — 0 entries (data absent?)")
                continue

            rows = [
                (e.lemma, e.root, e.pattern, e.gloss, e.source, e.era,
                 e.domain, json.dumps(e.examples, ensure_ascii=False), e.priority)
                for e in entries
            ]
            with sqlite3.connect(tmp_path) as conn:
                conn.execute("DELETE FROM entries WHERE source = ?", (source.name,))
                conn.executemany(
                    "INSERT INTO entries "
                    "(lemma,root,pattern,gloss,source,era,domain,examples,priority)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
            total += len(entries)
            log.info(f"build_lexicons_db: {source.name} — {len(entries)} entries")

        # ── Minimum-entry guard ───────────────────────────────────────────────
        actual_count, actual_sources = _count_entries(tmp_path)
        if actual_count < _MIN_ENTRIES_FOR_UPLOAD:
            os.unlink(tmp_path)
            raise RuntimeError(
                f"build_lexicons_db aborted: DB contains only {actual_count} entries "
                f"(minimum {_MIN_ENTRIES_FOR_UPLOAD}). "
                "Check that raw source files are present and parsers are working."
            )

        out.unlink(missing_ok=True)
        os.replace(tmp_path, out)
        log.info(
            f"build_lexicons_db: wrote {actual_count} entries from "
            f"{actual_sources} to {out}"
        )
        return actual_count

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def upload_to_hf(db_path: str, repo_id: str, token: str | None = None) -> None:
    """Upload *db_path* to a HuggingFace dataset repository.

    Parameters
    ----------
    db_path  : local path to lexicons.db
    repo_id  : HF dataset repo ID, e.g. ``"username/ancient-ocr-lexicons"``
    token    : HF API token; falls back to ``HF_TOKEN`` env var
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError(
            "upload_to_hf requires 'huggingface_hub'. "
            "Install it: pip install huggingface_hub"
        )

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "No HuggingFace token provided. "
            "Pass token= or set the HF_TOKEN environment variable."
        )
    if not repo_id:
        raise ValueError(
            "No HuggingFace repo_id provided. "
            "Pass repo_id= or set config.lexicon.hf_repo_id."
        )

    count, sources = _count_entries(db_path)
    log.info(
        f"upload_to_hf: uploading {db_path} ({count} entries, "
        f"sources={sources}) → {repo_id}"
    )

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=db_path,
        path_in_repo="lexicons.db",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update lexicons.db: {count} entries from {sources}",
    )
    log.info(f"upload_to_hf: uploaded successfully to {repo_id}")


# ── Runtime direction ─────────────────────────────────────────────────────────


def _etag_path(db_path: str) -> str:
    return db_path + _ETAG_SUFFIX


def _read_local_etag(db_path: str) -> str | None:
    try:
        return Path(_etag_path(db_path)).read_text().strip()
    except OSError:
        return None


def _write_local_etag(db_path: str, etag: str) -> None:
    Path(_etag_path(db_path)).write_text(etag)


def _hf_remote_etag(repo_id: str, token: str | None = None) -> str | None:
    """Return the current ETag/SHA256 of ``lexicons.db`` in the HF repo, or None."""
    try:
        import requests
        hf_token = token or os.environ.get("HF_TOKEN")
        url = (
            f"https://huggingface.co/datasets/{repo_id}"
            f"/resolve/main/lexicons.db"
        )
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return resp.headers.get("ETag") or resp.headers.get("X-Linked-ETag")
    except Exception as exc:
        log.debug(f"_hf_remote_etag: {exc}")
    return None


def ensure_lexicons_db(
    config=None,
    progress_cb: Callable[[float], None] | None = None,
) -> str | None:
    """Ensure a local ``lexicons.db`` is present and up-to-date.

    Runtime path. Downloads from HuggingFace if absent or stale; falls back
    to the local copy when HF is unreachable. After this call load_entries()
    and build_index() read from the same file this function manages.

    Checks the local copy first; downloads from HuggingFace only if the file
    is absent or its ETag sidecar differs from the remote ETag.

    Parameters
    ----------
    config      : optional config object (for db_path and hf_repo_id)
    progress_cb : called with float in [0.0, 1.0] during download

    Returns
    -------
    str  : absolute path to the local lexicons.db on success
    None : if HF is unreachable and no local copy exists (soft failure —
           the pipeline continues with fixture data)
    """
    local_path = str(Path(_db_path(config)).resolve())
    repo_id = _hf_repo(config)

    # ── Check local copy ──────────────────────────────────────────────────────
    if Path(local_path).exists():
        if not repo_id:
            # No HF repo configured — use whatever is local
            return local_path

        local_etag = _read_local_etag(local_path)
        remote_etag = _hf_remote_etag(repo_id)

        if remote_etag is None:
            # HF unreachable but we have a local copy — use it
            log.warning(
                "ensure_lexicons_db: cannot reach HuggingFace to check for updates. "
                "Using existing local copy."
            )
            return local_path

        if local_etag and local_etag == remote_etag:
            log.debug(f"ensure_lexicons_db: local copy is current (etag={local_etag[:16]}…)")
            return local_path

        log.info("ensure_lexicons_db: remote copy is newer, downloading update")
    else:
        if not repo_id:
            log.warning(
                "ensure_lexicons_db: lexicons.db not found locally and no "
                "hf_repo_id configured. Using fixture data only."
            )
            return None
        log.info(f"ensure_lexicons_db: lexicons.db not found, downloading from {repo_id}")

    # ── Download from HuggingFace ─────────────────────────────────────────────
    try:
        import requests
    except ImportError:
        log.warning("ensure_lexicons_db: 'requests' not installed, cannot download.")
        return None if not Path(local_path).exists() else local_path

    hf_token = os.environ.get("HF_TOKEN")
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/lexicons.db"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total_size = int(resp.headers.get("Content-Length", 0))
            remote_etag = (
                resp.headers.get("ETag") or resp.headers.get("X-Linked-ETag") or ""
            )
            downloaded = 0

            with tempfile.NamedTemporaryFile(
                suffix=".db", dir=Path(local_path).parent, delete=False
            ) as tmp:
                tmp_path = tmp.name
                for chunk in resp.iter_content(chunk_size=65536):
                    tmp.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total_size:
                        progress_cb(min(downloaded / total_size, 1.0))

        os.replace(tmp_path, local_path)
        if remote_etag:
            _write_local_etag(local_path, remote_etag)

        # ── Log what was actually loaded ──────────────────────────────────────
        count, sources = _count_entries(local_path)
        log.info(
            f"ensure_lexicons_db: lexicons.db loaded — "
            f"{count:,} entries from {len(sources)} source(s): {sources}"
        )
        if progress_cb:
            progress_cb(1.0)

        # Rebuild the in-memory index so the new data is immediately available
        try:
            from lexicon_ingestion.index_builder import get_index
            get_index(config=config, force_rebuild=True)
        except Exception as exc:
            log.warning(f"ensure_lexicons_db: index rebuild failed: {exc}")

        return local_path

    except Exception as exc:
        log.warning(
            f"ensure_lexicons_db: download from HuggingFace failed ({exc}). "
            "Falling back to fixture lexicon data. "
            "Re-run after network is restored."
        )
        try:
            os.unlink(tmp_path)
        except (OSError, NameError):
            pass
        return None if not Path(local_path).exists() else local_path
