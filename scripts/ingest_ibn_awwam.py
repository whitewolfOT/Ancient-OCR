"""
Ingest Ibn al-ʿAwwām al-Filāḥa from OpenITI mARkdown.

Usage:
    python scripts/ingest_ibn_awwam.py [--force]

Downloads the OpenITI text to data/lexicons/ibn_awwam/ then ingests into
lexicons.db. Safe to re-run; existing entries are replaced (upsert).

OpenITI source:
  0637IbnAwwamIshbili.Filaha — Kitāb al-Filāḥa, c. 12th century Andalusia.
  Public domain text hosted at github.com/OpenITI.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Repo root on sys.path so imports work when run from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEST_DIR = Path("data/lexicons/ibn_awwam")
OPENITI_URL = (
    "https://raw.githubusercontent.com/OpenITI/0700AH/master/data/"
    "0637IbnAwwamIshbili/0637IbnAwwamIshbili.Filaha/"
    "0637IbnAwwamIshbili.Filaha.Shamela0010924-ara1.mARkdown"
)
DEST_FILE = DEST_DIR / "filaha.txt"


def download(force: bool = False) -> bool:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    if DEST_FILE.exists() and not force:
        print(f"[ibn_awwam] already present at {DEST_FILE} (use --force to re-download)")
        return True
    print(f"[ibn_awwam] downloading from OpenITI …")
    try:
        urllib.request.urlretrieve(OPENITI_URL, DEST_FILE)
        print(f"[ibn_awwam] saved {DEST_FILE.stat().st_size:,} bytes → {DEST_FILE}")
        return True
    except Exception as exc:
        print(f"[ibn_awwam] download failed: {exc}")
        print(
            "[ibn_awwam] manual fallback: download the file from\n"
            f"  {OPENITI_URL}\n"
            f"and save it as {DEST_FILE}"
        )
        return False


def ingest() -> int:
    from lexicon_ingestion.sources import get_source
    from lexicon_ingestion.parser import parse_openiti_markdown
    from lexicon_ingestion.index_builder import ingest_source

    source = get_source("ibn_awwam_filaha")
    if source is None:
        print("[ibn_awwam] ERROR: source 'ibn_awwam_filaha' not found in sources.py")
        return 0

    entries = parse_openiti_markdown(source)
    if not entries:
        print("[ibn_awwam] no entries parsed — check that the file exists and is valid OpenITI mARkdown")
        return 0

    count = ingest_source(source)
    print(f"[ibn_awwam] ingested {count} entries into lexicons.db")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Ibn al-ʿAwwām Filāḥa lexicon")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--no-download", action="store_true", help="Skip download, ingest only")
    args = parser.parse_args()

    if not args.no_download:
        ok = download(force=args.force)
        if not ok:
            sys.exit(1)

    count = ingest()
    if count == 0:
        print("[ibn_awwam] WARNING: 0 entries ingested — data file may be absent or empty")
    else:
        print(f"[ibn_awwam] done — {count} entries available in lexicon index")


if __name__ == "__main__":
    main()
