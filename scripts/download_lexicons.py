"""
Download classical Arabic lexicon data files.

Usage:
    python scripts/download_lexicons.py --lanes      # Lane's Lexicon SQLite
    python scripts/download_lexicons.py --qamus      # al-Qāmūs LMF XML
    python scripts/download_lexicons.py --all        # both of the above
    python scripts/download_lexicons.py --help       # show this message + Shamela instructions
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "lexicons"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, desc: str, retries: int = 3) -> bytes:
    """Download url → bytes with retry + progress dot."""
    print(f"  Downloading {desc} …", end="", flush=True)
    delay = 2
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ancient-ocr/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            print(f" {len(data) // 1024} KB")
            return data
        except Exception as exc:
            if attempt < retries - 1:
                print(f" retry({attempt+1})…", end="", flush=True)
                time.sleep(delay)
                delay *= 2
            else:
                print(f" FAILED: {exc}")
                raise


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Lane's Lexicon — laneslexicon/LexiconDatabase (GitHub releases)
# ---------------------------------------------------------------------------

LANES_GITHUB_API = "https://api.github.com/repos/laneslexicon/LexiconDatabase/releases/latest"
LANES_OUT = DATA / "lanes" / "lexicon.sqlite"


def download_lanes():
    print("\n=== Lane's Arabic-English Lexicon ===")
    LANES_OUT.parent.mkdir(parents=True, exist_ok=True)

    if LANES_OUT.exists():
        print(f"  Already present: {LANES_OUT}  (delete to re-download)")
        return True

    # Fetch release metadata
    try:
        meta_bytes = _fetch(LANES_GITHUB_API, "release metadata")
        release = json.loads(meta_bytes)
    except Exception as exc:
        print(f"  Cannot reach GitHub API: {exc}")
        _lanes_manual_instructions()
        return False

    assets = release.get("assets", [])
    zip_asset = next(
        (a for a in assets if a["name"].lower().endswith(".zip")),
        None,
    )
    if not zip_asset:
        print("  No ZIP asset found in latest release.")
        _lanes_manual_instructions()
        return False

    try:
        zip_bytes = _fetch(zip_asset["browser_download_url"], zip_asset["name"])
    except Exception:
        _lanes_manual_instructions()
        return False

    # Extract — may be a nested zip (zip containing .sqlite.zip)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as outer:
            names = outer.namelist()
            # Look for sqlite directly or inner zip
            sqlite_name = next((n for n in names if n.endswith(".sqlite")), None)
            inner_zip_name = next((n for n in names if n.endswith(".zip")), None)

            if sqlite_name:
                LANES_OUT.write_bytes(outer.read(sqlite_name))
            elif inner_zip_name:
                inner_bytes = outer.read(inner_zip_name)
                with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                    sqlite_name = next(
                        (n for n in inner.namelist() if n.endswith(".sqlite")), None
                    )
                    if sqlite_name:
                        LANES_OUT.write_bytes(inner.read(sqlite_name))
                    else:
                        raise FileNotFoundError("No .sqlite in inner zip")
            else:
                raise FileNotFoundError(f"No .sqlite found in release ZIP (contents: {names})")
    except Exception as exc:
        print(f"  ZIP extraction failed: {exc}")
        _lanes_manual_instructions()
        return False

    size_kb = LANES_OUT.stat().st_size // 1024
    print(f"  Saved → {LANES_OUT}  ({size_kb} KB)")
    return True


def _lanes_manual_instructions():
    print("""
  Manual download:
    1. Go to https://github.com/laneslexicon/LexiconDatabase/releases/latest
    2. Download the .zip release asset
    3. Extract until you have lexicon.sqlite
    4. Place it at:  data/lexicons/lanes/lexicon.sqlite
""")


# ---------------------------------------------------------------------------
# al-Qāmūs al-Muḥīṭ — ILC4CLARIN LMF XML (CC BY-SA 4.0)
# ---------------------------------------------------------------------------

# Known direct download URL for the Qamus LMF dataset (ILC CNR Pisa).
# The CLARIN DSpace endpoint returns a ZIP of XML files split by Arabic letter.
QAMUS_CLARIN_URL = (
    "https://dspace-clarin-it.ilc.cnr.it/repository/xmlui/bitstream/"
    "handle/20.500.11752/ILC-97/AlQamusAlMuhit.zip"
)
QAMUS_OUT = DATA / "qamus"


def download_qamus():
    print("\n=== al-Qāmūs al-Muḥīṭ (LMF XML, CC BY-SA 4.0) ===")
    QAMUS_OUT.mkdir(parents=True, exist_ok=True)

    existing = list(QAMUS_OUT.glob("*.xml"))
    if existing:
        print(f"  Already present: {len(existing)} XML file(s) in {QAMUS_OUT}  (delete to re-download)")
        return True

    try:
        zip_bytes = _fetch(QAMUS_CLARIN_URL, "AlQamusAlMuhit.zip")
    except Exception:
        _qamus_manual_instructions()
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
            if not xml_names:
                raise FileNotFoundError("No XML files found in ZIP")
            for name in xml_names:
                dest = QAMUS_OUT / Path(name).name
                dest.write_bytes(zf.read(name))
            print(f"  Extracted {len(xml_names)} XML file(s) → {QAMUS_OUT}")
    except Exception as exc:
        print(f"  ZIP extraction failed: {exc}")
        _qamus_manual_instructions()
        return False

    return True


def _qamus_manual_instructions():
    print("""
  Manual download:
    1. Go to https://dspace-clarin-it.ilc.cnr.it/repository/xmlui/handle/20.500.11752/ILC-97
    2. Download the dataset ZIP (LMF XML format, CC BY-SA 4.0)
    3. Extract the XML files into:  data/lexicons/qamus/
""")


# ---------------------------------------------------------------------------
# Shamela4 — Lisān al-ʿArab + Tāj al-ʿArūs (manual — 19 GB)
# ---------------------------------------------------------------------------

def show_shamela_instructions():
    print("""
=== Lisān al-ʿArab + Tāj al-ʿArūs (Shamela4 SQLite) ===

These two lexicons live inside the Shamela4 database (8,589 books, 19 GB).
Manual download required:

  Option A — Hugging Face (full DB):
    1. Install: pip install huggingface_hub
    2. Run:
         from huggingface_hub import snapshot_download
         snapshot_download(
             repo_id="AuthenticIlm/Shamela4_Full_DB",
             repo_type="dataset",
             local_dir="data/lexicons/shamela",
         )

  Option B — Direct download of individual books (smaller):
    The Shamela library is mirrored at https://shamela.ws/
    Find the book IDs for:
      - لسان العرب  (Ibn Manzur)
      - تاج العروس  (al-Zabidi)
    Download their individual .db files and place them at:
      data/lexicons/shamela/<book_id>.db
    Also place/create an index.db with the book metadata table 'b'.

  After placing data:
    python scripts/ingest_lexicons.py

""")


# ---------------------------------------------------------------------------
# Post-download ingestion
# ---------------------------------------------------------------------------

def ingest_source(name: str):
    sys.path.insert(0, str(ROOT))
    try:
        from lexicon_ingestion.index_builder import ingest_source as _ingest
        count = _ingest(name)
        print(f"  Ingested {count} entries from '{name}'")
    except Exception as exc:
        print(f"  Ingestion failed for '{name}': {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download classical Arabic lexicon data for Ancient-OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--lanes", action="store_true", help="Download Lane's Lexicon SQLite")
    parser.add_argument("--qamus", action="store_true", help="Download al-Qāmūs LMF XML")
    parser.add_argument("--all", dest="all_", action="store_true", help="Download both of the above")
    parser.add_argument("--no-ingest", action="store_true", help="Skip ingestion after download")
    args = parser.parse_args()

    if not any([args.lanes, args.qamus, args.all_]):
        parser.print_help()
        print()
        show_shamela_instructions()
        sys.exit(0)

    downloaded = []

    if args.lanes or args.all_:
        ok = download_lanes()
        if ok:
            downloaded.append("lanes")

    if args.qamus or args.all_:
        ok = download_qamus()
        if ok:
            downloaded.append("qamus")

    show_shamela_instructions()

    if downloaded and not args.no_ingest:
        print("\n=== Ingesting downloaded sources ===")
        for name in downloaded:
            ingest_source(name)
        print("\nDone. Run 'python scripts/ingest_lexicons.py' any time to re-ingest.")
    elif not downloaded:
        print("\nNothing downloaded. Check the errors above.")


if __name__ == "__main__":
    main()
