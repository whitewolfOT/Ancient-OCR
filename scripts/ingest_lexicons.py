"""
Re-ingest all enabled lexicon sources into the SQLite index.

Run after downloading new lexicon data:
    python scripts/ingest_lexicons.py

Or for a single source:
    python scripts/ingest_lexicons.py --source lanes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", "-s",
        metavar="NAME",
        help="Ingest only this source (default: all enabled)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be ingested without writing anything",
    )
    args = parser.parse_args()

    from lexicon_ingestion.sources import enabled_sources, get_source

    if args.dry_run:
        sources = (
            [get_source(args.source)] if args.source else enabled_sources()
        )
        print("Would ingest:")
        for s in sources:
            if s:
                data_exists = Path(s.path).exists() if s.path else False
                status = "data present" if data_exists else "data MISSING"
                print(f"  {s.name:12s}  adapter={s.parser_adapter:15s}  {status}")
        return

    if args.source:
        from lexicon_ingestion.index_builder import ingest_source
        try:
            count = ingest_source(args.source)
            print(f"{args.source}: {count} entries ingested")
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        from lexicon_ingestion.index_builder import ingest_all_enabled
        results = ingest_all_enabled()
        total = sum(results.values())
        print("\nIngestion results:")
        for name, count in results.items():
            status = f"{count} entries" if count > 0 else "0  (data file absent or not downloaded)"
            print(f"  {name:12s}  {status}")
        print(f"\nTotal: {total} entries in index")

        if total == 0:
            print(
                "\nNo entries loaded. Run 'python scripts/download_lexicons.py --help' "
                "for download instructions.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
