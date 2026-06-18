"""
Extract pages from a manuscript PDF as numbered JPG images.

Usage:
  python scripts/extract_pdf_pages.py manuscript.pdf
  python scripts/extract_pdf_pages.py manuscript.pdf --output data/test_images/
  python scripts/extract_pdf_pages.py manuscript.pdf --start 1 --end 100 --dpi 300
  python scripts/extract_pdf_pages.py manuscript.pdf --prefix ms_buldan --dpi 400

Output filenames: {prefix}_{page_number:04d}.jpg
Default prefix: "page"
Default output: data/test_images/
Default DPI: 300 (sufficient for HTR; 400 for high-quality originals)
Default range: all pages
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


def extract_pdf_pages(
    pdf_path: Path,
    output_dir: Path,
    prefix: str = "page",
    dpi: int = 300,
    start: int | None = None,
    end: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Extract pages [start, end] (1-indexed, inclusive) from pdf_path as JPGs.

    Returns the list of output file paths actually written (or already present
    when overwrite=False). Never raises on a single bad page — logs and skips.
    """
    import fitz  # PyMuPDF

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    doc = fitz.open(str(pdf_path))
    try:
        total = doc.page_count
        first = max(1, start) if start else 1
        last = min(total, end) if end else total

        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for page_number in range(first, last + 1):
            out_path = output_dir / f"{prefix}_{page_number:04d}.jpg"
            print(f"Page {page_number}/{last} → {out_path}")

            if out_path.exists() and not overwrite:
                written.append(out_path)
                continue

            try:
                page = doc.load_page(page_number - 1)
                pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
                pix.save(str(out_path), jpg_quality=95)
                written.append(out_path)
            except Exception as exc:
                log.warning(f"extract_pdf_pages: failed on page {page_number}: {exc}")
                continue
    finally:
        doc.close()

    total_size = sum(p.stat().st_size for p in written if p.exists())
    print(
        f"Done: {len(written)} pages extracted → {output_dir} "
        f"({total_size / 1024:.1f} KB total)"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pages from a manuscript PDF as numbered JPG images."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "--output", type=Path, default=Path("data/test_images/"),
        help="Output directory (default: data/test_images/)",
    )
    parser.add_argument("--prefix", type=str, default="page", help="Output filename prefix")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--start", type=int, default=None, help="First page (1-indexed)")
    parser.add_argument("--end", type=int, default=None, help="Last page (1-indexed, inclusive)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract pages even if the output file already exists",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    extract_pdf_pages(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        prefix=args.prefix,
        dpi=args.dpi,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
