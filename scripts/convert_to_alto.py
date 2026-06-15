"""
Convert .png + .gt.txt training pairs to minimal ALTO XML for Kraken 7.x training.

Each ALTO file references the original PNG (absolute path) and wraps the full
image as a single TextLine — appropriate for pre-extracted line crops.

Output layout:
  data/training_alto/book_IbnFaqihHamadhani.Buldan/a_000000.xml
  data/training_alto/book_Jahiz.Hayawan/a_000000.xml
  ...

Usage:
    python scripts/convert_to_alto.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

REPO = Path(__file__).parent.parent.resolve()

SOURCES = [
    REPO / "data/book_IbnFaqihHamadhani.Buldan/7_final",
    REPO / "data/book_Jahiz.Hayawan/7_final",
]

OUT_ROOT = REPO / "data/training_alto"

ALTO_TMPL = """\
<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# \
http://www.loc.gov/standards/alto/v4/alto-4-2.xsd">
  <Description>
    <MeasurementUnit>pixel</MeasurementUnit>
    <sourceImageInformation>
      <fileName>{img_path}</fileName>
    </sourceImageInformation>
  </Description>
  <Layout>
    <Page WIDTH="{w}" HEIGHT="{h}" PHYSICAL_IMG_NR="1" ID="page1">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}">
        <TextBlock ID="block1" HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}">
          <TextLine ID="line1" HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}">
            <String HPOS="0" VPOS="0" WIDTH="{w}" HEIGHT="{h}" CONTENT="{content}"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""


def escape_xml(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def convert_source(src_dir: Path) -> tuple[int, int]:
    book_name = src_dir.parent.name   # e.g. book_IbnFaqihHamadhani.Buldan
    out_dir = OUT_ROOT / book_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = sorted(src_dir.glob("*.png"))
    converted = skipped = 0

    for png_path in pairs:
        gt_path = png_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            skipped += 1
            continue

        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            skipped += 1
            continue

        try:
            w, h = Image.open(png_path).size
        except Exception as exc:
            print(f"  WARN: cannot open {png_path.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        alto_xml = ALTO_TMPL.format(
            img_path=str(png_path),
            w=w,
            h=h,
            content=escape_xml(text),
        )

        out_path = out_dir / png_path.with_suffix(".xml").name
        out_path.write_text(alto_xml, encoding="utf-8")
        converted += 1

    return converted, skipped


def main() -> None:
    total_converted = total_skipped = 0
    for src in SOURCES:
        if not src.exists():
            print(f"SKIP (not found): {src}", file=sys.stderr)
            continue
        c, s = convert_source(src)
        print(f"{src.parent.name}: {c} converted, {s} skipped")
        total_converted += c
        total_skipped += s

    print(f"\nTotal: {total_converted} ALTO files → {OUT_ROOT}")
    if total_skipped:
        print(f"Skipped: {total_skipped} (missing .gt.txt or unreadable PNG)")

    # Print the ketos train command
    alto_files = sorted(OUT_ROOT.rglob("*.xml"))
    print(f"\n# ketos train command ({len(alto_files)} ALTO files):")
    print(f"ketos train \\")
    print(f"  --load models/kraken/muharaf_rec_best.mlmodel \\")
    print(f"  --output models/kraken/arabic_finetuned \\")
    print(f"  -f alto \\")
    print(f"  --resize add \\")
    print(f"  -t data/training_alto/book_IbnFaqihHamadhani.Buldan/*.xml \\")
    print(f"  -t data/training_alto/book_Jahiz.Hayawan/*.xml")
    print()
    print("# Or with a list file (avoids shell glob limits):")
    list_file = OUT_ROOT / "train_files.txt"
    list_file.write_text(
        "\n".join(str(p) for p in alto_files) + "\n",
        encoding="utf-8",
    )
    print(f"ketos train \\")
    print(f"  --load models/kraken/muharaf_rec_best.mlmodel \\")
    print(f"  --output models/kraken/arabic_finetuned \\")
    print(f"  -f alto \\")
    print(f"  --resize add \\")
    print(f"  -t {list_file}")


if __name__ == "__main__":
    main()
