"""Generate line crops for all 9 pages for the line correction interface.

Strategy (in priority order):
1. Load existing segmentation from data/debug/{page}_segmentation.json
2. Run Kraken blla.segment() with default model
3. Fall back to token-based line clustering from results.json

Saves:
  data/lines/{page_id}/line_{N:03d}.png   — cropped line image
  data/lines/{page_id}/lines.json         — metadata + OCR text per line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

PAGES = [f"{i}.jpg" for i in range(1, 10)]
IMAGES_DIR = Path("data/test_images")
DEBUG_DIR = Path("data/debug")
LINES_DIR = Path("data/lines")
OCR_RESULTS = Path("data/ocr_results/results.json")
PADDING = 4  # extra px around each line crop


def _load_seg_from_cache(page_id: str) -> list[dict] | None:
    """Load pre-saved segmentation JSON (data/debug/{stem}_segmentation.json)."""
    stem = Path(page_id).stem
    seg_path = DEBUG_DIR / f"{stem}_segmentation.json"
    if not seg_path.exists():
        return None
    try:
        data = json.loads(seg_path.read_text())
        return data.get("lines", [])
    except Exception as exc:
        print(f"  [warn] could not load {seg_path}: {exc}")
        return None


def _seg_via_kraken(page_id: str, img: np.ndarray) -> list[dict] | None:
    """Run blla.segment() to get line boundaries."""
    try:
        from kraken import blla
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        seg = blla.segment(pil_img, text_direction="horizontal-rl")
        lines = []
        for i, line in enumerate(seg.lines):
            boundary = [[int(p[0]), int(p[1])] for p in (getattr(line, "boundary", None) or [])]
            baseline = [[int(p[0]), int(p[1])] for p in (getattr(line, "baseline", None) or [])]
            if not boundary:
                continue
            bxs = [p[0] for p in boundary]
            bys = [p[1] for p in boundary]
            lines.append({
                "detection_order": i,
                "line_id": str(id(line)),
                "boundary": boundary,
                "baseline": baseline,
                "bbox": {
                    "x": max(0, min(bxs)),
                    "y": max(0, min(bys)),
                    "w": max(bxs) - min(bxs),
                    "h": max(bys) - min(bys),
                },
            })
        return lines if lines else None
    except Exception as exc:
        print(f"  [warn] kraken seg failed: {exc}")
        return None


def _seg_from_tokens(page_tokens: list[dict], img_h: int) -> list[dict]:
    """Cluster tokens by y-coordinate to infer line bboxes."""
    if not page_tokens:
        return []
    # Collect token y-centers
    token_data = []
    for tok in page_tokens:
        bx, by, bw, bh = tok["bbox"]
        cy = by + bh / 2 if bh > 0 else by
        token_data.append((bx, by, bw, bh, cy))

    # Sort by y-center, group within 30px vertical tolerance
    token_data.sort(key=lambda t: t[4])
    groups: list[list] = []
    for td in token_data:
        if not groups or td[4] - groups[-1][-1][4] > 30:
            groups.append([td])
        else:
            groups[-1].append(td)

    # Sort groups top-to-bottom
    groups.sort(key=lambda g: min(t[4] for t in g))

    lines = []
    for i, grp in enumerate(groups):
        xs = [t[0] for t in grp]
        ys = [t[1] for t in grp]
        x2s = [t[0] + t[2] for t in grp]
        y2s = [t[1] + t[3] for t in grp]
        lx = max(0, min(xs))
        ly = max(0, min(ys))
        lx2 = max(x2s)
        ly2 = max(y2s)
        lw = lx2 - lx
        lh = max(ly2 - ly, 1)
        boundary = [[lx, ly], [lx2, ly], [lx2, ly2], [lx, ly2]]
        lines.append({
            "detection_order": i,
            "line_id": f"token_line_{i}",
            "boundary": boundary,
            "baseline": [],
            "bbox": {"x": lx, "y": ly, "w": lw, "h": lh},
        })
    return lines


def _crop_line(img: np.ndarray, boundary: list, padding: int = PADDING) -> np.ndarray:
    """Crop line region from image using boundary polygon bounding box."""
    pts = np.array(boundary, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    crop = img[y1:y2, x1:x2].copy()
    # White-background mask using boundary polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    shifted = pts - np.array([x1, y1])
    cv2.fillPoly(mask[y1:y2, x1:x2], [shifted], 255)
    result = np.full_like(crop, 255)
    result[mask[y1:y2, x1:x2] > 0] = crop[mask[y1:y2, x1:x2] > 0]
    return result


def _match_tokens_to_lines(
    tokens: list[dict], lines: list[dict]
) -> dict[int, list[dict]]:
    """Assign each token to the best-matching line index by y-overlap."""
    assignments: dict[int, list[dict]] = {i: [] for i in range(len(lines))}
    for tok in tokens:
        bx, by, bw, bh = tok["bbox"]
        cy = by + max(bh, 1) / 2
        best_i, best_overlap = 0, -1
        for i, line in enumerate(lines):
            lb = line["bbox"]
            ly, lh = lb["y"], lb["h"]
            overlap = min(by + bh, ly + lh) - max(by, ly)
            if overlap > best_overlap:
                best_overlap = overlap
                best_i = i
        assignments[best_i].append(tok)
    return assignments


def _line_ocr_text(tokens: list[dict]) -> str:
    """Concatenate tokens in RTL order (x descending)."""
    sorted_toks = sorted(tokens, key=lambda t: -t["bbox"][0])
    return " ".join(t["text"] for t in sorted_toks if t.get("text", "").strip())


def _mean_conf(tokens: list[dict]) -> float:
    confs = [t["confidence"] for t in tokens if "confidence" in t]
    return round(float(sum(confs) / len(confs)), 4) if confs else 0.0


def process_page(
    page_id: str,
    ocr_pages: dict[str, dict],
) -> dict:
    """Process one page: segment, crop, save. Returns summary dict."""
    img_path = IMAGES_DIR / page_id
    if not img_path.exists():
        print(f"  [skip] image not found: {img_path}")
        return {"page": page_id, "lines": 0, "status": "missing_image"}

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [skip] could not load: {img_path}")
        return {"page": page_id, "lines": 0, "status": "load_error"}

    h_img, w_img = img.shape[:2]
    out_dir = LINES_DIR / page_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get line boundaries
    print(f"  segmenting {page_id}...")
    seg_lines = _load_seg_from_cache(page_id)
    if seg_lines is not None:
        print(f"    loaded from cache: {len(seg_lines)} lines")
        source = "cache"
    else:
        seg_lines = _seg_via_kraken(page_id, img)
        if seg_lines is not None:
            print(f"    kraken segment: {len(seg_lines)} lines")
            source = "kraken"
        else:
            page_tokens = ocr_pages.get(page_id, {}).get("tokens", [])
            seg_lines = _seg_from_tokens(page_tokens, h_img)
            print(f"    token-cluster fallback: {len(seg_lines)} lines")
            source = "token_cluster"

    # Match OCR tokens to lines
    page_tokens = ocr_pages.get(page_id, {}).get("tokens", [])
    token_map = _match_tokens_to_lines(page_tokens, seg_lines)

    # Crop + save line images, build JSON records
    line_records = []
    for i, seg_line in enumerate(seg_lines):
        boundary = seg_line.get("boundary", [])
        if not boundary:
            continue

        crop = _crop_line(img, boundary)
        img_path_out = out_dir / f"line_{i:03d}.png"
        cv2.imwrite(str(img_path_out), crop)

        matched = token_map.get(i, [])
        ocr_text = _line_ocr_text(matched)
        conf = _mean_conf(matched)
        bbox = seg_line.get("bbox", {})

        line_records.append({
            "index": i,
            "image_path": str(img_path_out),
            "ocr_text": ocr_text,
            "baseline": seg_line.get("baseline", []),
            "bbox": [bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)],
            "confidence": conf,
        })

    manifest = {
        "page": page_id,
        "original_size": [w_img, h_img],
        "seg_source": source,
        "lines": line_records,
    }
    json_path = out_dir / "lines.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"    saved {len(line_records)} lines → {out_dir}/")
    return {"page": page_id, "lines": len(line_records), "status": "ok", "source": source}


def main():
    # Load OCR results
    ocr_pages: dict[str, dict] = {}
    if OCR_RESULTS.exists():
        data = json.loads(OCR_RESULTS.read_text())
        for pg in data.get("pages", []):
            fname = pg.get("filename", "")
            ocr_pages[fname] = pg
        print(f"Loaded OCR results for {len(ocr_pages)} pages")
    else:
        print(f"[warn] {OCR_RESULTS} not found — OCR text will be empty")

    LINES_DIR.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    results = []
    for page_id in PAGES:
        print(f"\nProcessing {page_id}...")
        r = process_page(page_id, ocr_pages)
        results.append(r)
        total_lines += r.get("lines", 0)

    print(f"\n{'='*50}")
    print(f"Done. {len(results)} pages processed, {total_lines} total lines.")
    print(f"{'Page':<12} {'Lines':>6}  {'Source':<16}  Status")
    print("-" * 50)
    for r in results:
        print(f"{r['page']:<12} {r.get('lines',0):>6}  {r.get('source',''):16}  {r['status']}")

    zero_pages = [r["page"] for r in results if r.get("lines", 0) == 0]
    if zero_pages:
        print(f"\n[warn] Pages with 0 lines: {zero_pages}")
    else:
        print("\nAll pages have at least 1 line.")


if __name__ == "__main__":
    main()
