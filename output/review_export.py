"""Export uncertain/review_required tokens to a review queue."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from utils.logging import get_logger

log = get_logger(__name__)

_REVIEW_DECISIONS = {"uncertain", "review_required"}


def export_review_queue(
    token_states: list,
    source_image_pages: list | None,
    source_file: str,
    config=None,
) -> dict:
    """Build a review queue for tokens that need human attention.

    Only tokens with decision in ('uncertain', 'review_required') are included.
    accept / accept_with_note tokens are NEVER included.

    Args:
        token_states:       Full list of TokenState objects from the pipeline.
        source_image_pages: list of np.ndarray (one per page) for crop extraction.
        source_file:        Original source filename, used for crop paths.
        config:             Pipeline config.

    Returns:
        dict with keys: {review_count, items: [...], source_file}
        Each item: {token_id, original, normalized, selected, alternatives,
                    decision, reason_code, confidence, crop_path, bbox, page_index,
                    suggestion}
    """
    crop_dir = "data/review/"
    if config is not None:
        out = getattr(config, "output", None)
        if out is not None:
            rv = getattr(out, "review", None)
            if rv is not None:
                crop_dir = getattr(rv, "crop_dir", crop_dir)

    stem = Path(source_file).stem if source_file else "unknown"
    items = []

    for i, ts in enumerate(token_states):
        if ts.decision not in _REVIEW_DECISIONS:
            continue

        token_id = f"{stem}_p{ts.page_index}_t{i}"
        crop_path = _extract_crop(ts, source_image_pages, crop_dir, token_id)

        alternatives = [
            {"text": c.text, "score": c.score, "reason": c.reason}
            for c in (ts.candidates or [])
            if c.text != ts.selected and c.score is not None
        ]
        alternatives.sort(key=lambda a: a["score"] or 0, reverse=True)

        # suggestion: best non-identity candidate, or None
        suggestion = next(
            (a["text"] for a in alternatives if a["reason"] != "identity"),
            None,
        )

        baseline = getattr(ts, "baseline", None)
        items.append({
            "token_id": token_id,
            "original": ts.original,
            "normalized": ts.normalized,
            "selected": ts.selected,
            "alternatives": alternatives[:5],
            "decision": ts.decision,
            "reason_code": ts.reason_code,
            "confidence": ts.confidence,
            "crop_path": crop_path,
            "bbox": list(ts.bbox) if ts.bbox else None,
            "page_index": ts.page_index,
            "suggestion": suggestion,
            "line_id": getattr(ts, "line_id", None),
            "baseline": [list(pt) for pt in baseline] if baseline else None,
        })

    log.debug(f"review_queue source={source_file} total_tokens={len(token_states)} review={len(items)}")
    return {
        "source_file": source_file,
        "review_count": len(items),
        "items": items,
    }


def to_review_json(queue: dict, indent: int = 2) -> str:
    return json.dumps(queue, ensure_ascii=False, sort_keys=False, indent=indent)


def to_review_html(queue: dict) -> str:
    """Minimal HTML with inline base64 crop images."""
    rows = []
    for item in queue["items"]:
        img_tag = ""
        if item["crop_path"] and os.path.exists(item["crop_path"]):
            try:
                with open(item["crop_path"], "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                img_tag = f'<img src="data:image/png;base64,{b64}" style="max-height:40px">'
            except Exception:
                pass

        alts = ", ".join(
            f"{a['text']} ({a['score']:.2f})" for a in item["alternatives"][:3]
        )
        rows.append(
            f"<tr>"
            f"<td>{item['token_id']}</td>"
            f"<td dir='rtl'>{item['original']}</td>"
            f"<td dir='rtl'>{item['selected']}</td>"
            f"<td>{item['decision']}</td>"
            f"<td>{item['confidence']:.2f}</td>"
            f"<td dir='rtl'>{item.get('suggestion') or '—'}</td>"
            f"<td dir='rtl'>{alts}</td>"
            f"<td>{img_tag}</td>"
            f"</tr>"
        )

    body = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head><meta charset="UTF-8"><title>Review Queue — {queue['source_file']}</title>
<style>table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px}}</style>
</head>
<body>
<h2>Review Queue: {queue['source_file']} ({queue['review_count']} tokens)</h2>
<table>
<thead><tr><th>ID</th><th>Original</th><th>Selected</th><th>Decision</th>
<th>Conf</th><th>Suggestion</th><th>Alternatives</th><th>Crop</th></tr></thead>
<tbody>{body}</tbody>
</table>
</body></html>"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_crop(ts, image_pages, crop_dir: str, token_id: str) -> str | None:
    """Save the token's bbox crop as a PNG; return the path or None.

    Kraken tokens: if ts.baseline is set, draws a red baseline overlay on the crop
    so reviewers can verify segmentation quality alongside the text.
    """
    if not ts.bbox or image_pages is None:
        return None
    try:
        import cv2
        import numpy as np

        pi = ts.page_index
        if pi >= len(image_pages):
            return None
        page_img = image_pages[pi]
        x, y, w, h = ts.bbox
        ph, pw = page_img.shape[:2]
        x2, y2 = min(pw, x + w), min(ph, y + h)
        if x2 <= x or y2 <= y:
            return None

        crop = page_img[y:y2, x:x2].copy()

        baseline = getattr(ts, "baseline", None)
        if baseline and len(baseline) >= 2:
            # Translate page-space baseline points into crop-local coords
            pts = [(max(0, int(bx) - x), max(0, int(by) - y)) for bx, by in baseline]
            for i in range(len(pts) - 1):
                cv2.line(crop, pts[i], pts[i + 1], (0, 0, 255), 1, cv2.LINE_AA)

        out_dir = Path(crop_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{token_id}.png")
        cv2.imwrite(out_path, crop)
        return out_path
    except Exception as exc:
        log.debug(f"crop save failed for {token_id}: {exc}")
        return None
