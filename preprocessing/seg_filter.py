"""Filter Kraken blla segmentation results to remove noise lines.

Called from generate_line_crops.py and the /api/import segment + accept-lines
endpoints. Accepts line dicts with a 'bbox' field shaped either as
[x, y, w, h] (the /api/import schema) or {"x","y","w","h"} (generate_line_crops.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LineFilterConfig:
    min_height_px: int = 20
    min_width_px: int = 80
    min_area_px2: int = 1600
    max_lines_per_page: int = 60


def config_from_global() -> LineFilterConfig:
    """Build a LineFilterConfig from config.yaml's segmentation.line_filters.

    Falls back to the dataclass defaults if the section is missing.
    """
    try:
        from utils.config import get_config

        lf = get_config().segmentation.line_filters
        return LineFilterConfig(
            min_height_px=lf.get("min_height_px", 20),
            min_width_px=lf.get("min_width_px", 80),
            min_area_px2=lf.get("min_area_px2", 1600),
            max_lines_per_page=lf.get("max_lines_per_page", 60),
        )
    except Exception:
        log.warning("seg_filter: segmentation.line_filters not found in config, using defaults")
        return LineFilterConfig()


def _bbox_wh(bbox) -> tuple[float, float]:
    if isinstance(bbox, dict):
        return bbox.get("w", 0), bbox.get("h", 0)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return bbox[2], bbox[3]
    return 0, 0


def filter_lines(lines: list[dict], cfg: LineFilterConfig | None = None) -> list[dict]:
    """Filter a list of line dicts to remove noise.

    Removes lines with h < min_height_px, w < min_width_px, or w*h < min_area_px2.
    Lines whose 'id' (or 'line_id') starts with "manual_" always bypass the
    size filter — the user explicitly drew them. Truncates to
    max_lines_per_page if exceeded (logs a warning). Never raises.
    """
    cfg = cfg or LineFilterConfig()
    if not lines:
        return []

    kept = []
    for line in lines:
        try:
            line_id = str(line.get("id") or line.get("line_id") or "")
            if line_id.startswith("manual_"):
                kept.append(line)
                continue
            w, h = _bbox_wh(line.get("bbox"))
            if h < cfg.min_height_px or w < cfg.min_width_px or (w * h) < cfg.min_area_px2:
                continue
            kept.append(line)
        except Exception:
            log.warning("seg_filter: skipping malformed line entry: %r", line)

    if len(kept) > cfg.max_lines_per_page:
        log.warning(
            "seg_filter: %d lines exceeds max_lines_per_page=%d, truncating",
            len(kept), cfg.max_lines_per_page,
        )
        kept = kept[: cfg.max_lines_per_page]

    return kept
