"""Crop layout regions from a page image and stitch OCR results back to page space."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)


class RegionCrop(TypedDict):
    image: np.ndarray
    region_id: str
    bbox_in_page: tuple[int, int, int, int]  # x, y, w, h in page space
    type: str


def crop_regions(page_image: np.ndarray, regions: list[dict]) -> list[RegionCrop]:
    """Crop each detected layout region from the page image.

    If regions is empty, returns one crop covering the full page with
    region_id='full'.

    All returned crops are independent numpy arrays (copied, not views).
    """
    if not regions:
        h, w = page_image.shape[:2]
        regions = [{"x": 0, "y": 0, "w": w, "h": h, "type": "text_block", "region_id": "full"}]

    crops: list[RegionCrop] = []
    ph, pw = page_image.shape[:2]

    for region in regions:
        x = max(0, int(region["x"]))
        y = max(0, int(region["y"]))
        w = int(region["w"])
        h = int(region["h"])

        # Clamp to page boundaries
        x2 = min(pw, x + w)
        y2 = min(ph, y + h)

        if x2 <= x or y2 <= y:
            log.warning(f"region_id={region.get('region_id')} is zero-size after clamping; skipping")
            continue

        crop_img = page_image[y:y2, x:x2].copy()
        crops.append(RegionCrop(
            image=crop_img,
            region_id=str(region.get("region_id", "unknown")),
            bbox_in_page=(x, y, x2 - x, y2 - y),
            type=str(region.get("type", "text_block")),
        ))
        log.debug(f"crop region_id={region.get('region_id')} bbox=({x},{y},{x2-x},{y2-y})")

    return crops


def stitch_results(region_results: list, page_index: int):
    """Merge per-region OCRResult objects into a single page-level OCRResult.

    Each region result must be an OCRResult whose WordTokens have bboxes in
    crop space. This function translates every bbox to page space using the
    region_id → page offset mapping embedded in the OCRResult.raw dict
    (key: 'crop_bbox_in_page': [x_off, y_off, w, h]).

    WordTokens in the merged result always carry page-space bboxes.
    """
    from ocr_engine.schema import OCRResult, WordToken

    if not region_results:
        return OCRResult(
            text="", words=[], confidence=0.0,
            page_index=page_index, source="ensemble",
        )

    all_words: list[WordToken] = []
    all_texts: list[str] = []
    confidences: list[float] = []

    for result in region_results:
        crop_bbox = result.raw.get("crop_bbox_in_page")  # [x_off, y_off, w, h]
        x_off = int(crop_bbox[0]) if crop_bbox else 0
        y_off = int(crop_bbox[1]) if crop_bbox else 0

        for token in result.words:
            tx, ty, tw, th = token.bbox
            page_bbox = (tx + x_off, ty + y_off, tw, th)
            all_words.append(WordToken(
                text=token.text,
                confidence=token.confidence,
                bbox=page_bbox,
                page_index=page_index,
                source=token.source,
                region_id=token.region_id,
            ))

        if result.text:
            all_texts.append(result.text)
        confidences.append(result.confidence)

    page_confidence = float(np.mean(confidences)) if confidences else 0.0
    page_text = " ".join(all_texts)

    log.debug(
        f"stitch page={page_index} regions={len(region_results)} "
        f"tokens={len(all_words)} conf={page_confidence:.3f}"
    )

    return OCRResult(
        text=page_text,
        words=all_words,
        confidence=page_confidence,
        page_index=page_index,
        source="ensemble",
        raw={"region_count": len(region_results)},
    )
