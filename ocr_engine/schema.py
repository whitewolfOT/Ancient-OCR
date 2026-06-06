"""Shared OCR data contracts — build once, never change field names."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class WordToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float          # 0..1
    bbox: tuple[int, int, int, int]  # x, y, w, h (top-left origin, page space)
    page_index: int
    source: str                # "paddle" | "tesseract" | "trocr" | "kraken" | "ensemble"
    region_id: str | None = None  # set by region_cropper; None = full-page
    line_id: str | None = None    # Kraken line UUID from segment JSON
    baseline: list[tuple[int, int]] | None = None  # raw baseline points, page-space (Kraken only)


class OCRResult(BaseModel):
    text: str
    words: list[WordToken]
    confidence: float          # page-level aggregate 0..1
    page_index: int
    source: str
    raw: dict = {}             # engine-specific payload (kept for debug)
