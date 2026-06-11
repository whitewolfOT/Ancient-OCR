"""Shared OCR data contracts — build once, never change field names."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict


class WordToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float          # 0..1
    bbox: Tuple[int, int, int, int]  # x, y, w, h (top-left origin, page space)
    page_index: int
    source: str                # "paddle" | "tesseract" | "trocr" | "kraken" | "ensemble"
    region_id: Optional[str] = None
    line_id: Optional[str] = None
    baseline: Optional[List[Tuple[int, int]]] = None
    char_confidences: Optional[List[float]] = None   # per-character, same length as text
    candidates: Optional[List[Dict[str, object]]] = None  # [{text, confidence}, ...]


class OCRResult(BaseModel):
    text: str
    words: List[WordToken]
    confidence: float          # page-level aggregate 0..1
    page_index: int
    source: str
    raw: dict = {}             # engine-specific payload (kept for debug)
