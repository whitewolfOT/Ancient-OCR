"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

VALID_MODES = {"clean", "annotated", "debug"}


class ProcessResponse(BaseModel):
    mode: str
    text: str
    word_count: int
    page_count: int
    tokens: list[dict] | None = None
    raw_ocr: list[dict] | None = None


class HealthResponse(BaseModel):
    status: str
    engines: dict[str, bool]


class FeedbackSubmitRequest(BaseModel):
    image_path: str
    bbox: list[int]
    page_index: int
    predicted: str
    ground_truth: str
    source_file: str

    @field_validator("bbox")
    @classmethod
    def bbox_length(cls, v):
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 integers [x, y, w, h]")
        return v

    @field_validator("bbox", mode="before")
    @classmethod
    def bbox_non_negative(cls, v):
        if any(int(x) < 0 for x in v):
            raise ValueError("bbox values must be non-negative")
        return v


class FeedbackSubmitResponse(BaseModel):
    id: str
    status: str = "ok"


class CalibrateResponse(BaseModel):
    sample_size: int
    current_weights: dict[str, float]
    suggested_weights: dict[str, float]
    delta: dict[str, float]
    warning: str | None = None


class FeedbackStatsResponse(BaseModel):
    total: int
    applied: int
    pending: int
    by_source_file: dict[str, int]
    error_rate: float
