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


# ---------------------------------------------------------------------------
# Document calibration schemas
# ---------------------------------------------------------------------------

class ClusterInfo(BaseModel):
    cluster_id: str
    label: str
    page_count: int
    representative_page_id: str


class UploadResponse(BaseModel):
    doc_id: str
    page_count: int
    clusters: list[ClusterInfo]


class PageInfo(BaseModel):
    page_id: str
    page_num: int
    cluster_id: str
    similarity_to_representative: float
    status: str
    has_ground_truth: bool
    has_settings: bool
    thumbnail_url: str


class PreviewRequest(BaseModel):
    clahe: float = 2.0
    denoise: int = 3
    deskew_threshold: float = 0.5
    binarization: str = "Adaptive"


class SettingsApplied(BaseModel):
    clahe: float
    denoise: int
    deskew_threshold: float
    binarization: str


class PreviewResponse(BaseModel):
    preview_image_b64: str
    settings_applied: SettingsApplied
    preview_id: str
    processing_time_ms: float


class GroundTruthRequest(BaseModel):
    text: str
    page_id: str


class GroundTruthResponse(BaseModel):
    page_id: str
    saved_at: str


class ClusterSettingsPayload(BaseModel):
    clahe: float = 2.0
    denoise: int = 3
    deskew_threshold: float = 0.5
    binarization: str = "Adaptive"


class ApplyClusterSettingsRequest(BaseModel):
    cluster_id: str
    settings: ClusterSettingsPayload


class ApplyClusterSettingsResponse(BaseModel):
    pages_updated: int
    cluster_id: str
