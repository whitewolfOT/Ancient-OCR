"""Shared pipeline contracts — build once, never change field names."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LexiconEntry(BaseModel):
    lemma: str
    root: str | None = None
    pattern: str | None = None   # wazn
    gloss: str
    source: str
    era: str                     # "classical" | "modern"
    domain: str | None = None
    examples: list[str] = []
    priority: int


class Candidate(BaseModel):
    text: str
    reason: str                  # "spelling_variant"|"normalization"|"root_alt"|"morph_alt"|"identity"
    lexicon_entries: list[LexiconEntry] = []
    features: dict[str, float] = {}  # lexicon_score, morph_score, ocr_score, context_score
    score: float | None = None


class RankedResult(BaseModel):
    best: Candidate | None
    ranked: list[Candidate]
    selected_text: str


class TokenState(BaseModel):
    original: str
    normalized: str
    normalization_log: list[dict[str, str]]  # {"step","before","after","rule"} per change
    candidates: list[Candidate]
    selected: str
    confidence: float
    sources: list[str]
    decision: str               # "accept"|"accept_with_note"|"uncertain"|"review_required"
    reason_code: str
    bbox: tuple[int, int, int, int] | None = None  # page-space; needed for review crops
    page_index: int = 0


class FeedbackEntry(BaseModel):
    id: str                           # UUID
    image_path: str                   # path to the cropped image
    bbox: tuple[int, int, int, int]   # page space
    page_index: int
    predicted: str
    ground_truth: str
    source_file: str
    submitted_at: str                 # ISO 8601
    applied: bool = False             # True once used in calibration


def build_token_state(
    *,
    original: str,
    normalized: str,
    norm_log: list[dict[str, str]],
    candidates: list[Candidate],
    ranked_result: RankedResult,
    conf: float,
    decision: str,
    reason_code: str,
    bbox: tuple[int, int, int, int] | None = None,
    page_index: int = 0,
) -> TokenState:
    sources = list({e.source for c in candidates for e in c.lexicon_entries})
    return TokenState(
        original=original,
        normalized=normalized,
        normalization_log=norm_log,
        candidates=candidates,
        selected=ranked_result.selected_text,
        confidence=conf,
        sources=sources,
        decision=decision,
        reason_code=reason_code,
        bbox=bbox,
        page_index=page_index,
    )
