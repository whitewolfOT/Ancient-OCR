"""Shared pipeline contracts — build once, never change field names."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class LexiconEntry(BaseModel):
    lemma: str
    root: Optional[str] = None
    pattern: Optional[str] = None   # wazn
    gloss: str
    source: str
    era: str                     # "classical" | "modern"
    domain: Optional[str] = None
    examples: List[str] = []
    priority: int


class Candidate(BaseModel):
    text: str
    reason: str                  # "spelling_variant"|"normalization"|"root_alt"|"morph_alt"|"identity"
    lexicon_entries: List[LexiconEntry] = []
    features: Dict[str, Union[float, str]] = {}  # numeric scores + string labels (e.g. morph_evidence)
    score: Optional[float] = None


class RankedResult(BaseModel):
    best: Optional[Candidate]
    ranked: List[Candidate]
    selected_text: str


class TokenState(BaseModel):
    original: str
    normalized: str
    normalization_log: List[Dict[str, str]]  # {"step","before","after","rule"} per change
    candidates: List[Candidate]
    selected: str
    confidence: float
    sources: List[str]
    decision: str               # "accept"|"accept_with_note"|"uncertain"|"review_required"
    reason_code: str
    bbox: Optional[Tuple[int, int, int, int]] = None  # page-space; needed for review crops
    page_index: int = 0
    line_id: Optional[str] = None                     # Kraken line UUID (None for other backends)
    baseline: Optional[List[Tuple[int, int]]] = None  # Kraken baseline points, page-space


class FeedbackEntry(BaseModel):
    id: str                           # UUID
    image_path: str                   # path to the cropped image
    bbox: Tuple[int, int, int, int]   # page space
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
    norm_log: List[Dict[str, str]],
    candidates: List[Candidate],
    ranked_result: RankedResult,
    conf: float,
    decision: str,
    reason_code: str,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    page_index: int = 0,
    line_id: Optional[str] = None,
    baseline: Optional[List[Tuple[int, int]]] = None,
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
        line_id=line_id,
        baseline=baseline,
    )
