"""Passim / rapidfuzz alignment of OCR page text against Ibn al-Awwam reference corpus.

Called from main.py AFTER confidence_engine/decision.py, BEFORE output/formatter.py.
Results go into aligned_text on the page result only.
Never called from lexicon_engine/.
Never overwrites TokenState.selected.

Graceful degradation: Passim (Java required) → rapidfuzz → None (skipped).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class AlignmentResult:
    corrected_text: str
    ocr_text: str
    confidence: float        # Levenshtein ratio of best match
    method: str              # "passim" | "rapidfuzz"
    accepted: bool           # confidence >= config.align.passim.threshold
    match_start: int
    match_end: int


def _load_corpus(config=None) -> str:
    """Load Ibn al-Awwam reference text from disk."""
    path = Path("data/lexicons/ibn_awwam/filaha.txt")
    if config is not None:
        ac = getattr(config, "align", None)
        pc = getattr(ac, "passim", None) if ac else None
        override = getattr(pc, "corpus_path", None) if pc else None
        if override:
            path = Path(override)
    if not path.exists():
        log.warning(f"align/openiti: reference corpus not found at {path}")
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _rapidfuzz_align(
    ocr_text: str,
    reference: str,
    window_chars: int,
    threshold: float,
) -> AlignmentResult:
    """Sliding-window partial_ratio alignment via rapidfuzz."""
    from rapidfuzz import fuzz

    n = len(reference)
    if n == 0:
        return AlignmentResult(
            corrected_text=ocr_text, ocr_text=ocr_text, confidence=0.0,
            method="rapidfuzz", accepted=False, match_start=0, match_end=0,
        )

    best_score = 0.0
    best_start = 0
    best_end = min(n, window_chars)
    step = max(1, window_chars // 2)

    for start in range(0, max(1, n - window_chars + 1), step):
        end = min(n, start + window_chars)
        score = fuzz.partial_ratio(ocr_text, reference[start:end]) / 100.0
        if score > best_score:
            best_score, best_start, best_end = score, start, end

    accepted = best_score >= threshold
    corrected = reference[best_start:best_end] if accepted else ocr_text
    log.debug(
        f"rapidfuzz align: conf={best_score:.3f} accepted={accepted} "
        f"window=[{best_start},{best_end}]"
    )
    return AlignmentResult(
        corrected_text=corrected,
        ocr_text=ocr_text,
        confidence=best_score,
        method="rapidfuzz",
        accepted=accepted,
        match_start=best_start,
        match_end=best_end,
    )


def _passim_align(
    ocr_text: str,
    reference: str,
    threshold: float,
) -> AlignmentResult | None:
    """Attempt Passim alignment (requires Java + spark-submit).

    Returns None if Java is absent or Passim is not configured.
    Full Passim integration (building inverted index, calling spark-submit) is
    out of scope for this container — this hook exists for future enablement.
    """
    if not shutil.which("java"):
        return None
    log.debug("align/openiti: Java present but Passim spark-submit not configured; using fallback")
    return None


def align_page(ocr_text: str, config=None) -> AlignmentResult | None:
    """Align OCR page text against the Ibn al-Awwam reference corpus.

    Returns None when:
    - alignment is disabled in config
    - reference corpus file is absent
    - both Passim and rapidfuzz are unavailable

    Never overwrites TokenState.selected — caller stores result in aligned_text only.
    """
    if not ocr_text or not ocr_text.strip():
        return None

    enabled = False
    threshold = 0.85
    window_chars = 2000
    fallback = "rapidfuzz"

    if config is not None:
        ac = getattr(config, "align", None)
        pc = getattr(ac, "passim", None) if ac else None
        if pc is not None:
            enabled = getattr(pc, "enabled", False)
            threshold = getattr(pc, "threshold", 0.85)
            window_chars = getattr(pc, "window_chars", 2000)
            fallback = getattr(pc, "fallback", "rapidfuzz")

    if not enabled:
        return None

    reference = _load_corpus(config)
    if not reference:
        return None

    result = _passim_align(ocr_text, reference, threshold)
    if result is not None:
        return result

    if fallback == "rapidfuzz":
        try:
            return _rapidfuzz_align(ocr_text, reference, window_chars, threshold)
        except ImportError:
            log.warning("align/openiti: rapidfuzz not installed — skipping alignment")

    return None
