"""Recalibrate confidence weights from stored feedback corrections.

Never auto-applies weights — always returns suggestions only.
Caller must call apply_weights() explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_MIN_ENTRIES = 50


@dataclass
class CalibrationResult:
    suggested_weights: dict
    current_weights: dict
    delta: dict
    sample_size: int
    warning: str | None = None


def calibrate(config=None, min_entries: int | None = None) -> CalibrationResult:
    """Compute suggested scoring weights from pending feedback.

    Requires at least `min_entries` pending entries. If fewer are available,
    returns a CalibrationResult with a `warning` field and no suggested_weights.

    Never auto-applies. Returns suggestions only.
    """
    from training.feedback_store import load, stats

    threshold = min_entries
    if threshold is None:
        threshold = _DEFAULT_MIN_ENTRIES
        if config is not None:
            try:
                threshold = config.training.min_feedback_for_calibration
            except AttributeError:
                pass

    current = _current_weights(config)
    pending = load(pending_only=True, config=config)

    if len(pending) < threshold:
        return CalibrationResult(
            suggested_weights={},
            current_weights=current,
            delta={},
            sample_size=len(pending),
            warning=(
                f"Insufficient feedback: {len(pending)} entries available, "
                f"{threshold} required for calibration."
            ),
        )

    # Compute per-component accuracy from corrections
    # For each entry where predicted != ground_truth, we attribute the miss
    # to the component with the lowest signal. This is a heuristic approach.
    correct = [e for e in pending if e.predicted == e.ground_truth]
    accuracy = len(correct) / len(pending) if pending else 0.0

    # Simple heuristic: scale weights proportionally to observed accuracy
    # The more errors, the more we should up-weight lexicon (most reliable signal)
    # and slightly reduce OCR weight (noisy input)
    error_rate = 1.0 - accuracy
    lex_boost = min(0.05, error_rate * 0.1)
    ocr_reduction = min(0.05, error_rate * 0.05)

    suggested = {
        "ocr_weight":        round(max(0.15, current["ocr_weight"] - ocr_reduction), 3),
        "lexicon_weight":    round(min(0.45, current["lexicon_weight"] + lex_boost), 3),
        "morphology_weight": round(current["morphology_weight"], 3),
        "context_weight":    round(current["context_weight"], 3),
    }
    # Normalise to sum to 1.0
    total = sum(suggested.values())
    suggested = {k: round(v / total, 3) for k, v in suggested.items()}

    delta = {k: round(suggested[k] - current.get(k, 0), 3) for k in suggested}

    log.info(
        f"calibrate sample={len(pending)} accuracy={accuracy:.2f} "
        f"suggested={suggested}"
    )
    return CalibrationResult(
        suggested_weights=suggested,
        current_weights=current,
        delta=delta,
        sample_size=len(pending),
    )


def apply_weights(result: CalibrationResult, config_path: str = "config.yaml") -> None:
    """Write suggested weights to config.yaml. Must be called explicitly."""
    if not result.suggested_weights:
        raise ValueError("No suggested weights to apply (calibration warning present).")

    import yaml
    from pathlib import Path

    path = Path(config_path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    raw.setdefault("scoring", {})
    for key, value in result.suggested_weights.items():
        raw["scoring"][key] = value

    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(raw, fh, allow_unicode=True, default_flow_style=False)

    # Invalidate config singleton so next get_config() reloads
    from utils.config import reload_config
    reload_config(config_path)
    log.info(f"apply_weights written to {config_path} weights={result.suggested_weights}")


def _current_weights(config=None) -> dict:
    defaults = {
        "ocr_weight": 0.30,
        "lexicon_weight": 0.30,
        "morphology_weight": 0.20,
        "context_weight": 0.20,
    }
    if config is None:
        return defaults
    s = getattr(config, "scoring", None)
    if s is None:
        return defaults
    return {
        "ocr_weight": getattr(s, "ocr_weight", defaults["ocr_weight"]),
        "lexicon_weight": getattr(s, "lexicon_weight", defaults["lexicon_weight"]),
        "morphology_weight": getattr(s, "morphology_weight", defaults["morphology_weight"]),
        "context_weight": getattr(s, "context_weight", defaults["context_weight"]),
    }
