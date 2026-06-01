"""Map a confidence score to a decision label and reason code."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

_LABELS = ("accept", "accept_with_note", "uncertain", "review_required")


def decide(confidence: float, config=None) -> tuple[str, str]:
    """Return (decision_label, reason_code) for a confidence score.

    Thresholds (from config.decision, conservative defaults):
      >= 0.90  → accept
      >= 0.70  → accept_with_note
      >= 0.50  → uncertain
      < 0.50   → review_required

    reason_code format: "conf_{label}_{int(confidence * 100)}"
    e.g. "conf_accept_with_note_77"
    """
    t_accept = 0.90
    t_note = 0.70
    t_uncertain = 0.50

    if config is not None:
        d = getattr(config, "decision", None)
        if d is not None:
            t_accept = getattr(d, "accept", t_accept)
            t_note = getattr(d, "accept_with_note", t_note)
            t_uncertain = getattr(d, "uncertain", t_uncertain)

    c = max(0.0, min(1.0, float(confidence)))

    if c >= t_accept:
        label = "accept"
    elif c >= t_note:
        label = "accept_with_note"
    elif c >= t_uncertain:
        label = "uncertain"
    else:
        label = "review_required"

    label_slug = label.replace("_", "_")
    reason_code = f"conf_{label_slug}_{int(c * 100)}"

    log.debug(f"decide confidence={c:.4f} → {label} ({reason_code})")
    return label, reason_code
