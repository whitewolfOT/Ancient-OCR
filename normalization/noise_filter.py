"""Remove non-Arabic noise characters from OCR output."""

from __future__ import annotations

import re
import unicodedata

from utils.logging import get_logger

log = get_logger(__name__)

# Valid Arabic Unicode ranges to preserve
_ARABIC_RANGES = (
    (0x0600, 0x06FF),   # Arabic
    (0x0750, 0x077F),   # Arabic Supplement
    (0x08A0, 0x08FF),   # Arabic Extended-A
    (0xFB50, 0xFDFF),   # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
)

# Whitespace and common punctuation to keep
_KEEP_CATEGORIES = {"Zs", "Po", "Pd", "Ps", "Pe"}  # space, punctuation


def _is_valid(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _ARABIC_RANGES:
        if lo <= cp <= hi:
            return True
    cat = unicodedata.category(ch)
    if cat in _KEEP_CATEGORIES:
        return True
    if cat == "Cc" and ch in ("\n", "\r", "\t"):
        return True
    # ASCII digits and Latin letters (mixed documents)
    if "\x20" <= ch <= "\x7E":
        return True
    return False


def clean_noise(text: str) -> tuple[str, list[dict]]:
    """Remove stray noise characters; preserve all valid Arabic and ASCII.

    Returns:
        (cleaned_text, change_log)  where each log entry is
        {"step": "noise_filter", "before": char, "after": "", "rule": rule_name}
    """
    log_entries: list[dict] = []
    out: list[str] = []

    for ch in text:
        if _is_valid(ch):
            out.append(ch)
        else:
            cat = unicodedata.category(ch)
            cp = ord(ch)
            if cat == "Cc":
                rule = "control_char_removal"
            elif cat in {"Cf"}:
                rule = "format_char_removal"
            elif cp > 0x10000:
                rule = "non_bmp_removal"
            else:
                rule = "noise_char_removal"
            log_entries.append({
                "step": "noise_filter",
                "before": ch,
                "after": "",
                "rule": rule,
            })

    cleaned = "".join(out)
    # Collapse runs of whitespace to single space
    cleaned = re.sub(r" {2,}", " ", cleaned).strip()
    log.debug(f"clean_noise removed={len(log_entries)} chars")
    return cleaned, log_entries
