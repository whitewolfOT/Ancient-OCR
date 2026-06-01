"""Conservative Arabic text normalization with per-step change logging."""

from __future__ import annotations

import re

from utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Folding tables (documented here; reverse_normalization_map() echoes these)
# ---------------------------------------------------------------------------

_ALEF_VARIANTS = str.maketrans("أإآٱ", "اااا")   # alef variants → bare alef
_ALEF_MAQSURA = str.maketrans("ى", "ي")           # alef maqsura → ya
_TAA_MARBUTA = str.maketrans("ة", "ه")            # taa marbuta → ha (off by default)
_TATWEEL = re.compile(r"ـ+")                       # kashida/tatweel
_DIACRITICS = re.compile(
    r"[ً-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]"
)


def normalize_text(text: str, config=None) -> tuple[str, list[dict]]:
    """Apply configurable Arabic normalization rules in canonical order.

    Rules applied (each toggled by config):
      1. tatweel (kashida) removal
      2. alef variant folding  (أ إ آ ٱ → ا)
      3. alef maqsura folding  (ى → ي)
      4. taa marbuta folding   (ة → ه)  — OFF by default
      5. hamza normalization   — conservative, OFF by default
      6. diacritic handling    (strip_all | strip_partial | preserve)
      7. whitespace normalisation

    Returns:
        (normalized_text, change_log) where each log entry is
        {"step": rule_name, "before": str, "after": str, "rule": rule_name}
    """
    cfg = _get_norm_config(config)
    change_log: list[dict] = []
    current = text

    if cfg["tatweel"]:
        current, entries = _apply_tatweel(current)
        change_log.extend(entries)

    if cfg["alef_variants"]:
        current, entries = _apply_table(current, _ALEF_VARIANTS, "alef_variant_fold")
        change_log.extend(entries)

    if cfg["alef_maqsura"]:
        current, entries = _apply_table(current, _ALEF_MAQSURA, "alef_maqsura_fold")
        change_log.extend(entries)

    if cfg["taa_marbuta"]:
        current, entries = _apply_table(current, _TAA_MARBUTA, "taa_marbuta_fold")
        change_log.extend(entries)

    diacritics_mode = cfg["diacritics"]
    if diacritics_mode == "strip_all":
        current, entries = _strip_diacritics(current, partial=False)
        change_log.extend(entries)
    elif diacritics_mode == "strip_partial":
        current, entries = _strip_diacritics(current, partial=True)
        change_log.extend(entries)
    # "preserve" → no-op

    # Whitespace normalization
    normalized_ws = re.sub(r"\s+", " ", current).strip()
    if normalized_ws != current:
        change_log.append({
            "step": "whitespace_normalize",
            "before": current,
            "after": normalized_ws,
            "rule": "whitespace_normalize",
        })
    current = normalized_ws

    log.debug(f"normalize_text changes={len(change_log)}")
    return current, change_log


def normalize_token(text: str, config=None) -> tuple[str, list[dict]]:
    """Convenience wrapper for single-token normalization."""
    return normalize_text(text, config)


def reverse_normalization_map() -> dict:
    """Document the folding rules applied by normalize_text.

    This is a reference/debug helper ONLY. It is NOT a lossless inverse:
    several rules are many-to-one (e.g. أ/إ/آ all map to ا) and cannot
    be reversed from the normalized form alone. Use the per-token
    normalization_log in TokenState for traceability.
    """
    return {
        "alef_variant_fold": {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"},
        "alef_maqsura_fold": {"ى": "ي"},
        "taa_marbuta_fold": {"ة": "ه"},
        "tatweel_removal": {"ـ": ""},
        "diacritics_strip": "all shadda/harakat/tanwin → removed",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_tatweel(text: str) -> tuple[str, list[dict]]:
    entries: list[dict] = []
    def _rep(m):
        entries.append({"step": "tatweel_removal", "before": m.group(), "after": "", "rule": "tatweel_removal"})
        return ""
    result = _TATWEEL.sub(_rep, text)
    return result, entries


def _apply_table(text: str, table, rule: str) -> tuple[str, list[dict]]:
    entries: list[dict] = []
    result = []
    for ch in text:
        mapped = ch.translate(table)
        if mapped != ch:
            entries.append({"step": rule, "before": ch, "after": mapped, "rule": rule})
        result.append(mapped)
    return "".join(result), entries


def _strip_diacritics(text: str, partial: bool) -> tuple[str, list[dict]]:
    """Strip diacritics; partial mode keeps shadda (ّ) as it's structurally significant."""
    SHADDA = "ّ"
    entries: list[dict] = []
    result = []
    for ch in text:
        if _DIACRITICS.match(ch):
            if partial and ch == SHADDA:
                result.append(ch)
                continue
            entries.append({"step": "diacritic_strip", "before": ch, "after": "", "rule": "diacritic_strip"})
        else:
            result.append(ch)
    return "".join(result), entries


def _get_norm_config(config) -> dict:
    defaults = {
        "alef_variants": True,
        "alef_maqsura": True,
        "taa_marbuta": False,
        "tatweel": True,
        "hamza": False,
        "diacritics": "strip_partial",
    }
    if config is None:
        return defaults
    n = getattr(config, "normalization", None)
    if n is None:
        return defaults
    return {
        "alef_variants": getattr(n, "alef_variants", defaults["alef_variants"]),
        "alef_maqsura": getattr(n, "alef_maqsura", defaults["alef_maqsura"]),
        "taa_marbuta": getattr(n, "taa_marbuta", defaults["taa_marbuta"]),
        "tatweel": getattr(n, "tatweel", defaults["tatweel"]),
        "hamza": getattr(n, "hamza", defaults["hamza"]),
        "diacritics": getattr(n, "diacritics", defaults["diacritics"]),
    }
