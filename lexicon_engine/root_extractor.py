"""
Root extractor for Arabic words.
All tiers degrade gracefully — never raises, always returns str | None.
Loads CAMeL analyzer once lazily (not at import time).

Tier 1: CAMeL Tools (calima-msa-r13 — returns dicts; root in dot-notation e.g. ك.ت.ب)
Tier 2: Pattern-based extraction (prefix/suffix stripping)
Tier 3: Farasa stem approximation (requires Java — usually absent)
Tier 4: Abstain → None
"""
from __future__ import annotations

import re
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_camel_analyzer = None
_farasa_segmenter = None
_camel_available = False
_farasa_available = False

# Arabic Unicode block
_ARABIC_RE = re.compile(r'[؀-ۿ]')


def _init_camel(preset: str = "calima-msa-r13") -> bool:
    global _camel_analyzer, _camel_available
    if _camel_available:
        return True
    try:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB
        db = MorphologyDB.builtin_db(preset)
        _camel_analyzer = Analyzer(db)
        _camel_available = True
        return True
    except Exception as e:
        logger.warning(f"CAMeL Tools unavailable: {e}")
        return False


def _init_farasa() -> bool:
    global _farasa_segmenter, _farasa_available
    if _farasa_available:
        return True
    try:
        from farasapy.farasa.segmenter import FarasaSegmenter
        _farasa_segmenter = FarasaSegmenter(interactive=True)
        _farasa_available = True
        return True
    except Exception as e:
        logger.warning(f"Farasa unavailable (Java required): {e}")
        return False


def _tier2_pattern(word: str) -> str | None:
    """
    Pattern-based root extraction for common Arabic morphological forms.
    Strips common prefixes (ال، و، ف، ب، ل، م) and suffixes (ة، ات، ون، ين).
    Returns best-guess root or None.
    """
    if not word:
        return None

    w = re.sub(r'^ال', '', word)        # definite article
    w = re.sub(r'^[وفبلكم]', '', w)    # common single-letter prefixes
    w = re.sub(r'[هاتونين]+$', '', w)  # common suffixes
    w = re.sub(r'ة$', '', w)           # taa marbuta

    arabic_letters = re.sub(r'[^؀-ۿ]', '', w)
    if len(arabic_letters) == 3:
        return arabic_letters
    if len(arabic_letters) == 4:
        return arabic_letters  # quadrilateral root
    return None


@lru_cache(maxsize=10000)
def extract_root(word: str, camel_preset: str = "calima-msa-r13") -> str | None:
    """
    Extract Arabic root with tiered fallback.
    Returns trilateral/quadrilateral root string or None.
    Results are cached in memory via lru_cache.

    CAMeL Tools returns dicts and encodes roots in dot-notation (ك.ت.ب) —
    dots are stripped before returning.
    """
    if not word or not _ARABIC_RE.search(word):
        return None

    # Tier 1: CAMeL Tools
    if _init_camel(camel_preset):
        try:
            analyses = _camel_analyzer.analyze(word)
            if analyses:
                # Analyses are dicts; root key is 'root', value like 'ك.ت.ب' or 'NOAN'
                root = analyses[0].get('root') if isinstance(analyses[0], dict) else getattr(analyses[0], 'root', None)
                if root and root != 'NOAN':
                    return root.replace('.', '')  # strip dot-notation separators
        except Exception:
            pass

    # Tier 2: Pattern-based extraction
    root = _tier2_pattern(word)
    if root:
        return root

    # Tier 3: Farasa stem approximation
    if _init_farasa():
        try:
            stem = _farasa_segmenter.stem(word)
            if stem:
                arabic_only = re.sub(r'[^؀-ۿ]', '', stem)
                if len(arabic_only) >= 3:
                    return arabic_only
        except Exception:
            pass

    # Tier 4: Abstain
    return None


def extract_root_cached(word: str, cache_path: Path | None = None,
                        camel_preset: str = "calima-msa-r13") -> str | None:
    """
    Like extract_root but optionally persists an on-disk cache for startup speed.
    The disk cache is best-effort — failures are silently ignored.
    """
    result = extract_root(word, camel_preset)

    if cache_path is not None and result is not None:
        try:
            import json
            cache: dict = {}
            if cache_path.exists():
                cache = json.loads(cache_path.read_text(encoding='utf-8'))
            cache[word] = result
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
        except Exception:
            pass

    return result
