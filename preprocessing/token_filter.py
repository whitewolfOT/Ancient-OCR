"""Filter OCR word tokens that are glyph fragments or non-Arabic noise."""

from __future__ import annotations

import re

# Arabic consonants: basic (U+0621–U+063A) + extended (U+0641–U+064A)
_ARABIC_CONSONANTS = re.compile(r"[ء-غف-ي]")

# Arabic diacritics / tashkeel: U+064B–U+065F (fathatan … small high seen),
# U+0670 (superscript alef used as maddah mark)
_ARABIC_DIACRITICS = re.compile(r"[ً-ٰٟ]")


def is_noise_token(text: str) -> bool:
    """Return True if *text* is a glyph fragment or contains no Arabic script.

    Two rules (applied in order):
      1. Zero Arabic consonants  → noise (pure digits, punctuation, spaces, …)
      2. Exactly one consonant AND any diacritic mark → glyph fragment
         (e.g. نْ  or  وً).  Single-letter stopwords like و / ف / ل pass
         because they carry no diacritic.
    """
    if not text or not text.strip():
        return True
    consonants = _ARABIC_CONSONANTS.findall(text)
    n_cons = len(consonants)
    if n_cons == 0:
        return True
    if n_cons == 1 and _ARABIC_DIACRITICS.search(text):
        return True
    return False


def filter_noise_tokens(tokens: list) -> tuple[list, list]:
    """Partition *tokens* into (kept, discarded) based on :func:`is_noise_token`.

    Args:
        tokens: List of objects with a ``text`` attribute (WordToken or similar).

    Returns:
        ``(kept, discarded)`` — two lists of the same token objects.
    """
    kept: list = []
    discarded: list = []
    for tok in tokens:
        if is_noise_token(tok.text):
            discarded.append(tok)
        else:
            kept.append(tok)
    return kept, discarded
