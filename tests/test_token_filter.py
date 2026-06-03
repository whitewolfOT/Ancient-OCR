"""Tests for preprocessing.token_filter."""

import pytest
from preprocessing.token_filter import is_noise_token, filter_noise_tokens


# ---------------------------------------------------------------------------
# is_noise_token — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    # Empty / blank → noise
    ("",     True),
    (" ",    True),
    # Pure punctuation / digits → noise
    ("123",  True),
    (".",    True),
    (",",    True),
    # Isolated diacritics (0 consonants) → noise
    ("ً",    True),   # fathatan
    ("ّ",    True),   # shadda
    # Glyph fragment: 1 consonant + diacritic → noise
    ("نْ",  True),   # nun + sukun
    ("وً",  True),   # waw + fathatan
    ("بَ",  True),   # ba + fatha
    # Single-letter stopwords (1 consonant, NO diacritic) → KEEP
    ("و",   False),   # waw — valid stopword
    ("ف",   False),   # fa  — valid stopword
    ("ل",   False),   # lam — valid stopword
    # Normal words → KEEP
    ("كتاب", False),
    ("الله", False),
    ("في",   False),
    # Mixed Arabic + digits → KEEP (has consonants)
    ("ص١٢",  False),
])
def test_is_noise_token(text, expected):
    assert is_noise_token(text) is expected


# ---------------------------------------------------------------------------
# filter_noise_tokens — integration
# ---------------------------------------------------------------------------

class _FakeTok:
    def __init__(self, text):
        self.text = text


def test_filter_splits_correctly():
    tokens = [_FakeTok(t) for t in ["كتاب", "نْ", "و", "ً", "في"]]
    kept, discarded = filter_noise_tokens(tokens)
    assert {t.text for t in kept} == {"كتاب", "و", "في"}
    assert {t.text for t in discarded} == {"نْ", "ً"}


def test_filter_all_kept():
    tokens = [_FakeTok(t) for t in ["كتاب", "رجل", "بيت"]]
    kept, discarded = filter_noise_tokens(tokens)
    assert len(kept) == 3
    assert len(discarded) == 0


def test_filter_empty_list():
    kept, discarded = filter_noise_tokens([])
    assert kept == []
    assert discarded == []
