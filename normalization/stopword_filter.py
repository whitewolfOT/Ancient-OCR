"""Tier-1 token gate: Arabic function-word (stopword) filter.

Called after basic normalization, before morphology and lexicon lookup.
If a token is in the stopword set it is accepted immediately with
reason_code="stopword" — morphology and the full lexicon pipeline are
never reached.

The canonical word list lives in config.yaml under lexicon.stopwords so it
can be extended per-project without code changes. All words are normalized
at load time using the same rules as the main pipeline, so matching is
always exact against the normalized token.
"""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Built-in ~500-word list, organized by category.
# Stored as pre-normalized Arabic (alef variants already collapsed, no
# diacritics, ى → ي).  This set is the fallback; config.yaml can extend it.
# ---------------------------------------------------------------------------

_BUILTIN: frozenset[str] = frozenset({

    # ── Prepositions ──────────────────────────────────────────────────────
    "في", "من", "الي", "علي", "عن", "مع", "ب", "ك", "ل",
    "منذ", "حتي", "خلال", "رغم", "نحو", "دون", "سوي", "غير", "الا",
    "ضد", "ازاء", "حيال", "تجاه", "حول", "بين", "عند", "لدي",
    "امام", "خلف", "فوق", "تحت", "قبل", "بعد", "وراء", "اثر",
    "بجانب", "بجوار", "مقابل", "قرب", "لدن", "ازاء",
    # with pronoun suffixes (commonly appear as tokens after tokenisation)
    "منه", "منها", "منهم", "منهن", "منهما", "منك", "مني", "منا",
    "فيه", "فيها", "فيهم", "فيهن", "فيهما", "فيك", "في", "فينا",
    "عليه", "عليها", "عليهم", "عليهن", "عليهما", "عليك", "علينا",
    "به", "بها", "بهم", "بهن", "بهما", "بك", "بي", "بنا",
    "اليه", "اليها", "اليهم", "اليهن", "اليهما", "اليك", "الينا",
    "عنه", "عنها", "عنهم", "عنهن", "عنهما", "عنك", "عني", "عنا",
    "له", "لها", "لهم", "لهن", "لهما", "لك", "لي", "لنا",
    "معه", "معها", "معهم", "معهن", "معهما", "معك", "معي", "معنا",

    # ── Conjunctions ──────────────────────────────────────────────────────
    "و", "ف", "ثم", "او", "ام", "بل", "لكن", "لكن",
    "بينما", "حيث", "اذ", "حين", "عندما", "بعدما", "قبلما",
    "ريثما", "كلما", "لما", "فيما", "حال", "طالما", "مادام",

    # ── Detached pronouns ─────────────────────────────────────────────────
    "هو", "هي", "هم", "هن", "هما",
    "انت", "انتِ", "انتم", "انتن", "انتما",
    "انا", "نحن",
    "ذاته", "ذاتها", "ذواتهم", "نفسه", "نفسها", "انفسهم",
    "اياه", "اياها", "اياهم", "اياهن", "اياك", "اياكم", "اياي", "ايانا",

    # ── Demonstratives ────────────────────────────────────────────────────
    "هذا", "هذه", "هذان", "هاتان", "هؤلاء",
    "ذلك", "تلك", "ذانك", "تانك", "اولئك",
    "هنا", "هناك", "هنالك", "ثم", "هكذا", "كذلك", "كذا",
    "ذا", "ذي", "ذين", "تين", "هاهنا", "هاهناك",

    # ── Relative pronouns ─────────────────────────────────────────────────
    "الذي", "التي", "اللذان", "اللتان", "الذين",
    "اللواتي", "اللاتي", "اللتين", "اللذين",

    # ── Interrogatives ────────────────────────────────────────────────────
    "ماذا", "متي", "اين", "كيف", "كم", "هل",
    "هلا", "الا", "لماذا", "بماذا", "الي متي", "كيفما",

    # ── Conditional particles ─────────────────────────────────────────────
    "ان", "اذا", "لو", "لولا", "لوما",
    "مهما", "حيثما", "اينما", "كيفما",
    "انما", "اما",

    # ── Negation ──────────────────────────────────────────────────────────
    "لا", "لم", "لن", "ليس", "لات",
    "بلا", "بغير",
    "ليست", "لست", "لستِ", "لستم", "لستن", "لسنا", "ليسوا", "ليسا",
    "ما",  # negation use; context-ambiguous but high-frequency as function word

    # ── Verbal auxiliaries — كان conjugations ─────────────────────────────
    "كان", "كانت", "كانوا", "كانتا", "كانا",
    "كنت", "كنتِ", "كنتم", "كنتن", "كنتما", "كنا",
    "يكون", "تكون", "اكون", "نكون",
    "يكونوا", "يكونون", "تكوني", "تكونوا", "تكونين",

    # كان negated forms
    "لست", "لستِ", "لستم", "لستن", "لسنا",

    # أصبح / أمسى / صار / بات / ظل / غدا
    "اصبح", "اصبحت", "اصبحوا", "اصبحنا", "يصبح", "تصبح",
    "امسي", "امست", "امسوا",
    "صار", "صارت", "صاروا", "يصير", "تصير",
    "بات", "باتت", "باتوا", "يبيت",
    "ظل", "ظلت", "ظلوا", "يظل",
    "غدا", "غدت",
    "اضحي", "اضحت",

    # كاد / عسى / أوشك
    "كاد", "يكاد", "كادت", "اوشك", "يوشك", "عسي",

    # ── Discourse and modal particles ─────────────────────────────────────
    "قد", "سوف", "اذن", "اذا", "لما",
    "كي", "لكي", "فلما", "ايضا", "كذلك",
    "نعم", "بلي", "قط", "ابدا", "دائما", "احيانا",
    "عادة", "غالبا", "نادرا", "حقا", "فعلا",
    "هيا", "يا", "ايها", "ايتها",

    # ── Classical / Quranic particles ─────────────────────────────────────
    "ان",    # إنّ — emphatic
    "انه", "انها", "انهم",
    "كان",  # already above
    "كانما", "انما",
    "ليت", "لعل",
    "لكن",
    "اما", "وامّا",
    "واذ", "واذا", "فاذا", "فلما", "وقد", "وكان", "وكانت",
    "فان", "فانه", "فانها",
    "اي",    # "أي" meaning "that is" (discourse)
    "اعني", "يعني", "يقال", "قيل",
    "فصل", "باب", "فان", "فاذا", "فمن", "فلا", "فلم", "فلن",

    # ── Numerals 1–10 (both genders) and ordinals ─────────────────────────
    "واحد", "واحدة",
    "اثنان", "اثنتان", "اثنين", "اثنتين",
    "ثلاثة", "ثلاث",
    "اربعة", "اربع",
    "خمسة", "خمس",
    "ستة", "ست",
    "سبعة", "سبع",
    "ثمانية", "ثماني",
    "تسعة", "تسع",
    "عشرة", "عشر",
    # Ordinals
    "الاول", "الثاني", "الثالث", "الرابع", "الخامس",
    "السادس", "السابع", "الثامن", "التاسع", "العاشر",
    "الاولي", "الثانية", "الثالثة", "الرابعة", "الخامسة",
    "السادسة", "السابعة", "الثامنة", "التاسعة", "العاشرة",
    # Isolated numeral forms that appear as tokens
    "احد", "احدي", "احدي عشر", "اثني عشر",

    # ── Quantifiers (high-frequency, grammatically functional) ────────────
    "كل", "بعض", "جميع", "كافة", "سائر", "جل", "اغلب",
    "مختلف", "عامة", "كلا", "كلتا",

    # ── Temporal adverbs ──────────────────────────────────────────────────
    "الان", "اليوم", "امس", "غدا", "مرة", "مرات",
    "انئذ", "حينئذ", "يومئذ", "ساعتئذ",

    # ── Degree / modal adverbs ─────────────────────────────────────────────
    "جدا", "تماما", "فقط", "مثل", "كمثل", "مثلما",

    # ── Common clitics and high-frequency bound forms ──────────────────────
    # These appear after tokenisation as standalone tokens
    "ال",    # definite article clitic
    "و",     # waaw clitic
    "ف",     # fa clitic
    "ب",     # ba clitic
    "ل",     # lam clitic
    "ك",     # ka clitic (as)
})


# Module-level cache — built once per config hash, never rebuilt during a run
_cache: dict[str, frozenset[str]] = {}


def _build_set(config=None) -> frozenset[str]:
    """Normalise the canonical list + any config extras, cache the result."""
    from normalization.arabic_normalizer import normalize_text as _norm
    from utils.cache import config_hash

    cache_key = config_hash()
    if cache_key in _cache:
        return _cache[cache_key]

    words: set[str] = set()
    for w in _BUILTIN:
        normalized, _ = _norm(w)
        words.add(normalized)

    # Load per-project extras from config
    if config is not None:
        try:
            extras = config.lexicon.stopwords or []
            for w in extras:
                normalized, _ = _norm(str(w))
                words.add(normalized)
        except AttributeError:
            pass

    result = frozenset(words)
    _cache[cache_key] = result
    log.debug(f"stopword_filter: built set size={len(result)}")
    return result


def is_stopword(normalized_text: str, config=None) -> bool:
    """Return True if *normalized_text* is a function word.

    The input must already be normalized by arabic_normalizer.normalize_text
    before calling this — that is always the case in the pipeline.
    """
    return normalized_text in _build_set(config)


def stopword_count(config=None) -> int:
    """Return the current size of the stopword set (for diagnostics)."""
    return len(_build_set(config))
