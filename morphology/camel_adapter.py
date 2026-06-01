"""CAMeL Tools morphology adapter with rule-based fallback."""

from __future__ import annotations

from utils.logging import get_logger

log = get_logger(__name__)

_camel_available = False
_analyzer = None

try:
    from camel_tools.morphology.database import MorphologyDB as _MorphologyDB  # noqa: F401
    from camel_tools.morphology.analyzer import Analyzer as _Analyzer  # noqa: F401
    _camel_available = True
except ImportError:
    log.warning("camel-tools not installed; morphology will use rule-based fallback")


def _load_analyzer():
    global _analyzer
    if _analyzer is not None:
        return _analyzer
    if not _camel_available:
        return None
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        db = MorphologyDB.builtin_db()
        _analyzer = Analyzer(db)
        log.debug("CAMeL Tools analyzer loaded")
        return _analyzer
    except Exception as exc:
        log.warning(
            f"CAMeL Tools analyzer failed to load: {exc}. "
            "If the DB is missing, run: camel_data -i morphology-db-msa-r13. "
            "Falling back to rule-based morphology."
        )
        return None


def analyze(word: str) -> dict | None:
    """Analyze a word with CAMeL Tools.

    Returns a dict with keys: {lemma, root, pos, pattern, features}
    or None if CAMeL Tools is unavailable or analysis fails.
    """
    analyzer = _load_analyzer()
    if analyzer is None:
        return None

    try:
        analyses = analyzer.analyze(word)
        if not analyses:
            return None

        best = analyses[0]
        return {
            "lemma": best.get("lex", word),
            "root": best.get("root", None),
            "pos": best.get("pos", None),
            "pattern": best.get("pattern", None),
            "features": {k: v for k, v in best.items() if k not in ("lex", "root", "pos", "pattern")},
        }
    except Exception as exc:
        log.warning(f"camel analyze failed for {word!r}: {exc}")
        return None


def is_available() -> bool:
    return _camel_available
