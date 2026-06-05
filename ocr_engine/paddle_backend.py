"""PaddleOCR Arabic backend — primary OCR engine (PaddleOCR 3.x API).

Multi-hypothesis note: PaddleOCR 3.x does not expose n-best alternative
readings from its predict() API (only top-1 per line). Alternatives are
generated via classical Arabic visual confusion pairs at edit-distance-1.
Alternatives are stored in OCRResult.raw["paddle_alternatives"] and are
NOT mixed into the words list.
"""

from __future__ import annotations

import numpy as np

from ocr_engine.base import OCRBackend
from ocr_engine.schema import OCRResult, WordToken
from utils.logging import get_logger

log = get_logger(__name__)

# Visual confusion pairs common in classical Arabic manuscripts.
# Each tuple (a, b) means a and b are visually similar; substitution goes both ways.
ARABIC_CONFUSION_PAIRS: list[tuple[str, str]] = [
    ("ب", "ت"), ("ب", "ث"), ("ب", "ن"), ("ب", "ي"),
    ("ج", "ح"), ("ج", "خ"),
    ("ف", "ق"),
    ("ص", "ض"),
    ("ط", "ظ"),
    ("د", "ذ"), ("ر", "ز"),
]

# Build a fast lookup: char → list of confused chars
_CONFUSION_MAP: dict[str, list[str]] = {}
for _a, _b in ARABIC_CONFUSION_PAIRS:
    _CONFUSION_MAP.setdefault(_a, []).append(_b)
    _CONFUSION_MAP.setdefault(_b, []).append(_a)


def _generate_alternatives(
    text: str,
    primary_conf: float,
    top_n: int = 3,
) -> list[dict]:
    """Generate up to top_n edit-distance-1 alternatives via confusion pairs.

    Returns list of {text, confidence, source} dicts. Confidence is scaled
    at primary_conf * 0.85 (below primary but non-trivial signal).
    """
    alt_conf = round(primary_conf * 0.85, 4)
    seen: set[str] = {text}
    alts: list[dict] = []

    for i, ch in enumerate(text):
        if ch not in _CONFUSION_MAP:
            continue
        for confused in _CONFUSION_MAP[ch]:
            candidate = text[:i] + confused + text[i + 1:]
            if candidate not in seen:
                seen.add(candidate)
                alts.append({"text": candidate, "confidence": alt_conf, "source": "paddle_alt"})
            if len(alts) >= top_n:
                return alts

    return alts


_paddle_available = False
try:
    from paddleocr import PaddleOCR as _PaddleOCR  # noqa: F401
    _paddle_available = True
except ImportError:
    log.warning("paddleocr not installed; PaddleOCR backend disabled")

# Module-level cached model instance; None = not yet initialised.
_model: object | None = None
_model_init_failed: bool = False


def _get_model(lang: str = "ar", model_dir: str | None = None) -> object | None:
    """Return the cached PaddleOCR model instance, initialising on first call.

    PaddleOCR 3.x only accepts lang= (and optional model dir overrides).
    All other constructor arguments (use_gpu, device, use_textline_orientation,
    show_log, use_angle_cls) raise ValueError: Unknown argument in 3.x.

    Returns None and sets _model_init_failed if initialisation fails.
    """
    global _model, _model_init_failed
    if _model_init_failed:
        return None
    if _model is not None:
        return _model
    try:
        from paddleocr import PaddleOCR

        kwargs: dict = {"lang": lang}

        if model_dir:
            # Pre-supplied models: skip lang-based network download.
            # Convention: model_dir/{det,rec} holding PaddleX inference models.
            kwargs["text_detection_model_dir"] = model_dir
            kwargs["text_recognition_model_dir"] = model_dir
            del kwargs["lang"]

        _model = PaddleOCR(**kwargs)
        log.debug("PaddleOCR model loaded")
    except Exception as exc:
        log.warning(
            f"PaddleOCR model init failed — backend disabled. "
            f"Cause: {type(exc).__name__}: {exc}. "
            f"To fix: ensure network access to a model host, or set "
            f"config.ocr.paddle.model_dir to a local model directory."
        )
        _model_init_failed = True
        _model = None
    return _model


class PaddleBackend(OCRBackend):
    name = "paddle"

    def __init__(self, config=None):
        self._config = config
        self._lang = "ar"
        self._model_dir: str | None = None
        if config is not None:
            p = getattr(getattr(config, "ocr", None), "paddle", None)
            if p is not None:
                self._lang = getattr(p, "lang", self._lang)
                self._model_dir = getattr(p, "model_dir", None)

    @classmethod
    def is_available(cls) -> bool:
        """Return True only if PaddleOCR is installed AND the model can init."""
        if not _paddle_available:
            return False
        try:
            from paddleocr import PaddleOCR
            PaddleOCR(lang="ar")
            return True
        except Exception:
            return False

    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        if not _paddle_available:
            raise RuntimeError("PaddleOCR is not installed")

        model = _get_model(lang=self._lang, model_dir=self._model_dir)
        if model is None:
            raise RuntimeError(
                "PaddleOCR model unavailable (init failed — check logs for details)"
            )

        # PaddleOCR 3.x: predict() returns a generator; list() materialises it.
        # result[0] is the single-image dict with keys:
        #   rec_texts  — list[str]           recognised text per line
        #   rec_scores — list[float]         confidence per line
        #   rec_polys  — list[ndarray(4,2)]  4-corner polygon per line
        raw_list = list(model.predict(image))
        data = raw_list[0] if raw_list else {}

        rec_texts: list[str] = data.get("rec_texts", []) if data else []
        rec_scores: list[float] = data.get("rec_scores", []) if data else []
        rec_polys = data.get("rec_polys", []) if data else []

        words: list[WordToken] = []
        texts: list[str] = []

        for idx, text in enumerate(rec_texts):
            conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
            poly = rec_polys[idx] if idx < len(rec_polys) else None
            bbox = _polys_to_xywh(poly) if poly is not None else (0, 0, 0, 0)
            words.append(WordToken(
                text=str(text),
                confidence=conf,
                bbox=bbox,
                page_index=page_index,
                source="paddle",
            ))
            texts.append(str(text))

        page_conf = float(np.mean([w.confidence for w in words])) if words else 0.0
        log.debug(f"paddle extract page={page_index} tokens={len(words)} conf={page_conf:.3f}")

        # Multi-hypothesis: generate confusion-pair alternatives for each token.
        # PaddleOCR 3.x predict() exposes only top-1 — alternatives are edit-distance-1
        # substitutions via ARABIC_CONFUSION_PAIRS. Stored in raw, NOT in words.
        top_n = 3
        try:
            paddle_cfg = getattr(getattr(self._config, "ocr", None), "paddle", None)
            if paddle_cfg:
                top_n = getattr(paddle_cfg, "top_n_candidates", top_n)
        except Exception:
            pass

        paddle_alternatives: dict[str, list[dict]] = {}
        for w in words:
            alts = _generate_alternatives(w.text, w.confidence, top_n)
            if alts:
                paddle_alternatives[w.text] = alts

        return OCRResult(
            text=" ".join(texts),
            words=words,
            confidence=page_conf,
            page_index=page_index,
            source="paddle",
            raw={"paddle_raw": raw_list, "paddle_alternatives": paddle_alternatives},
        )


def _polys_to_xywh(polys) -> tuple[int, int, int, int]:
    """Convert PaddleOCR 3.x 4-corner polygon to (x, y, w, h) top-left bbox."""
    try:
        pts = np.array(polys)
        xs = pts[:, 0].astype(int)
        ys = pts[:, 1].astype(int)
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x, int(ys.max()) - y
        return (x, y, w, h)
    except Exception:
        return (0, 0, 0, 0)
