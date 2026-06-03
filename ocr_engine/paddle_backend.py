"""PaddleOCR Arabic backend — primary OCR engine (PaddleOCR 3.x API)."""

from __future__ import annotations

import numpy as np

from ocr_engine.base import OCRBackend
from ocr_engine.schema import OCRResult, WordToken
from utils.logging import get_logger

log = get_logger(__name__)

_paddle_available = False
try:
    from paddleocr import PaddleOCR as _PaddleOCR  # noqa: F401
    _paddle_available = True
except ImportError:
    log.warning("paddleocr not installed; PaddleOCR backend disabled")

# Module-level cached model instance; None = not yet initialised.
# _model_init_failed = True means the last init attempt raised an exception
# (network unavailable, 403 from model servers, missing weights, etc.).
_model: object | None = None
_model_init_failed: bool = False


def _get_model(
    lang: str = "ar",
    device: str = "cpu",
    use_textline_orientation: bool = True,
    model_dir: str | None = None,
) -> object | None:
    """Return the cached PaddleOCR model instance, initialising on first call.

    Returns None and sets ``_model_init_failed`` if initialisation fails for
    any reason (missing models, no network access, 403 from model servers).
    """
    global _model, _model_init_failed
    if _model_init_failed:
        return None
    if _model is not None:
        return _model
    try:
        from paddleocr import PaddleOCR

        kwargs: dict = dict(
            lang=lang,
            # PaddleOCR 3.x: device replaces use_gpu. Passed via **kwargs →
            # parse_common_args → create_pipeline(device=...).
            device=device,
            use_textline_orientation=use_textline_orientation,
        )

        if model_dir:
            # Pre-supplied models: skip the lang-based network download by
            # pointing directly at local inference model directories.
            # Convention: model_dir/{det,rec} holding PaddleX inference models.
            # Users can also set text_detection_model_dir /
            # text_recognition_model_dir independently via future config keys.
            kwargs["text_detection_model_dir"] = model_dir
            kwargs["text_recognition_model_dir"] = model_dir
            kwargs.pop("lang", None)

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
        self._lang = "ar"
        self._device = "cpu"
        self._use_textline_orientation = True
        self._model_dir: str | None = None
        if config is not None:
            p = getattr(getattr(config, "ocr", None), "paddle", None)
            if p is not None:
                self._lang = getattr(p, "lang", self._lang)
                # use_gpu (legacy config key) → device string
                use_gpu = getattr(p, "use_gpu", False)
                self._device = "gpu" if use_gpu else "cpu"
                self._model_dir = getattr(p, "model_dir", None)

    @classmethod
    def is_available(cls) -> bool:
        """Return True only if PaddleOCR is installed AND the model can init.

        The first call attempts model initialisation with default parameters.
        If it fails (network blocked, models absent), the failure is cached and
        this method returns False for all subsequent calls in the same process.
        """
        if not _paddle_available:
            return False
        model = _get_model()
        return model is not None

    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        if not _paddle_available:
            raise RuntimeError("PaddleOCR is not installed")

        model = _get_model(
            lang=self._lang,
            device=self._device,
            use_textline_orientation=self._use_textline_orientation,
            model_dir=self._model_dir,
        )
        if model is None:
            raise RuntimeError(
                "PaddleOCR model unavailable (init failed — check logs for details)"
            )

        # PaddleOCR 3.x: predict() returns a list of result objects, one per
        # input image. Each result is dict-like with keys:
        #   rec_texts  — list[str]             recognised text per line
        #   rec_scores — list[float]           confidence per line
        #   rec_polys  — list[array(4×2)]      4-corner polygon per line
        #   rec_boxes  — list[array(4,)]       axis-aligned bbox per line
        raw = model.predict(image)

        words: list[WordToken] = []
        texts: list[str] = []

        for page_result in (raw or []):
            rec_texts = page_result.get("rec_texts", []) if page_result else []
            rec_scores = page_result.get("rec_scores", []) if page_result else []
            rec_polys = page_result.get("rec_polys", []) if page_result else []

            for idx, text in enumerate(rec_texts):
                conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                polys = rec_polys[idx] if idx < len(rec_polys) else None
                bbox = _polys_to_xywh(polys) if polys is not None else (0, 0, 0, 0)
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

        return OCRResult(
            text=" ".join(texts),
            words=words,
            confidence=page_conf,
            page_index=page_index,
            source="paddle",
            raw={"paddle_raw": raw},
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
