"""PaddleOCR Arabic backend — primary OCR engine."""

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

_model = None  # lazy-loaded on first extract()


def _get_model(lang: str = "ar", use_gpu: bool = False):
    global _model
    if _model is None:
        from paddleocr import PaddleOCR
        _model = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)
        log.debug("PaddleOCR model loaded")
    return _model


class PaddleBackend(OCRBackend):
    name = "paddle"

    def __init__(self, config=None):
        self._lang = "ar"
        self._use_gpu = False
        if config is not None:
            p = getattr(getattr(config, "ocr", None), "paddle", None)
            if p is not None:
                self._lang = getattr(p, "lang", self._lang)
                self._use_gpu = getattr(p, "use_gpu", self._use_gpu)

    @classmethod
    def is_available(cls) -> bool:
        return _paddle_available

    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        if not _paddle_available:
            raise RuntimeError("PaddleOCR is not installed")

        model = _get_model(self._lang, self._use_gpu)
        raw = model.ocr(image, cls=True)

        words: list[WordToken] = []
        texts: list[str] = []

        # PaddleOCR returns [[[bbox_points, [text, conf]], ...], ...]
        pages = raw if raw else []
        for page_result in pages:
            if not page_result:
                continue
            for item in page_result:
                bbox_points, (text, conf) = item
                bbox = _points_to_xywh(bbox_points)
                words.append(WordToken(
                    text=str(text),
                    confidence=float(conf),
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


def _points_to_xywh(points) -> tuple[int, int, int, int]:
    """Convert PaddleOCR 4-corner [[x,y],...] to (x, y, w, h) top-left."""
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return (x, y, w, h)
