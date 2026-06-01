"""Tesseract Arabic backend — fallback OCR engine."""

from __future__ import annotations

import shutil

import numpy as np

from ocr_engine.base import OCRBackend
from ocr_engine.schema import OCRResult, WordToken
from utils.logging import get_logger

log = get_logger(__name__)

_tesseract_available = False
_tesseract_ara_available = False

try:
    import pytesseract as _pytesseract  # noqa: F401
    if shutil.which("tesseract"):
        _tesseract_available = True
        try:
            langs = _pytesseract.get_languages(config="")
            _tesseract_ara_available = "ara" in langs
        except Exception:
            _tesseract_ara_available = False
    if not _tesseract_available:
        log.warning("tesseract binary not found; Tesseract backend disabled")
    elif not _tesseract_ara_available:
        log.warning("tesseract-ocr-ara language pack not found; Tesseract backend disabled")
except ImportError:
    log.warning("pytesseract not installed; Tesseract backend disabled")


class TesseractBackend(OCRBackend):
    name = "tesseract"

    def __init__(self, config=None):
        self._lang = "ara"
        self._psm_config = "--psm 6"
        if config is not None:
            t = getattr(getattr(config, "ocr", None), "tesseract", None)
            if t is not None:
                self._lang = getattr(t, "lang", self._lang)
                self._psm_config = getattr(t, "config", self._psm_config)

    @classmethod
    def is_available(cls) -> bool:
        return _tesseract_available and _tesseract_ara_available

    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        if not _tesseract_available:
            raise RuntimeError(
                "tesseract-ocr binary not found. "
                "Install with: apt-get install tesseract-ocr"
            )
        if not _tesseract_ara_available:
            raise RuntimeError(
                "tesseract-ocr-ara language pack not found. "
                "Install with: apt-get install tesseract-ocr-ara"
            )

        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(
            image,
            lang=self._lang,
            config=self._psm_config,
            output_type=Output.DICT,
        )

        words: list[WordToken] = []
        texts: list[str] = []
        n = len(data["text"])

        for i in range(n):
            conf_raw = data["conf"][i]
            text = str(data["text"][i]).strip()
            if conf_raw == -1 or not text:  # block/paragraph rows have conf=-1
                continue

            conf = max(0.0, min(1.0, float(conf_raw) / 100.0))
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            words.append(WordToken(
                text=text,
                confidence=conf,
                bbox=(x, y, w, h),
                page_index=page_index,
                source="tesseract",
            ))
            texts.append(text)

        page_conf = float(np.mean([w.confidence for w in words])) if words else 0.0
        log.debug(f"tesseract extract page={page_index} tokens={len(words)} conf={page_conf:.3f}")

        return OCRResult(
            text=" ".join(texts),
            words=words,
            confidence=page_conf,
            page_index=page_index,
            source="tesseract",
            raw={"tesseract_data": {k: data[k] for k in ("text", "conf", "left", "top")}},
        )
