"""TrOCR backend — optional line-level verifier for low-confidence regions."""

from __future__ import annotations

import numpy as np

from ocr_engine.base import OCRBackend
from ocr_engine.schema import OCRResult, WordToken
from utils.logging import get_logger

log = get_logger(__name__)

_trocr_available = False
try:
    import torch as _torch  # noqa: F401
    import transformers as _transformers  # noqa: F401
    _trocr_available = True
except ImportError:
    log.warning("torch/transformers not installed; TrOCR backend disabled")

_processor = None
_model = None


def _get_model(model_id: str):
    global _processor, _model
    if _processor is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        log.debug(f"loading TrOCR model model_id={model_id}")
        _processor = TrOCRProcessor.from_pretrained(model_id)
        _model = VisionEncoderDecoderModel.from_pretrained(model_id)
        _model.eval()
    return _processor, _model


class TrOCRBackend(OCRBackend):
    name = "trocr"

    def __init__(self, config=None):
        self._model_id: str | None = None
        self._conf_threshold = 0.5
        if config is not None:
            t = getattr(getattr(config, "ocr", None), "trocr", None)
            if t is not None:
                self._model_id = getattr(t, "model_id", None)
                self._conf_threshold = getattr(t, "conf_threshold", 0.5)

    @classmethod
    def is_available(cls) -> bool:
        return _trocr_available

    def is_ready(self) -> bool:
        """Ready only if available AND a model_id is configured."""
        return _trocr_available and bool(self._model_id)

    def recognize_line(self, crop: np.ndarray) -> tuple[str, float]:
        """Recognize a single line crop. Returns (text, confidence)."""
        if not self.is_ready():
            return ("", 0.0)
        try:
            import torch
            from PIL import Image as PILImage

            processor, model = _get_model(self._model_id)
            pil = PILImage.fromarray(crop[:, :, ::-1])  # BGR → RGB
            pixel_values = processor(images=pil, return_tensors="pt").pixel_values

            with torch.no_grad():
                generated = model.generate(pixel_values)

            text = processor.batch_decode(generated, skip_special_tokens=True)[0]
            # TrOCR doesn't expose per-token confidence; use a fixed heuristic
            conf = 0.75 if text.strip() else 0.0
            return (text.strip(), conf)
        except Exception as exc:
            log.warning(f"trocr recognize_line failed: {exc}")
            return ("", 0.0)

    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        """Coarse extract: treat the full image as one line."""
        text, conf = self.recognize_line(image)
        words = []
        if text:
            h, w = image.shape[:2]
            words = [WordToken(
                text=text, confidence=conf,
                bbox=(0, 0, w, h),
                page_index=page_index,
                source="trocr",
            )]
        return OCRResult(
            text=text, words=words, confidence=conf,
            page_index=page_index, source="trocr",
        )
