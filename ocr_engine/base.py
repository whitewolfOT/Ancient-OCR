"""Abstract base class for OCR backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ocr_engine.schema import OCRResult


class OCRBackend(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        """Run OCR on a single image and return a structured result."""

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this backend's dependencies are installed and ready."""
        return False
