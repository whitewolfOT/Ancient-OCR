"""Load PDF/image files into page images for the OCR pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np

from utils.logging import get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


class PageImage(TypedDict):
    image: np.ndarray          # BGR uint8, OpenCV convention
    page_index: int
    dpi: int
    source_path: str
    page_count: int


def load_document(path: str | Path, dpi: int | None = None) -> list[PageImage]:
    """Load a PDF or image file and return one PageImage per page.

    Args:
        path: Path to a PDF, PNG, JPG, or TIFF file.
        dpi:  Render DPI for PDFs; falls back to config.preprocessing.dpi (default 300).

    Returns:
        List of PageImage dicts, one per page, in reading order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file extension is not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if dpi is None:
        try:
            from utils.config import get_config
            dpi = get_config().preprocessing.dpi
        except Exception:
            dpi = 300

    if ext == ".pdf":
        return _load_pdf(path, dpi)
    else:
        return _load_image(path, dpi)


def _load_pdf(path: Path, dpi: int) -> list[PageImage]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to load PDFs. "
            "Install it with: pip install pymupdf"
        ) from exc

    doc = fitz.open(str(path))
    page_count = len(doc)
    pages: list[PageImage] = []

    zoom = dpi / 72.0  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
        # PyMuPDF returns RGB; convert to BGR for OpenCV compatibility
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        bgr = rgb[:, :, ::-1].copy()
        pages.append(PageImage(
            image=bgr,
            page_index=i,
            dpi=dpi,
            source_path=str(path),
            page_count=page_count,
        ))
        log.debug(f"loaded pdf page={i} shape={bgr.shape}")

    doc.close()
    return pages


def _load_image(path: Path, dpi: int) -> list[PageImage]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to load image files. "
            "Install it with: pip install pillow"
        ) from exc

    img = Image.open(path).convert("RGB")

    # Read actual DPI from EXIF/image metadata; fall back to config value.
    actual_dpi = dpi
    try:
        info_dpi = img.info.get("dpi")
        if info_dpi and len(info_dpi) >= 2 and info_dpi[0] > 0 and info_dpi[1] > 0:
            actual_dpi = int(min(info_dpi[0], info_dpi[1]))
    except Exception:
        pass

    rgb = np.array(img, dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()

    log.debug(f"loaded image path={path} shape={bgr.shape} dpi={actual_dpi}")
    return [PageImage(
        image=bgr,
        page_index=0,
        dpi=actual_dpi,
        source_path=str(path),
        page_count=1,
    )]


def classify_page(image: np.ndarray) -> dict:
    """Return heuristic quality flags for a page image.

    These are informational metadata — not filtering gates.
    """
    h, w = image.shape[:2]
    pixel_count = h * w

    # Low-res heuristic: fewer than 300×300 effective pixels
    is_low_res = pixel_count < 90_000

    # Skew heuristic: delegated to preprocessing; flag as unknown here
    likely_skewed = False

    # Multi-column: wide aspect ratio suggests columns
    multi_column = (w / max(h, 1)) > 1.8

    # Text density: fraction of dark pixels (rough proxy)
    try:
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_pixels = int((binary == 0).sum())
        text_density = round(dark_pixels / max(pixel_count, 1), 4)
    except Exception:
        text_density = 0.0

    return {
        "is_low_res": is_low_res,
        "likely_skewed": likely_skewed,
        "multi_column": multi_column,
        "text_density": text_density,
    }
