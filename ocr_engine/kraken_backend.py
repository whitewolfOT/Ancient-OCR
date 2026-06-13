"""Kraken HTR backend — profile-driven, optional.

Binarizer (nlbin/sauvola/otsu) is this backend's responsibility.
General preprocessing (brightness, contrast, gamma, etc.) runs in
image_pipeline.py before this backend is called.
Models download lazily inside process_image(), never in __init__().
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ocr_engine.base import OCRBackend
from ocr_engine.profile_loader import OCRProfile
from ocr_engine.schema import OCRResult, WordToken
from utils.logging import get_logger

log = get_logger(__name__)

_ARABIC_WORD_SEP = re.compile(r'[\s‌‍]+')  # space + zero-width joiners
_CC_GAP_PX  = 15   # horizontal gap threshold for word grouping (pixels)
_CC_MIN_AREA = 6   # minimum CC area to count as ink (pixels²)


class KrakenBackend(OCRBackend):
    name = "kraken"

    def __init__(self, config=None, profile: OCRProfile | None = None):
        self.config = config
        self.profile = profile or OCRProfile(name="default")
        self._seg_model = None   # lazy-loaded
        self._rec_model = None
        self._rec_secondary = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            import kraken  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_models(self) -> bool:
        """Download/load models if absent. Return False on failure."""
        if self._seg_model is not None and self._rec_model is not None:
            return True
        try:
            from kraken.lib import models as kraken_models

            model_dir = Path(
                getattr(getattr(self.config, "kraken", None), "model_dir", "models/kraken")
                if self.config else "models/kraken"
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            all_mlmodels = sorted(model_dir.glob("*.mlmodel"))
            seg_candidates = [p for p in all_mlmodels if "seg" in p.name.lower()]
            rec_candidates = [p for p in all_mlmodels if "rec" in p.name.lower()]

            # Segmentation model
            seg_handle = self.profile.seg_model
            seg_path = model_dir / f"seg_{seg_handle.replace(':', '_').replace('/', '_')}.mlmodel"
            if not seg_path.exists():
                # Try any staged seg model before downloading
                if seg_candidates:
                    seg_path = seg_candidates[0]
                    log.info(f"kraken: using staged seg model {seg_path.name}")
                else:
                    log.info(f"kraken: downloading seg model {seg_handle}")
                    try:
                        from htrmopo import get_model as htr_get_model
                        htr_get_model(seg_handle, path=str(model_dir))
                    except ImportError:
                        from kraken.repo import get_model  # kraken <5 fallback
                        get_model(seg_handle, str(model_dir))
                    candidates = sorted(model_dir.glob("*.mlmodel"))
                    seg_path = next((p for p in candidates if "seg" in p.name.lower()), candidates[-1] if candidates else seg_path)
            # Seg models use TorchVGSLModel.load_model(), not load_any()
            from kraken.lib import vgsl as kraken_vgsl
            self._seg_model = kraken_vgsl.TorchVGSLModel.load_model(str(seg_path))

            # Primary recognition model
            rec_handle = self.profile.rec_model
            rec_path = model_dir / f"rec_{rec_handle}.mlmodel"
            if not rec_path.exists():
                # Try any staged rec model before downloading
                if rec_candidates:
                    rec_path = rec_candidates[0]
                    log.info(f"kraken: using staged rec model {rec_path.name}")
                else:
                    log.info(f"kraken: downloading rec model {rec_handle}")
                    try:
                        from htrmopo import get_model as htr_get_model
                        htr_get_model(rec_handle, path=str(model_dir))
                    except ImportError:
                        from kraken.repo import get_model  # kraken <5 fallback
                        get_model(rec_handle, str(model_dir))
                    candidates = sorted(model_dir.glob("*.mlmodel"))
                    rec_path = next((p for p in candidates if "rec" in p.name.lower()), candidates[-1] if candidates else rec_path)
            self._rec_model = kraken_models.load_any(str(rec_path))

            # Secondary recognition model (optional, for N-best)
            if self.profile.rec_model_secondary and self.profile.n_best > 1:
                sec_handle = self.profile.rec_model_secondary
                sec_path = model_dir / f"rec_{sec_handle}.mlmodel"
                if not sec_path.exists():
                    alt = [p for p in rec_candidates if p != rec_path]
                    if alt:
                        sec_path = alt[0]
                if sec_path.exists():
                    self._rec_secondary = kraken_models.load_any(str(sec_path))

            return True
        except Exception as exc:
            log.warning(f"kraken: model init failed — {type(exc).__name__}: {exc}")
            return False

    def _binarize(self, img: np.ndarray):
        """Return binarized PIL Image."""
        from PIL import Image as PILImage
        binarizer = self.profile.binarizer
        if binarizer == "nlbin":
            # nlbin is Kraken-internal; only call inside this backend
            from kraken.binarization import nlbin
            pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img)
            return nlbin(pil)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if binarizer == "sauvola":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2,
            )
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return PILImage.fromarray(binary)

    @staticmethod
    def _baseline_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y = min(xs), min(ys)
        return (x, y, max(xs) - x, max(ys) - y)

    def extract(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        """Alias for process_image() to satisfy OCRBackend interface."""
        return self.process_image(image, page_index)

    def process_image(self, image: np.ndarray, page_index: int = 0) -> OCRResult:
        log.info(f"kraken: process_image start page={page_index} image={image.shape}")
        if not self.is_available():
            log.warning("kraken not installed; returning empty result")
            return self._empty_result(page_index)

        if not self._ensure_models():
            return self._empty_result(page_index)

        try:
            from kraken import blla, rpred

            # _binarize returns PIL Image; blla.segment and rpred both need PIL
            pil_img = self._binarize(image)

            text_dir = "horizontal-rl" if self.profile.rtl else "horizontal-lr"
            seg = blla.segment(
                pil_img,
                model=self._seg_model,
                text_direction=text_dir,
                device=self.profile.device or "cpu",
            )

            all_words: list[WordToken] = []
            line_texts: list[str] = []
            line_confidences: list[float] = []

            # Run recognition once over the full segmentation (one record per line)
            records = list(rpred.rpred(
                self._rec_model, pil_img, seg,
                bidi_reordering=self.profile.rtl,
            ))
            page_np_gray = np.array(pil_img.convert("L"))

            # Secondary model records (N-best signal)
            secondary_records = []
            if self._rec_secondary and self.profile.n_best > 1:
                try:
                    secondary_records = list(rpred.rpred(
                        self._rec_secondary, pil_img, seg,
                        bidi_reordering=self.profile.rtl,
                    ))
                except Exception as exc:
                    log.debug(f"kraken secondary rec failed: {exc}")

            sec_by_line = {}
            for i, sec_rec in enumerate(secondary_records):
                sec_by_line[i] = sec_rec

            for line_idx, (line, rec) in enumerate(zip(seg.lines, records)):
                line_id = str(id(line))
                sec_rec = sec_by_line.get(line_idx)
                sec_words = {w: c for w, c, _ in self._split_prediction(sec_rec)} if sec_rec else {}

                # Compute line crop for CC word detection
                line_crop_arr = None
                lcrop_x = lcrop_y = 0
                boundary = getattr(line, "boundary", None) or []
                if boundary:
                    bxs = [int(p[0]) for p in boundary]
                    bys = [int(p[1]) for p in boundary]
                    lx1 = max(0, min(bxs)); ly1 = max(0, min(bys))
                    lx2 = min(page_np_gray.shape[1], max(bxs))
                    ly2 = min(page_np_gray.shape[0], max(bys))
                    if lx2 > lx1 and ly2 > ly1:
                        line_crop_arr = page_np_gray[ly1:ly2, lx1:lx2]
                        lcrop_x, lcrop_y = lx1, ly1
                for word_text, span_confs, span_baseline in self._split_prediction(rec, line_crop_arr, lcrop_x, lcrop_y):
                    if not word_text.strip():
                        continue
                    conf = float(sum(span_confs) / len(span_confs)) if span_confs else 0.0
                    bbox = self._baseline_bbox(span_baseline) if span_baseline else (0, 0, 0, 0)
                    candidates = [{"text": word_text, "confidence": round(conf, 4)}]
                    if sec_rec and word_text in sec_words:
                        sec_conf = sec_words[word_text]
                        if sec_conf != conf:
                            candidates.append({"text": word_text, "confidence": round(sec_conf, 4)})
                    all_words.append(WordToken(
                        text=word_text,
                        confidence=conf,
                        bbox=bbox,
                        page_index=page_index,
                        source="kraken",
                        line_id=line_id,
                        baseline=span_baseline or None,
                        char_confidences=[round(float(c), 4) for c in span_confs] if span_confs else None,
                        candidates=candidates,
                    ))

                line_texts.append(rec.prediction)
                confs = rec.confidences
                line_confidences.append(
                    float(sum(confs) / len(confs)) if confs else 0.0
                )

            page_conf = float(sum(line_confidences) / len(line_confidences)) if line_confidences else 0.0
            log.info(f"kraken: process_image done page={page_index} lines={len(line_texts)} words={len(all_words)} conf={page_conf:.3f}")

            return OCRResult(
                text="\n".join(line_texts),
                words=all_words,
                confidence=page_conf,
                page_index=page_index,
                source="kraken",
                raw={
                    "seg_model": self.profile.seg_model,
                    "rec_model": self.profile.rec_model,
                    "n_best": self.profile.n_best,
                    "profile": self.profile.name,
                },
            )
        except Exception as exc:
            log.warning(f"kraken process_image failed: {type(exc).__name__}: {exc}")
            return self._empty_result(page_index)

    def _split_prediction(
        self,
        record,
        line_crop: "np.ndarray | None" = None,
        crop_x: int = 0,
        crop_y: int = 0,
    ) -> list[tuple[str, list[float], list[tuple[int, int]]]]:
        """Split a Kraken OCR record into (word_text, confidences, baseline_pts) triples.

        Kraken record has .prediction (str), .cuts (bbox/baseline list per char),
        .confidences (per-char float list). Split on Arabic word boundaries.
        """
        text = record.prediction
        cuts = getattr(record, "cuts", []) or []
        confs = list(getattr(record, "confidences", []) or [])

        if not text:
            return []

        if line_crop is not None and line_crop.size > 0:
            word_bboxes = self._words_from_cc(line_crop, crop_x, crop_y)
            if word_bboxes:
                return self._assign_chars_to_words(text, cuts, confs, word_bboxes)

        result = []
        char_idx = 0

        for word in _ARABIC_WORD_SEP.split(text):
            if not word:
                char_idx += 1  # skip separator char
                continue
            start = char_idx
            end = char_idx + len(word)

            word_confs = confs[start:end] if confs else []
            word_cuts = cuts[start:end] if cuts else []

            # Convert cuts to baseline points
            baseline_pts: list[tuple[int, int]] = []
            for cut in word_cuts:
                try:
                    if hasattr(cut, "__iter__"):
                        pts = list(cut)
                        if pts and hasattr(pts[0], "__iter__"):
                            # polygon/baseline: take first and last point
                            baseline_pts.append((int(pts[0][0]), int(pts[0][1])))
                            baseline_pts.append((int(pts[-1][0]), int(pts[-1][1])))
                        else:
                            # flat [x1,y1,x2,y2]
                            if len(pts) >= 4:
                                baseline_pts.append((int(pts[0]), int(pts[1])))
                                baseline_pts.append((int(pts[2]), int(pts[3])))
                except Exception:
                    pass

            result.append((word, word_confs, baseline_pts))
            char_idx = end + 1  # +1 for separator

        return result

    def _words_from_cc(
        self,
        crop: np.ndarray,
        crop_x: int,
        crop_y: int,
        gap_px: int = _CC_GAP_PX,
    ) -> list[tuple[int, int, int, int]]:
        """CC-based word segmentation. Returns (x,y,w,h) bboxes in PAGE coords, RTL order."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        if float(gray.mean()) > 127.0:
            gray = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comps = []
        for lbl in range(1, n):
            x, y, w, h, area = stats[lbl]
            if area >= _CC_MIN_AREA:
                comps.append((x, y, w, h))
        if not comps:
            return []
        comps.sort(key=lambda c: c[0])
        groups: list[list[tuple]] = [[comps[0]]]
        for comp in comps[1:]:
            cx = comp[0]
            right_of_group = max(c[0] + c[2] for c in groups[-1])
            if cx - right_of_group <= gap_px:
                groups[-1].append(comp)
            else:
                groups.append([comp])
        word_bboxes: list[tuple[int, int, int, int]] = []
        for grp in groups:
            gx1 = min(c[0] for c in grp)
            gy1 = min(c[1] for c in grp)
            gx2 = max(c[0] + c[2] for c in grp)
            gy2 = max(c[1] + c[3] for c in grp)
            word_bboxes.append((gx1 + crop_x, gy1 + crop_y, gx2 - gx1, gy2 - gy1))
        word_bboxes.sort(key=lambda b: -(b[0] + b[2]))  # RTL: right edge descending
        return word_bboxes

    def _assign_chars_to_words(
        self,
        text: str,
        cuts,
        confs: list[float],
        word_bboxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[str, list[float], list[tuple[int, int]]]]:
        """Assign characters to CC-derived word bboxes by cut x-position."""
        chars = list(text)

        def _cut_xcenter(cut) -> float | None:
            try:
                if hasattr(cut, "__iter__"):
                    pts = list(cut)
                    if pts and hasattr(pts[0], "__iter__"):
                        return float(sum(p[0] for p in pts) / len(pts))
                    if len(pts) >= 4:
                        return float((pts[0] + pts[2]) / 2)
            except Exception:
                pass
            return None

        char_x = [_cut_xcenter(cuts[i]) if i < len(cuts) else None for i in range(len(chars))]

        assignments: list[int] = []
        for cx in char_x:
            if cx is None:
                assignments.append(0)
                continue
            found = next(
                (wi for wi, (wx, wy, ww, wh) in enumerate(word_bboxes) if wx <= cx <= wx + ww),
                None,
            )
            if found is None:
                found = min(
                    range(len(word_bboxes)),
                    key=lambda wi: abs(word_bboxes[wi][0] + word_bboxes[wi][2] / 2 - cx),
                )
            assignments.append(found)

        result: list[tuple[str, list[float], list[tuple[int, int]]]] = []
        for wi, (wx, wy, ww, wh) in enumerate(word_bboxes):
            indices = [i for i, a in enumerate(assignments) if a == wi]
            if not indices:
                continue
            word_text = "".join(chars[i] for i in indices)
            if not word_text.strip():
                continue
            word_confs = [confs[i] for i in indices if i < len(confs)]
            baseline_pts = [(wx, wy + wh), (wx + ww, wy + wh)]
            result.append((word_text, word_confs, baseline_pts))
        return result

    def _empty_result(self, page_index: int) -> OCRResult:
        return OCRResult(
            text="", words=[], confidence=0.0,
            page_index=page_index, source="kraken", raw={},
        )
