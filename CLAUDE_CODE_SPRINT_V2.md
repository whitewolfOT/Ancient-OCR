# Ancient-OCR: Sprint V2
## Kraken · Agri Lexicon · Passim · Profile Router · Preprocessing Toolbox · Streamlit UI

**Read `CLAUDE.md` and `docs/MASTER_PLAN.md` fully before writing anything.
Then read this document fully. Then start Step 0.**

---

## Where the Claude Code session stopped

Branch: `claude/adoring-carson-UqiLp`

As of the last session, status was:

| Component | Status |
|---|---|
| Preprocessing pipeline (CLAHE, denoise, deskew) | ✓ Working — but no Sauvola, no bleed-through handling |
| Line segmentation | ✗ Critical gap — projection profiles only, no Kraken baseline |
| OCR ensemble (Paddle + Tesseract) | ⚠ Partial — N-best not end-to-end |
| Normalization + morphology | ✓ Working |
| Lexicon resolution (Lane's, Khorsi, Quranic — 187k entries) | ✓ Working |
| Corpus alignment (OpenITI / Passim) | ✗ Missing |
| Confidence engine + HITL | ✓ Working (272 tests, feedback loop active) |
| review_export | ✓ Working — but needs Kraken baseline coordinate support |
| API (FastAPI) + CLI | ✓ At least partially built |
| Streamlit UI | ✗ Not started |

This sprint adds everything missing and extends what's partial.
Do not re-implement anything from the ✓ column. Extend it.

---

## What this sprint adds (summary)

In order of dependency:

```
Step 0-A  WordToken schema extension + config.yaml additions     ← prerequisite for all
Step 0-B  config/profiles.yaml + ocr_engine/profile_loader.py   ← prerequisite for 1, 6, 7
Step 0-C  preprocessing/adjustments.py (new module, uses existing preprocessing/)
Step 1    ocr_engine/kraken_backend.py (profile-driven, with Muharaf + Agapet)
Step 2    lexicon_ingestion: ibn_awwam_filaha source (independent)
Step 3    align/openiti.py — Passim alignment (independent, post-scoring only)
Step 4    output/review_export.py — Kraken baseline coordinate support
Step 5    lexicon_engine/candidate_generator.py — weighted confusion costs
Step 6    api/server.py — new routes: /api/preview, /api/profiles, /api/suggest_profile
Step 7    ui/streamlit_app.py — profile editor, live preview, HITL review, alignment view
Step 8    Integration smoke-test
```

---

## Step 0-A — Schema + Config (prerequisite for everything)

**Touch only `ocr_engine/schema.py` and `config.yaml`.**

### WordToken extension

```python
class WordToken(BaseModel):
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]        # x, y, w, h — always populated, page-space
    page_index: int
    source: str                            # "paddle"|"tesseract"|"trocr"|"kraken"|"ensemble"
    region_id: str | None = None
    line_id: str | None = None             # NEW: Kraken line UUID
    baseline: list[tuple[int, int]] | None = None  # NEW: raw baseline points, page-space
```

Rules:
- `bbox` is always populated even for Kraken tokens. Compute from baseline polygon:
  `x=min_x, y=min_y, w=max_x−min_x, h=max_y−min_y`.
- `baseline` is None for Paddle/Tesseract/TrOCR.
- These are additive fields — nothing downstream breaks.

### config.yaml additions

Append (do not remove anything already in the file):

```yaml
profiles:
  enabled: true
  profiles_file: "config/profiles.yaml"
  active_profile: "default"

kraken:
  enabled: false
  seg_model: "zenodo:14295555"
  rec_model: "agapet"
  rec_model_secondary: "muharaf"
  n_best: 3
  rtl: true
  binarizer: "nlbin"          # profile overrides this per-request
  device: "cpu"
  model_dir: "models/kraken"

align:
  passim:
    enabled: false
    threshold: 0.85
    window_chars: 2000
    fallback: "rapidfuzz"

lexicon:
  confusion_costs:
    dot_pairs:      0.78      # ب/ت/ث/ن/ي
    body_pairs:     0.74      # ج/ح/خ
    qaf_faa:        0.77      # ف/ق
    emphatic_pairs: 0.81      # ص/ض  ط/ظ
    tail_pairs:     0.84      # د/ذ  ر/ز
    taa_haa:        0.68      # ة/ه
    default:        0.85
```

### Smoke-check

```python
from ocr_engine.schema import WordToken
t = WordToken(text="test", confidence=0.9, bbox=(0,0,10,10),
              page_index=0, source="kraken",
              line_id="line_01", baseline=[(0,8),(10,9)])
assert t.baseline is not None and t.line_id == "line_01"
```

Wait for confirmation before Step 0-B.

---

## Step 0-B — Profile System

**New files: `config/profiles.yaml` and `ocr_engine/profile_loader.py`.**
Do not touch any existing file except to add one import to `main.py` at the end.

### `config/profiles.yaml`

This file is user-editable. It lives separately from `config.yaml` so users can modify
profiles without touching core configuration.

```yaml
profiles:
  default:
    description: "Fallback — printed Arabic, clean scan"
    binarizer: "otsu"
    seg_model: "zenodo:14295555"
    rec_model: "agapet"
    rec_model_secondary: null
    n_best: 3
    rtl: true
    device: "cpu"
    preprocessing:
      brightness: 0
      contrast: 1.0
      gamma: 1.0
      saturation: 1.0
      stroke_normalization:
        enabled: false
        target_width: 2
      denoise_strength: 0
      sharpen: 0.0

  andalusian_naskh:
    description: "Ibn al-Awwam manuscript — Andalusian Naskh, 12th–13th c., yellowed"
    binarizer: "nlbin"
    seg_model: "zenodo:14295555"
    rec_model: "agapet"
    rec_model_secondary: "muharaf"
    n_best: 5
    rtl: true
    device: "cpu"
    preprocessing:
      brightness: 8
      contrast: 1.15
      gamma: 1.1
      saturation: 0.0           # grayscale — ignore
      stroke_normalization:
        enabled: false
        target_width: 2
      denoise_strength: 10
      sharpen: 0.3

  maghrebi_degraded:
    description: "Maghrebi script — bleed-through, curved lines"
    binarizer: "sauvola"
    seg_model: "zenodo:14295555"
    rec_model: "muharaf"
    rec_model_secondary: "agapet"
    n_best: 5
    rtl: true
    device: "cpu"
    preprocessing:
      brightness: 5
      contrast: 1.2
      gamma: 1.1
      saturation: 1.0
      stroke_normalization:
        enabled: true
        target_width: 2
      denoise_strength: 15
      sharpen: 0.5

  low_contrast:
    description: "Faded ink — low contrast scan"
    binarizer: "sauvola"
    seg_model: "zenodo:14295555"
    rec_model: "agapet"
    rec_model_secondary: null
    n_best: 3
    rtl: true
    device: "cpu"
    preprocessing:
      brightness: 15
      contrast: 1.4
      gamma: 0.9
      saturation: 1.0
      stroke_normalization:
        enabled: false
        target_width: 2
      denoise_strength: 5
      sharpen: 0.8
```

### `ocr_engine/profile_loader.py`

```python
"""
Profile loader — reads config/profiles.yaml and provides typed OCRProfile objects.
No import-time side effects. No model loading here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import copy
import yaml


@dataclass
class PreprocessingParams:
    brightness: int = 0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    stroke_normalization_enabled: bool = False
    stroke_target_width: int = 2
    denoise_strength: int = 0
    sharpen: float = 0.0


@dataclass
class OCRProfile:
    name: str
    description: str = ""
    binarizer: str = "otsu"          # "otsu" | "nlbin" | "sauvola"
    seg_model: str = "zenodo:14295555"
    rec_model: str = "agapet"
    rec_model_secondary: str | None = None
    n_best: int = 3
    rtl: bool = True
    device: str = "cpu"
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)


class ProfileManager:
    """Load, save, and serve OCRProfile objects from a YAML file."""

    def __init__(self, profiles_path: Path):
        self._path = profiles_path
        self._profiles: dict[str, OCRProfile] = {}
        self.load()

    def load(self) -> None:
        """Re-read the YAML file. Safe to call at any time (hot-reload)."""
        with open(self._path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._profiles = {}
        for name, cfg in data.get("profiles", {}).items():
            pp = cfg.get("preprocessing", {})
            sn = pp.get("stroke_normalization", {})
            pre = PreprocessingParams(
                brightness=int(pp.get("brightness", 0)),
                contrast=float(pp.get("contrast", 1.0)),
                gamma=float(pp.get("gamma", 1.0)),
                saturation=float(pp.get("saturation", 1.0)),
                stroke_normalization_enabled=bool(sn.get("enabled", False)),
                stroke_target_width=int(sn.get("target_width", 2)),
                denoise_strength=int(pp.get("denoise_strength", 0)),
                sharpen=float(pp.get("sharpen", 0.0)),
            )
            self._profiles[name] = OCRProfile(
                name=name,
                description=cfg.get("description", ""),
                binarizer=cfg.get("binarizer", "otsu"),
                seg_model=cfg.get("seg_model", "zenodo:14295555"),
                rec_model=cfg.get("rec_model", "agapet"),
                rec_model_secondary=cfg.get("rec_model_secondary"),
                n_best=int(cfg.get("n_best", 3)),
                rtl=bool(cfg.get("rtl", True)),
                device=cfg.get("device", "cpu"),
                preprocessing=pre,
            )

    def save(self) -> None:
        """Write current profiles back to YAML."""
        data: dict = {"profiles": {}}
        for name, p in self._profiles.items():
            pr = p.preprocessing
            data["profiles"][name] = {
                "description": p.description,
                "binarizer": p.binarizer,
                "seg_model": p.seg_model,
                "rec_model": p.rec_model,
                "rec_model_secondary": p.rec_model_secondary,
                "n_best": p.n_best,
                "rtl": p.rtl,
                "device": p.device,
                "preprocessing": {
                    "brightness": pr.brightness,
                    "contrast": pr.contrast,
                    "gamma": pr.gamma,
                    "saturation": pr.saturation,
                    "stroke_normalization": {
                        "enabled": pr.stroke_normalization_enabled,
                        "target_width": pr.stroke_target_width,
                    },
                    "denoise_strength": pr.denoise_strength,
                    "sharpen": pr.sharpen,
                },
            }
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def get(self, name: str) -> OCRProfile:
        """Return named profile, or 'default' if not found."""
        return copy.deepcopy(self._profiles.get(name, self._profiles["default"]))

    def list(self) -> list[str]:
        return list(self._profiles.keys())

    def upsert(self, profile: OCRProfile) -> None:
        """Add or update a profile in memory. Call save() to persist."""
        self._profiles[profile.name] = copy.deepcopy(profile)

    def delete(self, name: str) -> bool:
        if name == "default":
            return False   # default profile is protected
        removed = self._profiles.pop(name, None)
        return removed is not None
```

### Wire into main.py

At the top of `main.py`, after existing imports, add:

```python
from ocr_engine.profile_loader import ProfileManager, OCRProfile
from pathlib import Path

_profile_manager: ProfileManager | None = None

def get_profile_manager() -> ProfileManager:
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager(Path("config/profiles.yaml"))
    return _profile_manager
```

`process_file()` gains an optional `profile_name: str = "default"` parameter:

```python
def process_file(file_path: str, mode: str = "clean", profile_name: str = "default") -> dict:
    profile = get_profile_manager().get(profile_name)
    # pass profile down to run_pipeline
    return run_pipeline(pages, mode, profile=profile)
```

**Tests — `tests/test_profile_loader.py`:**

```python
def test_load_profiles(tmp_profiles_yaml): ...          # load fixture YAML, check counts
def test_get_returns_default_for_unknown(): ...          # get("nonexistent") → default
def test_upsert_and_save(tmp_profiles_yaml): ...         # add profile, save, reload
def test_delete_default_forbidden(): ...                 # delete("default") → False
def test_preprocessing_params_typed(): ...               # brightness is int, gamma is float
def test_get_returns_deep_copy(): ...                    # mutating returned profile doesn't affect manager
```

Smoke-check: `pytest tests/test_profile_loader.py -v`
Wait for confirmation before Step 0-C.

---

## Step 0-C — `preprocessing/adjustments.py`

**New file inside the existing `preprocessing/` package.**
This does NOT replace or modify `preprocessing/image_pipeline.py` yet —
that integration happens at the end of this step.

### Why a separate module, not `ocr_engine/preprocessing.py`

The existing repo already has `preprocessing/` as the home for all image
manipulation. Adding profile-driven adjustments there keeps the package
coherent. `ocr_engine/` is for OCR engine adapters, not image processing.

### File: `preprocessing/adjustments.py`

```python
"""
Profile-driven image adjustments.
All functions: take np.ndarray (uint8, grayscale or BGR), return np.ndarray.
No import-time side effects. cv2 and numpy only — no kraken dependency.
"""
from __future__ import annotations
import cv2
import numpy as np
from ocr_engine.profile_loader import PreprocessingParams


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0,
                                contrast: float = 1.0) -> np.ndarray:
    """alpha=contrast, beta=brightness via convertScaleAbs."""
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01:
        return img
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def adjust_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """No-op on grayscale. Adjusts HSV saturation channel for colour."""
    if len(img.shape) == 2 or abs(factor - 1.0) < 0.01:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def normalize_stroke_thickness(img: np.ndarray, target_width: int = 2) -> np.ndarray:
    """
    Estimate current median stroke width via distance transform on binarized image,
    then dilate or erode to reach target_width.
    Input/output: grayscale uint8, white background (standard).
    IMPORTANT: apply only after denoise, never on colour images.
    Expensive — only enable when profile.stroke_normalization_enabled is True.
    """
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    non_zero = dist[dist > 0]
    if len(non_zero) == 0:
        return img
    current_width = float(np.median(non_zero)) * 2
    diff = target_width - current_width
    if abs(diff) < 0.5:
        return img
    ksize = max(1, int(abs(diff) + 0.5))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    adjusted = cv2.dilate(binary, kernel) if diff > 0 else cv2.erode(binary, kernel)
    return 255 - adjusted  # back to white-background grayscale


def denoise(img: np.ndarray, strength: int = 15) -> np.ndarray:
    if strength <= 0:
        return img
    # fastNlMeansDenoising requires grayscale uint8
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(img, None, h=float(strength), templateWindowSize=7,
                                    searchWindowSize=21)


def sharpen(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def apply_profile_adjustments(img: np.ndarray, params: PreprocessingParams) -> np.ndarray:
    """
    Apply all profile-driven adjustments in canonical order.
    This is called by preprocessing/image_pipeline.py when a profile is active.
    The binarizer (nlbin/sauvola/otsu) is NOT applied here — that's KrakenBackend's job.
    """
    # 1. Saturation (colour only — no-op on grayscale)
    img = adjust_saturation(img, params.saturation)

    # 2. Convert to grayscale for remaining steps
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Brightness / contrast
    img = adjust_brightness_contrast(img, params.brightness, params.contrast)

    # 4. Gamma
    img = adjust_gamma(img, params.gamma)

    # 5. Denoise
    img = denoise(img, params.denoise_strength)

    # 6. Stroke normalization (expensive — only when enabled)
    if params.stroke_normalization_enabled:
        img = normalize_stroke_thickness(img, params.stroke_target_width)

    # 7. Sharpen
    img = sharpen(img, params.sharpen)

    return img
```

### Integration into `preprocessing/image_pipeline.py`

Find the `run_pipeline(image, config)` function (or equivalent entry point).
Add an optional `profile` parameter:

```python
from preprocessing.adjustments import apply_profile_adjustments
from ocr_engine.profile_loader import OCRProfile

def run_pipeline(image: np.ndarray, config, profile: OCRProfile | None = None) -> np.ndarray:
    # ... existing CLAHE, denoise, deskew steps ...

    # NEW: apply profile-driven fine adjustments AFTER existing preprocessing,
    # BEFORE the pipeline returns the processed image.
    if profile is not None:
        image = apply_profile_adjustments(image, profile.preprocessing)

    return image
```

The profile adjustments run last in the existing pipeline — they are a refinement
layer on top of the general preprocessing, not a replacement.

**Tests — `tests/test_adjustments.py`:**

```python
def test_brightness_increases_mean(): ...
def test_contrast_increases_std(): ...
def test_gamma_gt1_darkens(): ...
def test_saturation_noop_on_grayscale(): ...
def test_stroke_normalization_synthetic(): ...   # synthetic thin lines → target_width=4
def test_denoise_noop_at_zero(): ...
def test_sharpen_noop_at_zero(): ...
def test_apply_pipeline_order_deterministic(): ...  # same input+params → same output
```

Smoke-check: `pytest tests/test_adjustments.py -v`
Wait for confirmation before Step 1.

---

## Step 1 — `ocr_engine/kraken_backend.py` (profile-driven)

**Depends on:** Steps 0-A, 0-B, 0-C all complete.

### Architecture note

The binarizer (nlbin / sauvola / otsu) is KrakenBackend's responsibility, not
`image_pipeline.py`'s. This is because Kraken's `nlbin` is a Kraken-internal
function tuned for manuscript parchment — it cannot meaningfully be called outside
the Kraken workflow. The general preprocessing (`apply_profile_adjustments`) runs
first (driven by `image_pipeline.py`), and then KrakenBackend binarizes the result
using the profile's `binarizer` setting before feeding into the segmenter.

This matches how CREMMA/CATMUS production pipelines are structured.

### `ocr_engine/kraken_backend.py`

```python
class KrakenBackend(BaseOCRBackend):
    name = "kraken"

    def __init__(self, config, profile: OCRProfile):
        self.config = config
        self.profile = profile
        self._seg_model = None   # lazy-loaded
        self._rec_model = None
        self._rec_secondary = None

    def is_available(self) -> bool:
        try:
            import kraken  # noqa
            return True
        except ImportError:
            return False

    def _ensure_models(self) -> bool:
        """Download models if absent. Return False if download fails."""
        # use kraken.repo.get_model(handle, model_dir) for each model
        # set self._seg_model, self._rec_model, self._rec_secondary
        # return True on success

    def _binarize(self, img: np.ndarray) -> np.ndarray:
        binarizer = self.profile.binarizer
        if binarizer == "nlbin":
            from kraken.binarization import nlbin
            return nlbin(img)
        elif binarizer == "sauvola":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        else:  # otsu default
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

    @staticmethod
    def _baseline_bbox(points: list[tuple[int,int]]) -> tuple[int,int,int,int]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y = min(xs), min(ys)
        return (x, y, max(xs) - x, max(ys) - y)

    def process_image(self, image: np.ndarray, page_index: int) -> OCRResult:
        if not self.is_available() or not self._ensure_models():
            return self._empty_result(page_index)

        from kraken import pageseg, rpred

        bin_img = self._binarize(image)

        seg = pageseg.segment(
            bin_img,
            model=self._seg_model,
            text_direction="rtl" if self.profile.rtl else "horizontal-lr",
        )

        all_words: list[WordToken] = []
        line_texts: list[str] = []
        line_confidences: list[float] = []

        for line in seg.lines:
            # Primary recognition
            records = list(rpred.rpred(
                self._rec_model, bin_img, seg,
                bidi_reorder=self.profile.rtl,
            ))
            # Secondary recognition (N-best from second model)
            secondary_records = []
            if self._rec_secondary and self.profile.n_best > 1:
                secondary_records = list(rpred.rpred(
                    self._rec_secondary, bin_img, seg,
                    bidi_reorder=self.profile.rtl,
                ))

            for rec in records[:1]:  # use primary for token building
                # Split prediction on whitespace to get word spans
                words_and_spans = self._split_prediction(rec)
                for word_text, span_confidences, span_baseline in words_and_spans:
                    if not word_text.strip():
                        continue
                    conf = float(sum(span_confidences) / len(span_confidences)) if span_confidences else 0.0
                    bbox = self._baseline_bbox(span_baseline) if span_baseline else (0, 0, 0, 0)
                    all_words.append(WordToken(
                        text=word_text,
                        confidence=conf,
                        bbox=bbox,
                        page_index=page_index,
                        source="kraken",
                        line_id=str(id(line)),  # stable within this call
                        baseline=span_baseline,
                    ))

            if records:
                line_texts.append(records[0].prediction)
                line_confidences.append(
                    float(sum(records[0].confidences) / len(records[0].confidences))
                    if records[0].confidences else 0.0
                )

        return OCRResult(
            text="\n".join(line_texts),
            words=all_words,
            confidence=float(sum(line_confidences) / len(line_confidences)) if line_confidences else 0.0,
            page_index=page_index,
            source="kraken",
            raw={
                "seg_model": self.profile.seg_model,
                "rec_model": self.profile.rec_model,
                "n_best": self.profile.n_best,
                "profile": self.profile.name,
            },
        )

    def _split_prediction(self, record) -> list[tuple[str, list[float], list[tuple[int,int]]]]:
        """Split a Kraken OCR record into (word, confidences, baseline_segment) triples."""
        # Kraken record has .prediction (str), .cuts (list of bbox or baseline points), .confidences
        # Split on Arabic word boundaries (spaces, zero-width non-joiners)
        # Return one tuple per word
        ...

    def _empty_result(self, page_index: int) -> OCRResult:
        return OCRResult(text="", words=[], confidence=0.0,
                         page_index=page_index, source="kraken", raw={})
```

### Integration with ensemble

In `ocr_engine/ensemble.py`, pass the profile when constructing KrakenBackend:

```python
from ocr_engine.kraken_backend import KrakenBackend
from ocr_engine.profile_loader import OCRProfile

def build_backends(config, profile: OCRProfile | None = None):
    backends = []
    # ... existing Paddle and Tesseract backend construction ...
    if config.kraken.enabled:
        if profile is None:
            from ocr_engine.profile_loader import ProfileManager
            from pathlib import Path
            profile = ProfileManager(Path("config/profiles.yaml")).get("default")
        backends.append(KrakenBackend(config=config, profile=profile))
    return backends
```

**Tests — `tests/test_kraken_backend.py`** (same spec as sprint V1, plus):

```python
def test_profile_binarizer_used(mock_kraken):
    profile = OCRProfile(name="test", binarizer="sauvola")
    backend = KrakenBackend(config=mock_config, profile=profile)
    # assert nlbin NOT called, adaptiveThreshold called

def test_profile_n_best_passed(mock_kraken):
    profile = OCRProfile(name="test", n_best=5)
    # assert secondary model called when n_best > 1
```

Wait for confirmation before Step 2.

---

## Step 2 — Agricultural lexicon (ibn_awwam_filaha)

**Fully independent of Step 1. Can run in parallel.**
Spec unchanged from sprint V1 — see that document's Step 2 for full details.
Summary: add `IBN_AWWAM_FILAHA` source to `lexicon_ingestion/sources.py`,
add `parse_openiti_markdown()` to `parser.py`, add `scripts/ingest_ibn_awwam.py`.

---

## Step 3 — `align/openiti.py` (Passim alignment)

**Fully independent of all other steps. Post-scoring only.**
Spec unchanged from sprint V1 — see Step 3 there for full details.

---

## Step 4 — `output/review_export.py` — Kraken baseline support

**Depends on:** Step 0-A and Step 1 complete.
Spec unchanged from sprint V1 — see Step 4 there for full details.

---

## Step 5 — Weighted confusion costs in `candidate_generator.py`

**Fully independent of all other steps.**
Spec unchanged from sprint V1 — see Step 5 there for full details.

---

## Step 6 — FastAPI new routes

**Depends on:** Steps 0-B, 0-C, 1 complete. Must NOT re-implement existing routes.**

### What already exists (do not touch)

```
POST /process          — full pipeline, existing
GET  /health           — existing
POST /feedback         — uses feedback_store, existing
POST /calibrate        — existing
GET  /review           — existing review queue, existing
```

### New routes to ADD to `api/server.py`

Add a `ProfileManager` instance at module level (reuse if already initialized by main.py):

```python
# At module level in api/server.py — ADD these, do not replace existing code
from ocr_engine.profile_loader import ProfileManager, OCRProfile
from preprocessing.adjustments import apply_profile_adjustments
import copy, base64, cv2, numpy as np
from pathlib import Path

_profile_mgr = ProfileManager(Path("config/profiles.yaml"))
```

#### `POST /api/preview`

Returns a processed (but not OCR'd) image for live tuning in Streamlit.
Accepts profile overrides so the user can drag sliders without saving.

```python
@app.post("/api/preview")
async def preview(
    file: UploadFile = File(...),
    profile_name: str = Form("default"),
    brightness:   float = Form(0),
    contrast:     float = Form(1.0),
    gamma:        float = Form(1.0),
    saturation:   float = Form(1.0),
    stroke_enabled: bool = Form(False),
    stroke_width:   int  = Form(2),
    denoise_strength: int = Form(0),
    sharpen:      float = Form(0.0),
):
    profile = _profile_mgr.get(profile_name)
    # Apply per-request overrides without saving to profile
    profile = copy.deepcopy(profile)
    profile.preprocessing.brightness = int(brightness)
    profile.preprocessing.contrast = contrast
    profile.preprocessing.gamma = gamma
    profile.preprocessing.saturation = saturation
    profile.preprocessing.stroke_normalization_enabled = stroke_enabled
    profile.preprocessing.stroke_target_width = stroke_width
    profile.preprocessing.denoise_strength = int(denoise_strength)
    profile.preprocessing.sharpen = sharpen

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    processed = apply_profile_adjustments(img, profile.preprocessing)
    _, buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    return {"processed_image_b64": b64, "profile_used": profile_name}
```

#### `GET /api/profiles`

```python
@app.get("/api/profiles")
def list_profiles():
    return {"profiles": _profile_mgr.list()}
```

#### `GET /api/profiles/{name}`

```python
@app.get("/api/profiles/{name}")
def get_profile(name: str):
    p = _profile_mgr.get(name)
    return {
        "name": p.name,
        "description": p.description,
        "binarizer": p.binarizer,
        "seg_model": p.seg_model,
        "rec_model": p.rec_model,
        "rec_model_secondary": p.rec_model_secondary,
        "n_best": p.n_best,
        "rtl": p.rtl,
        "device": p.device,
        "preprocessing": {
            "brightness": p.preprocessing.brightness,
            "contrast": p.preprocessing.contrast,
            "gamma": p.preprocessing.gamma,
            "saturation": p.preprocessing.saturation,
            "stroke_normalization": {
                "enabled": p.preprocessing.stroke_normalization_enabled,
                "target_width": p.preprocessing.stroke_target_width,
            },
            "denoise_strength": p.preprocessing.denoise_strength,
            "sharpen": p.preprocessing.sharpen,
        },
    }
```

#### `PUT /api/profiles/{name}`

```python
@app.put("/api/profiles/{name}")
async def update_profile(name: str, body: dict):
    """
    Accept the same dict shape as GET returns.
    Validate required fields; reject if profile name conflicts with a protected name.
    Save via _profile_mgr.upsert() + _profile_mgr.save().
    """
    if not body.get("name"):
        body["name"] = name
    # Build OCRProfile from dict (reuse profile_loader parsing logic)
    pp_data = body.get("preprocessing", {})
    sn = pp_data.get("stroke_normalization", {})
    pre = PreprocessingParams(
        brightness=int(pp_data.get("brightness", 0)),
        # ... etc.
    )
    profile = OCRProfile(name=name, ...)
    _profile_mgr.upsert(profile)
    _profile_mgr.save()
    return {"status": "saved", "name": name}
```

#### `DELETE /api/profiles/{name}`

```python
@app.delete("/api/profiles/{name}")
def delete_profile(name: str):
    if not _profile_mgr.delete(name):
        raise HTTPException(status_code=403, detail="Cannot delete protected profile")
    _profile_mgr.save()
    return {"status": "deleted", "name": name}
```

#### `POST /api/suggest_profile` (placeholder — intentionally simple)

```python
@app.post("/api/suggest_profile")
async def suggest_profile(file: UploadFile = File(...)):
    """
    Very lightweight heuristics — no machine learning, no fingerprinting.
    Returns a suggested profile name the user can accept or ignore.
    Phase 1 implementation: always returns 'default'. Enhance later.
    Phase 2 (future): check mean brightness, std, edge density and pick profile.
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"suggested_profile": "default", "confidence": 0.0}

    mean_brightness = float(np.mean(img))
    std = float(np.std(img))

    # Minimal heuristics — tuned conservatively
    if mean_brightness < 100:
        return {"suggested_profile": "low_contrast", "confidence": 0.6}
    elif std < 30:
        return {"suggested_profile": "low_contrast", "confidence": 0.5}
    else:
        return {"suggested_profile": "default", "confidence": 0.9}
```

#### `POST /api/correction` — use existing feedback_store

```python
@app.post("/api/correction")
async def submit_correction(token_id: str = Form(...),
                            corrected_text: str = Form(...),
                            original_text: str = Form(...)):
    """
    Store human correction via existing feedback_store.
    Does NOT directly modify confusion costs — that requires explicit calibration.
    """
    from training.feedback_store import submit_feedback, FeedbackEntry
    entry = FeedbackEntry(
        token_id=token_id,
        original=original_text,
        corrected=corrected_text,
        source="hitl_ui",
    )
    entry_id = submit_feedback(entry)
    return {"status": "stored", "entry_id": entry_id}
```

**Tests — `tests/test_api_profiles.py`:**

```python
def test_list_profiles_returns_names(client): ...
def test_get_profile_returns_structure(client): ...
def test_update_profile_persists(client, tmp_profiles_yaml): ...
def test_delete_default_returns_403(client): ...
def test_preview_returns_b64_image(client, sample_image_file): ...
def test_suggest_profile_dark_image(client, dark_image_file): ...  # → low_contrast
```

Wait for confirmation before Step 7.

---

## Step 7 — `ui/streamlit_app.py`

**Depends on:** Step 6 (API routes) complete and the FastAPI server running.
Add `streamlit` and `Pillow` to `requirements.txt` under `[ui]` optional group in `pyproject.toml`:

```toml
[project.optional-dependencies]
ui = ["streamlit>=1.35", "Pillow>=10"]
```

### File: `ui/streamlit_app.py`

Build as a four-tab application. Use `st.set_page_config(layout="wide")`.
All API calls go through `ui/api_client.py` — never call `requests` inline in the UI.

#### `ui/api_client.py`

```python
"""Single source of truth for all API calls from the Streamlit frontend."""
import os, requests, base64
from pathlib import Path
from PIL import Image
import io

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def get_profiles() -> list[str]: ...
def get_profile(name: str) -> dict: ...
def save_profile(name: str, profile_dict: dict) -> dict: ...
def delete_profile(name: str) -> dict: ...
def preview(image_bytes: bytes, profile_name: str, overrides: dict) -> Image.Image: ...
def suggest_profile(image_bytes: bytes) -> dict: ...
def run_full_ocr(image_bytes: bytes, profile_name: str) -> dict: ...
def submit_correction(token_id: str, corrected_text: str, original_text: str) -> dict: ...
```

#### Tab 1 — Profile Manager

```python
# Tab: Profile Manager
# Left column: list of profiles as st.radio buttons
# Right column: st.form with ALL profile fields
#   - Description: st.text_input
#   - Binarizer: st.selectbox ["otsu", "nlbin", "sauvola"]
#   - Seg model: st.text_input (placeholder shows zenodo handle)
#   - Rec model: st.text_input
#   - Rec model secondary: st.text_input (optional)
#   - N-best: st.slider(1, 10, 3)
#   - RTL: st.checkbox (default True)
#   - Device: st.selectbox ["cpu", "cuda"]
#   - Preprocessing expander:
#       brightness: st.slider(-50, 50, 0)
#       contrast:   st.slider(0.5, 3.0, 1.0, step=0.05)
#       gamma:      st.slider(0.5, 2.0, 1.0, step=0.05)
#       saturation: st.slider(0.0, 2.0, 1.0, step=0.05)
#       Stroke normalization: st.checkbox + st.slider(1, 6, 2) if enabled
#       denoise:    st.slider(0, 50, 0)
#       sharpen:    st.slider(0.0, 2.0, 0.0, step=0.1)
# Save / New / Delete buttons
# Protect "default" profile: disable Delete button when default is selected
```

#### Tab 2 — Live Preview

```python
# Tab: Live Preview
# Left column:
#   - st.file_uploader (PNG/JPG/TIFF)
#   - Profile selector (st.selectbox)
#   - "Suggest Profile" button → POST /api/suggest_profile → st.info(suggested)
#   - Preprocessing override sliders (same as Profile Manager but not saved to profile)
#   - "Refresh Preview" button (st.form submit — avoids constant reruns)
#   - "Save as new profile..." expandable form
# Right column:
#   - st.image side-by-side: original vs processed
#   - "Run Full OCR →" button that navigates to Tab 3 (set session_state)
```

#### Tab 3 — Full OCR and HITL Review

```python
# Tab: OCR & Review
# Requires: image uploaded in Tab 2 AND profile selected
# If not: st.warning("Upload an image and select a profile in the Preview tab first.")
#
# "Run OCR" button → POST /api/run_full → display results
# For each page result:
#   For each line (grouped by line_id):
#     - Show line image crop (from WordToken baseline/bbox)
#     - Show N-best candidates as st.radio
#     - Show confidence badge (color-coded: green ≥0.9, yellow 0.7-0.9, red <0.7)
#     - For uncertain/review_required: st.text_input pre-filled with top candidate
#     - "Submit correction" button → POST /api/correction
#   Line-level: "Accept all in line" button
# "Export corrected text" button → st.download_button
```

#### Tab 4 — Alignment View

```python
# Tab: Alignment
# Show side-by-side:
#   Left: raw OCR text (from OCRResult.text)
#   Right: aligned text (from page_result.aligned_text if accepted)
# Highlight diffs using difflib.ndiff
# If alignment not accepted: show st.warning("Alignment confidence below threshold")
# Show alignment_confidence as metric
```

### Running the UI

Add to README.md:

```bash
# Install UI dependencies
pip install ".[ui]" --break-system-packages

# Start API in one terminal
uvicorn api.server:app --reload

# Start Streamlit in another
streamlit run ui/streamlit_app.py
```

**Tests — `tests/test_ui_api_client.py`** (mock API responses):

```python
def test_get_profiles_returns_list(mock_requests): ...
def test_preview_returns_pil_image(mock_requests): ...
def test_run_full_ocr_returns_dict(mock_requests): ...
```

Wait for confirmation before Step 8.

---

## Step 8 — Integration smoke-test

**All steps complete. Run in this order.**

```bash
# 1. Full test suite
pytest tests/ -v --tb=short 2>&1 | tail -60

# 2. Start API
uvicorn api.server:app --reload &

# 3. Test profile routes
curl -s http://localhost:8000/api/profiles | python3 -m json.tool
curl -s http://localhost:8000/api/profiles/andalusian_naskh | python3 -m json.tool

# 4. Test preview endpoint (replace test.jpg with a real image)
curl -s -X POST http://localhost:8000/api/preview \
  -F "file=@data/test_images/sample_classical.jpg" \
  -F "profile_name=andalusian_naskh" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d['processed_image_b64']), 'b64 chars')"

# 5. Test suggest_profile
curl -s -X POST http://localhost:8000/api/suggest_profile \
  -F "file=@data/test_images/sample_classical.jpg" \
  | python3 -m json.tool

# 6. Full pipeline with profile
python -m cli.main process data/test_images/sample_classical.jpg \
  --mode debug --profile andalusian_naskh

# 7. Start Streamlit (confirm loads without error)
streamlit run ui/streamlit_app.py &
sleep 3 && curl -s http://localhost:8501 | grep -c "streamlit"
```

Check and report:
- All pytest tests pass (count, 0 failures)
- `/api/profiles` returns at least 4 profile names
- Preview returns a base64 string > 1000 chars
- Suggest profile returns a valid profile name
- CLI `--mode debug` output contains `"profile": "andalusian_naskh"` in at least one token's `raw`
- Streamlit loads (curl returns > 0 matches)

---

## Compatibility matrix (complete)

| Component | Reads from | Writes to / modifies | Breaking? |
|---|---|---|---|
| WordToken extension | — | `ocr_engine/schema.py` | No — additive |
| profiles.yaml + ProfileManager | new files | `main.py` (one import + param) | No |
| `preprocessing/adjustments.py` | new file | `preprocessing/image_pipeline.py` (one call at end) | No |
| KrakenBackend | `ocr_engine/base.py`, profile | `ocr_engine/ensemble.py` (registration) | No — optional |
| Ibn Awwam source | new source entry | `sources.py`, `parser.py` | No |
| `align/openiti.py` | new file | `main.py` (post-scoring hook only) | Minor — 2 new fields on page result |
| `review_export.py` update | `WordToken.baseline` | existing `ReviewItem` | No — additive |
| Confusion costs | `config.yaml` | `candidate_generator.py` | No |
| API new routes | new routes added | `api/server.py` | No — additive |
| `ui/streamlit_app.py` | new directory | nothing existing | No |

---

## Non-negotiable rules (from CLAUDE.md — in force for every step)

1. Never overwrite a token without storing original + alternatives.
2. Lexicon evidence > morphology > model guessing. Passim/alignment is a post-scoring layer only.
3. No side effects at import. ProfileManager is instantiated explicitly, never at import time.
4. Config drives everything. All costs, thresholds, model handles live in yaml files.
5. Graceful degradation. Kraken absent → log WARNING, pipeline continues. Java/rapidfuzz absent → alignment skipped. Profile missing → fallback to "default".
6. Each module independently unit-testable.
7. `review_required` and `uncertain` tokens are never silently corrected.
8. Alignment accepted results go into `aligned_text` on the page result — they never overwrite `TokenState.selected`.
9. Feedback never auto-applies. Corrections stored by `/api/correction` require explicit `/calibrate` trigger.
