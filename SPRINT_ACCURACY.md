# Sprint: Accuracy Improvements
## Items 5, 4, 3 — N-best Voting · Multispectral · Bleed-through Removal

Read CLAUDE.md fully before writing anything.
Token-efficient mode. Wait for confirmation after each item.

---

## Item 5 — N-best voting across two Kraken models

**Context:**
`KrakenBackend` already has `rec_model` (Agapet) and `rec_model_secondary`
(Muharaf) slots. The secondary model is loaded when `profile.rec_model_secondary`
is set and `profile.n_best > 1`. Currently the secondary model runs but its
output is not used in voting — only the primary is returned.

**What to build:**
When both models produce output for a line, vote on the best result per line:
- If both models agree (normalized edit distance < 0.1): use primary, boost confidence
- If they disagree: pick the one with higher mean character confidence
- Store both hypotheses in `WordToken.candidates` so the review UI can show both

**Changes — `ocr_engine/kraken_backend.py` only:**

```python
def _vote_hypotheses(self, primary: str, primary_conf: float,
                     secondary: str, secondary_conf: float) -> tuple[str, float]:
    """
    Compare two line hypotheses. Return (best_text, best_confidence).
    Agreement threshold: normalized edit distance < 0.1
    """
    from rapidfuzz.distance import Levenshtein
    max_len = max(len(primary), len(secondary), 1)
    dist = Levenshtein.distance(primary, secondary) / max_len
    if dist < 0.1:
        # Agreement — boost confidence slightly
        return primary, min(1.0, (primary_conf + secondary_conf) / 2 + 0.02)
    else:
        # Disagreement — pick higher confidence
        if secondary_conf > primary_conf:
            return secondary, secondary_conf
        return primary, primary_conf
```

Wire into `process_image()`: after getting primary records and secondary records,
call `_vote_hypotheses()` per line, use voted result for the WordToken text and
confidence. Store both in `candidates`.

**Config addition** (append to existing kraken section in config.yaml):
```yaml
kraken:
  # ... existing ...
  voting_enabled: true
  voting_agreement_threshold: 0.1
```

**Tests — add to `tests/test_kraken_backend.py`:**
```python
def test_vote_agrees_when_similar(): ...      # dist < 0.1 → primary returned, conf boosted
def test_vote_picks_higher_conf_when_different(): ...
def test_vote_symmetric(): ...                # same result regardless of arg order
```

Smoke-check: `pytest tests/test_kraken_backend.py -v`
Report pass/fail. Wait for confirmation before Item 4.

---

## Item 4 — Multispectral channel extraction

**What it does:**
Old manuscript ink often shows more clearly in one color channel than others.
Red channel often has better ink/parchment contrast than green or blue.
Run OCR on each channel separately, pick the one with highest mean confidence.

**Where it goes:** `preprocessing/adjustments.py` — new function, called from
`preprocessing/image_pipeline.py` before binarization when enabled.

**Implementation:**

```python
def best_channel_extraction(img: np.ndarray) -> np.ndarray:
    """
    For colour images: extract R, G, B channels separately.
    Return the channel with highest local contrast (std of pixel values).
    For grayscale: return as-is.
    Higher contrast = more ink/parchment separation = better for OCR.
    """
    if len(img.shape) == 2:
        return img  # already grayscale
    
    channels = cv2.split(img)  # B, G, R
    # Score each channel by standard deviation (higher = more contrast)
    scores = [np.std(ch) for ch in channels]
    best_idx = int(np.argmax(scores))
    return channels[best_idx]
```

**Config addition** (append to existing preprocessing section):
```yaml
preprocessing:
  multispectral:
    enabled: true
    method: "best_channel"   # "best_channel" | "all_channels" (future)
```

**Integration in `preprocessing/image_pipeline.py`:**
Call `best_channel_extraction()` as the first step before any other processing,
only when `config.preprocessing.multispectral.enabled` is True and the image
is colour (3 channels).

**Add to profile `andalusian_naskh` in config/profiles.yaml:**
```yaml
andalusian_naskh:
  preprocessing:
    # ... existing ...
    multispectral_enabled: true
```

**Tests — add to `tests/test_adjustments.py`:**
```python
def test_best_channel_returns_grayscale_for_color_input(): ...   # shape len == 2
def test_best_channel_noop_for_grayscale(): ...
def test_best_channel_picks_highest_contrast(): ...              # synthetic test
def test_best_channel_returns_ndarray(): ...
```

Smoke-check: `pytest tests/test_adjustments.py -v`
Report pass/fail. Wait for confirmation before Item 3.

---

## Item 3 — Bleed-through removal

**What it does:**
On thin parchment, text from the reverse side of the page shows through as
a faint ghost image. This adds noise that confuses the OCR segmenter and
recognizer. Remove it before binarization.

**Where it goes:** `preprocessing/adjustments.py` — new function.

**Implementation:**

```python
def remove_bleedthrough(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Remove bleed-through from manuscript images.
    
    Method: Background estimation + subtraction.
    1. Estimate background using large morphological closing
       (closing fills in the text, leaving only background texture)
    2. Subtract background from image to isolate foreground ink
    3. Normalize result
    
    strength: 0.0 = no effect, 1.0 = maximum removal
    Only applied when strength > 0.
    
    Input/output: grayscale uint8.
    """
    if strength <= 0:
        return img
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Large kernel closing estimates background (removes text, keeps texture)
    kernel_size = 51  # large enough to cover letter height
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Subtract: where background is dark (bleed-through), img becomes lighter
    result = cv2.subtract(background, img)
    result = cv2.subtract(background, result)
    
    # Blend with original based on strength
    result = cv2.addWeighted(img, 1.0 - strength, result, strength, 0)
    return result.astype(np.uint8)
```

**Config addition:**
```yaml
preprocessing:
  bleedthrough:
    enabled: false          # disabled by default — not all manuscripts need it
    strength: 0.5           # 0.0-1.0
```

**Add to `maghrebi_degraded` profile** (most likely to have bleed-through):
```yaml
maghrebi_degraded:
  preprocessing:
    # ... existing ...
    bleedthrough_enabled: true
    bleedthrough_strength: 0.5
```

Leave `andalusian_naskh` with bleedthrough disabled — test on real pages
before enabling.

**Integration in `preprocessing/image_pipeline.py`:**
Call `remove_bleedthrough()` after `best_channel_extraction()` and before
brightness/contrast adjustments. Only when enabled in profile/config.

**Tests — add to `tests/test_adjustments.py`:**
```python
def test_bleedthrough_noop_at_zero_strength(): ...
def test_bleedthrough_changes_image_at_nonzero(): ...
def test_bleedthrough_output_same_shape(): ...
def test_bleedthrough_output_uint8(): ...
def test_bleedthrough_noop_on_clean_image(): ...  # uniform white image unchanged
```

Smoke-check: `pytest tests/test_adjustments.py -v`
Report final test count across all tests: `pytest tests/ -x -q`
Commit and push all three items together.

---

## Final integration check

After all three items pass their tests:

```bash
pytest tests/ -x -q 2>&1 | tail -5
```

Then run the pipeline on `data/test_images/2.jpg` with `andalusian_naskh` profile
and report mean confidence. Compare to the previous baseline of 0.945.

If confidence improves: the items are working.
If confidence unchanged: expected — these help most on degraded/difficult pages,
not already-clean scans. Check by running on `veryold1.jpg` instead.
