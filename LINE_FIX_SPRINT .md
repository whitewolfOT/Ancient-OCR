# Line Fix Sprint
## Segmentation Quality · LineReviewView Edit · Import Tab Fixes · Layout

Read CLAUDE.md fully. Read this document fully.
Run Step 0 before writing any code.
Token-efficient mode. Wait for confirmation after each step.

---

## What this sprint fixes

1. Segmentation noise filtering (dots, fragments detected as lines)
2. LineReviewView: draw-to-replace broken lines, next-button skip bug
3. Import tab: delete pages, segmentation failed diagnosis, layout fixes
4. AnnotationView: move line crop panel below toolbar

---

## Step 0 — Diagnosis

Run and report ALL output:

```bash
# Check current segmentation filter code
grep -n "min_height\|min_width\|filter\|threshold\|line_height\|bbox" scripts/generate_line_crops.py | head -20

# Check accept-lines endpoint filtering
grep -n "min_height\|min_width\|filter\|bbox\|height\|width" api/routes.py | grep -i "import\|segment\|accept" | head -20

# Check config.yaml for any existing segmentation thresholds
grep -n "segment\|threshold\|min_\|filter" config.yaml

# Check LineReviewView next button logic
grep -n "nextLine\|next\|skip\|index\|pending\|currentLine" frontend/src/components/LineReviewView.jsx | head -30

# Check ImportView segmentation call
grep -n "segment\|error\|failed\|catch\|blla\|model" api/routes.py | grep -A3 "segment" | head -30

# Check AnnotationView line crop panel location in JSX
grep -n "WORD CROP\|word.crop\|WordCrop\|crop\|panel\|right" frontend/src/components/AnnotationView.jsx | head -20

# Check ImportView line crop panel location
grep -n "WORD CROP\|crop\|selected.*line\|line.*selected\|detail\|panel" frontend/src/components/ImportView.jsx | head -20
```

Report all output. Wait for confirmation before Step 1.

---

## Step 1 — Segmentation noise filtering

**Files: `config.yaml`, `scripts/generate_line_crops.py`, `api/routes.py`**

### 1-A  Add thresholds to config.yaml

```yaml
segmentation:
  line_filters:
    min_height_px: 20        # lines shorter than this are noise (dots, dashes)
    min_width_px: 80         # lines narrower than this are fragments
    min_area_px2: 1600       # lines smaller than this by area are noise
    max_lines_per_page: 60   # sanity cap — reject pages with absurd line counts
```

### 1-B  Add filter function

Create `preprocessing/seg_filter.py`:

```python
"""
Filter Kraken blla segmentation results to remove noise lines.
Called from generate_line_crops.py and /api/import/accept-lines.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LineFilterConfig:
    min_height_px: int = 20
    min_width_px: int = 80
    min_area_px2: int = 1600
    max_lines_per_page: int = 60

def filter_lines(lines: list[dict], cfg: LineFilterConfig | None = None) -> list[dict]:
    """
    Filter a list of line dicts (each with 'bbox': [x, y, w, h]) to remove:
    - Lines with h < min_height_px (dots, noise)
    - Lines with w < min_width_px (fragments, partial detections)
    - Lines with w*h < min_area_px2 (tiny blobs)
    - Truncate to max_lines_per_page if exceeded (log warning)
    Returns filtered list. Never raises.
    """
```

### 1-C  Apply in generate_line_crops.py

After getting `seg.lines` from blla, apply `filter_lines()` before processing.
Import config values from `utils/config.py`.

### 1-D  Apply in accept-lines endpoint

Same filter applied to the user-submitted lines before saving.
User-drawn lines (from "Add line box" mode, identified by `id` starting with
`manual_`) bypass the size filter — the user explicitly drew them.

### 1-E  Apply in segment endpoint

In `/api/import/segment/{session_id}/{page_id}`, filter the blla output before
returning to the frontend. The user sees only clean line detections.

**Tests — `tests/test_seg_filter.py`:**
```python
def test_removes_short_lines(): ...      # h=5 → filtered out
def test_removes_narrow_lines(): ...     # w=30 → filtered out  
def test_keeps_valid_lines(): ...        # h=30, w=200 → kept
def test_manual_lines_bypass_filter(): ...  # id="manual_001" → kept regardless
def test_max_lines_cap(): ...            # 80 lines → truncated to 60
```

Smoke-check: `pytest tests/test_seg_filter.py -v`
Wait for confirmation before Step 2.

---

## Step 2 — LineReviewView fixes

**File: `frontend/src/components/LineReviewView.jsx` only**

### 2-A  Next button skip bug

Investigate and fix. The likely cause: when saving a correction, the index
advances past lines that were already skipped or marked unreadable, jumping
more than one position. The correct behavior:

- "Save & Next" → save current, advance to next PENDING line
  (status: neither corrected nor skipped)
- If no more pending: show "All lines done" message
- "Skip" → mark as skipped, advance to next pending
- Arrow keys → navigate ALL lines (including already corrected/skipped)
- The progress bar shows corrected+skipped/total

Fix: check the `findNextPending(currentIndex)` logic. It should scan forward
from `currentIndex + 1` and wrap around once. If it returns the same index,
all lines are done.

### 2-B  Draw-to-replace broken lines

Add a "✏️ Fix line" button in the right panel, visible only when a line is
selected. When clicked:

1. Enter "draw replacement" mode — cursor becomes crosshair on the manuscript image
2. User draws a rectangle over the correct line region (drag on image)
3. On mouseup: the drawn rect defines a new line
4. POST to a new endpoint `/api/lines/{page_id}/replace-line` with:
   ```json
   {
     "old_line_index": 12,
     "new_bbox": [x, y, w, h],
     "replace_adjacent": true
   }
   ```
5. The endpoint:
   - Crops the new bbox from the raw page image
   - Saves as the replacement line PNG
   - Runs OCR on the preprocessed page image for the new bbox region
   - Updates `lines.json` — replaces the old line entry with the new one
   - Returns the new line with updated `ocr_text`
6. Frontend updates the line list with the new OCR text pre-filled
7. User can then correct the text and save

The "replace_adjacent" flag: if true, also remove any other lines whose bbox
overlaps significantly (>50% overlap) with the new_bbox. This handles the case
where one real line was split into 3 fragments — draw over all of them and they
all get replaced by the single new line.

### 2-C  Line number display

Lines are numbered by their Y-coordinate order on the page (top to bottom),
not array index. This is already how they're stored in lines.json (sorted by
Y during generation). Display in header: "Line 12 / 65" where 12 = 1-based
position in Y-sorted order.

Add new API endpoint: `POST /api/lines/{page_id}/replace-line`

```python
# Body: {old_line_index: int, new_bbox: [x,y,w,h], replace_adjacent: bool}
# Action:
#   1. Load lines.json for page_id
#   2. If replace_adjacent: find all lines with >50% bbox overlap with new_bbox
#      and collect their indices
#   3. Remove all overlapping lines from the list
#   4. Crop new_bbox from raw image → save as line_{new_index:03d}.png
#   5. Run KrakenBackend on preprocessed image cropped to new_bbox
#   6. Insert new line entry at correct Y-sorted position
#   7. Renumber all lines sequentially
#   8. Save updated lines.json
# Returns: {new_line: {index, ocr_text, confidence, bbox}, total_lines: int}
```

**Tests — add to `tests/test_line_corrections.py`:**
```python
def test_replace_line_creates_new_entry(client): ...
def test_replace_line_removes_overlapping(client): ...
def test_replace_adjacent_false_keeps_others(client): ...
```

Smoke-check: `pytest tests/test_line_corrections.py -v`
Wait for confirmation before Step 3.

---

## Step 3 — Import tab fixes

**Files: `frontend/src/components/ImportView.jsx`, `api/routes.py`**

### 3-A  Delete page

In the page sidebar, add a ✕ button next to each page name.
On click: show confirmation "Delete this page?" with Yes/No.
On Yes: POST to `/api/import/delete-page/{session_id}/{page_id}`.
Removes the page from the session, deletes its files from `data/import/`.
If the deleted page was the active one, load the next available page.
If no pages remain, show the upload prompt again.

Add endpoint:
```python
# DELETE /api/import/pages/{session_id}/{page_id}
# Removes page image + segments from data/import/{session_id}/
# Returns: {deleted: page_id, remaining_pages: [...]}
```

### 3-B  Segmentation failed — diagnosis and fix

The "Segmentation failed" error needs to show the actual error message to help
diagnose. Change the error display in ImportView from a generic message to
showing the server's error detail.

Also add better error handling in the segment endpoint:
```python
@router.post("/api/import/segment/{session_id}/{page_id}")
async def import_segment(session_id: str, page_id: str, profile_name: str = "default"):
    try:
        # ... existing blla call ...
    except Exception as e:
        logger.error(f"Segmentation failed for {session_id}/{page_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
```

Additionally: check that `muharaf_seg_best.mlmodel` exists in `models/kraken/`
before attempting segmentation. If missing, return a clear 503 with instructions.

### 3-C  Line crop panel layout (Import tab)

Move the "Selected Line" detail panel from the right side to below the main
image area, full width, horizontal layout:

```
┌─────────────────────────────────────────────────────┐
│  Page image with line overlays (full width)          │
└─────────────────────────────────────────────────────┘
┌──────────────────┐  ┌──────────────┐  ┌────────────┐
│  [Auto-segment]  │  │  [Add line]  │  │ [Accept all│
│  [Accept sel.(N)]│  │              │  │            │
└──────────────────┘  └──────────────┘  └────────────┘
┌─────────────────────────────────────────────────────┐
│  SELECTED LINE (shown only when a line is selected)  │
│  [3× line crop image]  Line N of M  [✓ Accept][✗ X] │
└─────────────────────────────────────────────────────┘
```

The right panel disappears entirely. The bottom panel appears only when a line
is selected (conditional render). The page image takes full width of the content
area. The sidebar (page list) stays on the left.

---

## Step 4 — AnnotationView: move crop panel

**File: `frontend/src/components/AnnotationView.jsx` only**

The "WORD CROP (4×)" panel currently appears on the right side, taking space
from the manuscript image. Move it to below the toolbar buttons, same pattern
as Step 3-C.

Toolbar stays at top. Manuscript image takes full remaining width.
Below the image: WORD CROP panel + LABEL + buttons in a horizontal row.
This panel shows only when a token is selected.

No logic changes — only layout/CSS.

---

## Step 5 — Smoke-test

```bash
pytest tests/ -x -q 2>&1 | tail -10
cd frontend && npm run build
```

Manual checks (run the servers, open browser):

1. Load page 4.jpg in LineReviewView — count line boxes, verify no dot/noise lines
2. Press Next several times — verify it advances one line at a time, no skipping
3. Click "✏️ Fix line" on a fragmented line — verify draw mode activates
4. Draw a box over a line region — verify new OCR text appears
5. Open Import tab, upload an image, segment — verify no "failed" error
6. In Import tab: verify ✕ delete button on each page in sidebar
7. In Import tab: verify line crop panel appears below image not on right side
8. In AnnotationView: verify crop panel is below toolbar not on right side

Report all results. Commit and push.

---

## Compatibility notes

| Component | Interaction | Risk |
|---|---|---|
| `lines.json` format | replace-line rewrites it — must preserve all existing fields | Low — same format, just updated entries |
| `LineReviewView` save flow | replace-line adds a new action type — must not break existing Save/Skip | None — new button/mode only |
| `generate_line_crops.py` | seg_filter applied — existing line JSONs NOT regenerated automatically | None — only affects new imports |
| `seg_filter` in accept-lines | manual lines (id starts with "manual_") bypass filter | Explicitly handled |
| Import tab layout | right panel removed, bottom panel added — no API changes | Frontend only |
| AnnotationView layout | same — no logic changes | Frontend only |

---

## Non-negotiable rules

1. The replace-line endpoint must renumber ALL lines in lines.json sequentially
   after the replacement — gaps in line numbers are not allowed.
2. Manual lines drawn by the user (id prefix "manual_") always bypass
   the size filter — the user knows what they're drawing.
3. The next-button fix must handle wrap-around correctly — if the last
   line is reached, scan from the beginning for any remaining pending lines.
4. Delete page must show a confirmation dialog — no accidental deletions.
5. Segmentation error must show the actual error text, not a generic message.
6. Layout changes are CSS/JSX only — no logic, no API changes for steps 3-C and 4.
