# Sprint 1 — Line Review Interface
## Expert Correction Tool · Kraken Training Data Generator

Read CLAUDE.md fully before writing anything. Token-efficient mode throughout.
Each step has a smoke-check. Wait for confirmation before proceeding.

---

## What this builds

A fast, keyboard-driven line correction interface.
The expert sees one manuscript line at a time, corrects the OCR text, presses Enter.
Output is a Kraken-compatible training dataset (image + .gt.txt pairs).
This is the foundation for fine-tuning and for the future crowd platform.

---

## Step 0 — Generate line crops for all 9 pages

Run this script. Do not modify any existing files.

```python
# scripts/generate_line_crops.py
"""
For each of pages 1-9:
1. Run Kraken blla.segment() to get line boundaries
2. Crop each line from the original image using the boundary polygon
3. Save cropped line images to data/lines/{page_id}/line_{N:03d}.png
4. Save per-page JSON to data/lines/{page_id}/lines.json with structure:
   {
     "page": "1.jpg",
     "original_size": [w, h],
     "lines": [
       {
         "index": 0,
         "image_path": "data/lines/1.jpg/line_000.png",
         "ocr_text": "من الخضرة الى ...",   # from results.json
         "baseline": [[x,y], ...],
         "bbox": [x, y, w, h],
         "confidence": 0.945
       }
     ]
   }

Match OCR tokens to lines by finding which line's bbox contains each token's y-coordinate.
Concatenate matched tokens in RTL order (x descending) to form the line OCR text.
"""
```

After running: report page count, total lines, any pages with 0 lines.
Commit data/lines/ structure (not the images — they're large, add to .gitignore).
Add `data/lines/` to .gitignore but commit the JSON files only.
Wait for confirmation before Step 1.

---

## Step 1 — API endpoints for line corrections

Add to `api/routes.py` — three new routes only:

```python
# GET /api/lines/{page_id}
# Returns lines.json for the page with corrections merged in
# If a correction exists for a line, corrected_text overrides ocr_text
# Response: {page, lines: [{index, image_path, ocr_text, corrected_text, bbox, confidence, status}]}
# status: "pending" | "corrected" | "skipped"

# POST /api/lines/{page_id}/{line_index}/correction
# Body: {corrected_text: str, status: "corrected"|"skipped"}
# Saves to data/corrections/{page_id}/line_{N:03d}.txt
# Also saves metadata to data/corrections/{page_id}/corrections.json
# Returns: {saved, total_corrected, total_lines}

# GET /api/corrections/export
# Returns a ZIP file containing Kraken training format:
#   For each corrected line:
#     data/lines/{page_id}/line_NNN.png  (the line image)
#     data/lines/{page_id}/line_NNN.gt.txt  (the corrected text, UTF-8)
#   Plus a manifest.txt listing all pairs
# Content-Disposition: attachment; filename=training_data.zip
```

Add `data/corrections/` to .gitignore.

Tests in `tests/test_line_corrections.py`:
```python
def test_get_lines_returns_structure(client): ...
def test_save_correction_persists(client): ...
def test_export_zip_contains_pairs(client): ...
```

Smoke-check: `pytest tests/test_line_corrections.py -v`
Wait for confirmation before Step 2.

---

## Step 2 — LineReviewView.jsx

New file: `frontend/src/components/LineReviewView.jsx`
Wire into App.jsx as fourth view: `activeView === 'line-review'`
Add "📝 Correct Lines" button on upload screen and workspace header.

### Layout — full screen, two panels

```
┌──────────────────────────────────────────────────────────────┐
│ ← Back  [1.jpg ▼]  Line 3 / 18  ████████░░░░  5/18 done     │
│                              [← Prev] [Skip] [Next →]        │
├─────────────────────────┬────────────────────────────────────┤
│                         │                                    │
│  MANUSCRIPT PAGE        │  LINE IMAGE (3× zoom)             │
│  (left panel)           │  ┌──────────────────────────────┐ │
│                         │  │  Cropped line, white bg      │ │
│  Current line           │  │  Clear, high contrast        │ │
│  highlighted in         │  └──────────────────────────────┘ │
│  blue overlay           │                                    │
│                         │  OCR TEXT (editable)              │
│  Other lines            │  ┌──────────────────────────────┐ │
│  shown as thin          │  │ من الخضرة الى المضاو الى    │ │
│  grey lines             │  │ الصغرة وهذ التبدل وتكون     │ │
│                         │  └──────────────────────────────┘ │
│                         │  dir=rtl, Amiri font, 1.4rem      │
│                         │  Auto-selected on load            │
│                         │                                    │
│                         │  Confidence: 94.5%  ████████░░    │
│                         │                                    │
│                         │  [✓ Save & Next (Enter)]          │
│                         │  [→ Skip (Tab)]                   │
│                         │  [✗ Mark unreadable (U)]          │
│                         │                                    │
│                         │  ← / → : prev / next line         │
│                         │  Enter : save correction + next   │
│                         │  Tab : skip                       │
│                         │  U : mark unreadable              │
└─────────────────────────┴────────────────────────────────────┘
```

### Behavior rules

- On load: fetch `/api/lines/1.jpg`, show line 0, focus the text input
- Text input pre-filled with `corrected_text` if exists, else `ocr_text`
- Enter key: POST correction, advance to next pending line
- Tab key: POST {status: "skipped"}, advance
- U key: POST {corrected_text: "", status: "skipped"}, advance (marks unreadable)
- Left/right arrow: navigate without saving
- Progress bar: corrected + skipped / total
- Left panel: clicking a line on the page navigates to it
- Current line highlighted with semi-transparent blue overlay using bbox coords
- Line image loaded from `/images/{page_id}/line_{N:03d}.png` (add Vite proxy for `/lines/`)

Wait for confirmation before Step 3.

---

## Step 3 — Page selector + progress overview

Add to LineReviewView left panel header:
- Dropdown to switch pages (1.jpg – 9.jpg)
- Per-page progress: "3/18 lines corrected"
- Color coding: green = corrected, yellow = skipped, grey = pending

Add to the main App upload screen:
- If any corrections exist, show "Resume correction" with total progress
  e.g. "47 / 162 lines corrected across 9 pages"

---

## Step 4 — Export training data

"↓ Export training data" button in LineReviewView header.
Calls GET `/api/corrections/export` → downloads ZIP.
Show count: "Export 47 training pairs"

Also add a one-line instruction in the export ZIP's README.txt:
```
To fine-tune Kraken: ketos train -f alto *.gt.txt
See https://kraken.re/main/training.html
```

---

## Step 5 — Smoke-test

```bash
python scripts/generate_line_crops.py  # generates line JSONs
pytest tests/test_line_corrections.py -v
npm run build
```

Manual checks:
- LineReviewView loads with page image
- Current line highlighted on left panel
- Line crop visible on right
- OCR text pre-filled in input
- Enter saves and advances
- Progress bar updates
- Export downloads valid ZIP with .gt.txt files

Report all results. Commit and push.

---

## Non-negotiable rules

1. Line images served from disk — not base64 in JSON
2. RTL input always — dir=rtl, Amiri font, lang=ar
3. Keyboard-first — every action has a keyboard shortcut
4. Corrections saved to disk immediately — not localStorage
5. Export must be valid Kraken training format — .png + .gt.txt pairs
6. Do not break any existing views
7. Works gracefully if API unavailable — show cached corrections from localStorage as fallback
