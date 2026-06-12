# Drawing Annotation Sprint
## Interactive Transcription — Draw on Manuscript to Generate Training Data

Read CLAUDE.md fully before writing anything. Then start Step 0.

---

## What this builds

A drawing annotation tool embedded in the React frontend.
User sees the manuscript page, clicks a word region, draws over the ink
to trace the correct letterforms, types the label, saves the training pair.

Output per correction:
- `data/training_pairs/1.jpg/patch_003.png` — cropped word image patch
- `data/training_pairs/1.jpg/patch_003.txt` — correct Arabic text label
- `data/training_pairs/manifest.json` — index of all pairs

This is the ground truth dataset that fine-tunes Muharaf.

---

## Step 0 — Check what's available

Run:
```bash
ls frontend/src/components/
cat frontend/public/results.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
pages=list(d.values()) if isinstance(d,dict) else d
p=pages[0]
print('page keys:', list(p.keys()))
print('first token:', json.dumps((p.get('words') or p.get('tokens') or [])[0], ensure_ascii=False))
"
ls data/test_images/
```

Report: component list, token structure, available images. Wait for confirmation.

---

## Step 1 — API endpoint for saving training pairs

Add to `api/routes.py`:

```python
@app.post("/api/training-pairs")
async def save_training_pair(
    page_id: str = Form(...),        # e.g. "1.jpg"
    token_index: int = Form(...),
    label: str = Form(...),          # correct Arabic text
    patch_b64: str = Form(...),      # base64 PNG of the drawn annotation
    original_bbox: str = Form(...),  # JSON string of [x,y,w,h]
):
    """
    Save a training pair: drawn annotation image + label.
    patch_b64 is the composite of original image patch + user drawing.
    Saves to data/training_pairs/{page_id}/patch_{token_index:04d}.png
    and data/training_pairs/{page_id}/patch_{token_index:04d}.txt
    Updates data/training_pairs/manifest.json
    Returns: {status, pair_id, total_pairs}
    """
```

Rules:
- Create directories as needed
- manifest.json format:
  ```json
  {
    "pairs": [
      {
        "id": "1.jpg/patch_0003",
        "page": "1.jpg",
        "token_index": 3,
        "label": "الخضرة",
        "patch_path": "data/training_pairs/1.jpg/patch_0003.png",
        "bbox": [1158, 46, 77, 18],
        "created_at": "ISO timestamp"
      }
    ],
    "total": 1
  }
  ```
- If pair for same page+token_index already exists, overwrite it
- Return 200 with pair count

Add `GET /api/training-pairs` to return the manifest.
Add `GET /api/training-pairs/count` to return just the total count.

Also add a static file mount so the frontend can load manuscript images directly:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory="data/test_images"), name="images")
```

Tests in `tests/test_training_pairs.py`:
```python
def test_save_pair_creates_files(client, tmp_path): ...
def test_save_pair_updates_manifest(client): ...
def test_overwrite_existing_pair(client): ...
def test_get_manifest_returns_list(client): ...
```

Smoke-check: `pytest tests/test_training_pairs.py -v`
Wait for confirmation before Step 2.

---

## Step 2 — AnnotationView.jsx (new top-level view)

New file: `frontend/src/components/AnnotationView.jsx`

This is a full-screen annotation workspace. Wire it into App.jsx as a third
view alongside 'workspace' and 'review': `activeView === 'annotate'`
Add "✏️ Annotate" button next to "View OCR results" on the upload screen
and in the workspace header.

### Layout (full screen, dark background)

```
┌─────────────────────────────────────────────────────────────┐
│ ← Back   1.jpg   Token 3/196   [< Prev word] [Next word >]  │
│                          ✓ 12 pairs saved   ↓ Export pairs  │
├────────────────────────┬────────────────────────────────────┤
│                        │  WORD DETAIL                       │
│   MANUSCRIPT PAGE      │  ┌──────────────────────────┐     │
│   (scrollable,         │  │  Zoomed word crop  4×    │     │
│    zoomable)           │  │  (original image patch)  │     │
│                        │  └──────────────────────────┘     │
│  Word boxes overlaid   │                                    │
│  as colored rectangles │  DRAW CORRECTION                  │
│  Click = select        │  ┌──────────────────────────┐     │
│                        │  │  Canvas same size as crop │     │
│  Selected word:        │  │  Shows image underneath  │     │
│  highlighted blue      │  │  User draws in red ink   │     │
│  bbox                  │  │  on top of original      │     │
│                        │  └──────────────────────────┘     │
│                        │  [Clear drawing]                   │
│                        │                                    │
│                        │  LABEL (Arabic text)               │
│                        │  ┌──────────────────────────┐     │
│                        │  │  RTL Arabic input        │     │
│                        │  │  dir=rtl, Amiri font     │     │
│                        │  └──────────────────────────┘     │
│                        │                                    │
│                        │  OCR suggestion: الخضرة            │
│                        │  [Use OCR text ↑]                  │
│                        │                                    │
│                        │  [💾 Save pair (Enter)]            │
│                        │  [Skip →]                          │
└────────────────────────┴────────────────────────────────────┘
```

### Manuscript image display

Load image from `/images/{pageId}` (the static mount from Step 1).
Display at full width of the left panel, scaled to fit.
Overlay word bboxes as semi-transparent colored rectangles.
Click a box → selects that token, shows it in the right panel.
Selected box has blue border + highlight.
Scale factor: compute `displayWidth / originalWidth` to map bbox coords
to screen coords correctly.

### Drawing canvas (RIGHT PANEL)

The drawing canvas is the key feature.

1. Show the zoomed word image patch as background (4× zoom of original bbox crop)
2. Canvas overlay on top, same dimensions, transparent background
3. User draws with mouse/trackpad — red ink, 2px stroke, smooth curves
4. Use `requestAnimationFrame` for smooth drawing
5. On mousedown: start path
6. On mousemove + button held: continue path
7. On mouseup: end path
8. "Clear drawing" button: clears canvas strokes only, keeps image background

On save:
- Composite: draw image background first, then the red strokes on top
- Export as PNG base64
- Send to `/api/training-pairs`

### Label input

- `dir="rtl"` Arabic input
- Pre-filled with OCR suggestion (`token.text`)
- "Use OCR text ↑" button: fills input with OCR text
- Clear when switching tokens (unless already saved)

### Save behavior

On "Save pair":
1. Composite canvas → base64 PNG
2. POST to `/api/training-pairs` with label + patch + bbox
3. Show brief success flash ("✓ Saved") 
4. Auto-advance to next token
5. Increment saved counter in header

On "Skip": advance to next token without saving.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| Enter | Save pair + advance |
| S | Save pair + advance |
| Space | Skip |
| ← → | Prev/next token |

---

## Step 3 — Page selector in AnnotationView

Left panel header: dropdown to switch between pages (1.jpg through 9.jpg).
When switching pages, reset token index to 0, clear canvas, clear label.
Show saved pair count per page next to each page name in dropdown.

---

## Step 4 — Export training data

"↓ Export pairs" button in header:
- GET `/api/training-pairs` → download manifest.json
- Also offer "Download all patches as ZIP" — use JSZip if available,
  otherwise just download manifest and instruct user to use the API

Add to `api/routes.py`:
```python
@app.get("/api/training-pairs/export")
async def export_training_pairs():
    """
    Returns a ZIP file containing:
    - manifest.json
    - all patch PNG files
    """
    import zipfile, io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        # add manifest and all patches
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=training_pairs.zip"})
```

---

## Step 5 — Smoke-test

```bash
npm run build
pytest tests/test_training_pairs.py -v
```

Manual check:
- AnnotationView loads with manuscript image
- Word boxes visible and clickable
- Drawing on canvas works
- Save creates files in data/training_pairs/
- Export ZIP downloads
- Counter increments on save

Report all results. Commit and push.

---

## Non-negotiable rules

1. Drawing canvas composite must include BOTH the original image patch AND
   the user's strokes — not strokes alone. The image context is essential
   for the training pair to be useful.
2. RTL throughout all Arabic text inputs and displays.
3. API saves to disk — not localStorage. These are permanent training pairs.
4. Scale factor must be computed correctly — bbox coords are in original
   image space, display is scaled. Wrong scale = wrong crops.
5. Works without the API running for viewing — gracefully shows
   "API not available, pairs will not be saved" if fetch fails.
6. Do not break ReviewTab or existing views.
