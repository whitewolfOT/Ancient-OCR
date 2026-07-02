# Next Phase Plan
## Priorities, RASAM Fine-Tuning, and Tooling Decision

Read this fully before starting any sprint.

---

## Current state (verified)

- RALM oracle: built, working
- Import + Contribute tabs: built, browser-verified
- LineReviewView: draw-to-replace + next-button fix done
- Segmentation noise filter: built
- LINE_FIX Steps 3-5: status UNCONFIRMED — verify before assuming done
- 460 tests passing

## Two known open issues

1. Run OCR button in main workflow does not trigger POST /pages/{page_id}/ocr
2. RASAM + TariMa datasets discovered — 14,000 lines real handwritten Arabic

---

## External resources verdict

| Resource | Verdict | Reason |
|---|---|---|
| RASAM dataset | USE | 11,290 lines real handwritten Arabic MS, Apache 2.0, PageXML |
| TariMa dataset | USE | +2,673 lines same script family, same format |
| TariMa model (tested) | Reject | 0.911 vs Muharaf 0.938 on our manuscript |
| ScribeArabic | Reject | Django annotation app — we built better |
| lexical-databases (Calfa) | Reject | Armenian not Arabic |
| Venice catalog | Later | Manuscript metadata only |

We do NOT redesign. We feed their DATA into our existing pipeline only.

---

## Tooling — codebase-memory-mcp

Install it. Accelerates Step 0 exploration phase of every sprint.
MIT licensed, local-only, single binary. Modest but real token savings.
Install when convenient, not blocking.

---

## Priority order

### PRIORITY 1 — Fix Run OCR button (BLOCKING)
```
Read CLAUDE.md. Fix ONE bug: the Run OCR button in PageViewer.jsx or
WorkflowBar.jsx does not trigger POST /pages/{page_id}/ocr.
Logs show apply-cluster-settings then nothing after button click.

Step 0: Find the onClick handler. Trace what it calls. Report the handler
code and why the call isn't firing. Wait for confirmation.

Step 1: Fix. Verify browser network tab shows POST /pages/{page_id}/ocr
and OCR results render in Review panel. Commit and push.
Token-efficient mode.
```

### PRIORITY 2 — Verify LINE_FIX Steps 3-5
```
git log --oneline -10
```
If Steps 3-5 are committed, confirm with a browser test.
If not, resume from where it stopped.

### PRIORITY 3 — RASAM + TariMa fine-tuning (HIGH VALUE)

New Colab notebook:

```python
# Cell 1
!pip install kraken==7.0.2 -q

# Cell 2 — Download datasets
!git clone https://github.com/calfa-co/rasam-dataset.git /content/rasam
!git clone https://github.com/calfa-co/tarima.git /content/tarima
!find /content/rasam /content/tarima -name "*.xml" | wc -l

# Cell 3 — Build binary dataset from PageXML (native Kraken format)
from kraken.lib.arrow_dataset import build_binary_dataset
import glob
xml_files = (glob.glob('/content/rasam/**/*.xml', recursive=True) +
             glob.glob('/content/tarima/**/*.xml', recursive=True))
print(f'Total XML files: {len(xml_files)}')
build_binary_dataset(
    files=xml_files,
    output_file='/content/rasam_tarima.arrow',
    format_type='page',
    random_split=(0.9, 0.1, 0.0)
)

# Cell 4 — Get base model from repo
!git clone --filter=blob:none --sparse \
  --branch claude/adoring-carson-UqiLp \
  https://github.com/whitewolfOT/Ancient-OCR.git /content/repo
import os; os.chdir('/content/repo')
!git sparse-checkout set models/kraken && git checkout
!ls models/kraken/

# Cell 5 — Fine-tune (T4 GPU, 1-2 hours)
!ketos train \
  --load models/kraken/muharaf_rec_best.mlmodel \
  --output /content/muharaf_rasam \
  -f binary \
  --resize union \
  --lag 5 \
  --min-epochs 5 \
  /content/rasam_tarima.arrow

# Cell 6 — Download best model
import glob
from google.colab import files
models = sorted(glob.glob('/content/muharaf_rasam*.safetensors'))
print('Models:', models)
if models:
    files.download(models[-1])

# Cell 7 — A/B test on OUR manuscript
from kraken import blla, rpred
from kraken.lib import models as kraken_models
from kraken.lib.vgsl import TorchVGSLModel
from PIL import Image

img = Image.open('/content/repo/data/test_images/1.jpg').convert('RGB')
seg_model = TorchVGSLModel.load_model(
    '/content/repo/models/kraken/muharaf_seg_best.mlmodel')
seg = blla.segment(img, model=seg_model)

for label, path in [
    ('Muharaf original', '/content/repo/models/kraken/muharaf_rec_best.mlmodel'),
    ('RASAM fine-tuned', models[-1] if models else None),
]:
    if not path: continue
    rec = kraken_models.load_any(path)
    records = list(rpred.rpred(rec, img, seg))
    for i, r in enumerate(records[:2]):
        conf = sum(r.confidences)/len(r.confidences) if r.confidences else 0
        print(f'{label} Line {i}: conf={conf:.3f} | {r.prediction}')
    print()
```

After Colab:
- Upload new model to repo via git add -f + push
- ONLY adopt if it beats muharaf_rec_best on our manuscript
- Measure on OUR manuscript, not RASAM test set

### PRIORITY 4 — Deploy (after 1-3 complete)
Render (backend) + Vercel (frontend) so Contribute page is public.

---

## What NOT to do

- Do NOT redesign the pipeline
- Do NOT adopt ScribeArabic
- Do NOT assume RASAM data helps — measure first
- Do NOT build HATFormer/ByT5 yet
- Do NOT deploy until Run OCR works

---

## Success criteria

1. Run OCR button works end to end
2. LINE_FIX fully confirmed complete
3. RASAM model measured and adopted or rejected with data
4. Deployment decision based on measured accuracy
