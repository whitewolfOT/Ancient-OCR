# CLAUDE.md — Ancient-OCR (Arabic OCR + Lexicon-Augmented Intelligence)

This file tells Claude Code everything it needs to build and extend this project correctly.
Read it fully before writing any code. Follow the rules below without exception.

---

## What this system does

Ingests scanned Arabic PDFs/images → runs an OCR ensemble → normalizes noisy output → resolves uncertain words using classical Arabic lexicons + morphology → returns corrected Arabic text with full per-token provenance.

Three output modes: `clean` | `annotated` | `debug`.
Two interfaces: FastAPI API + CLI. No separate frontend required.

---

## Repo structure

```
/
├── CLAUDE.md                  ← you are here
├── main.py                    ← pipeline orchestrator (wire-up only)
├── config.yaml                ← all settings; no magic numbers in code
├── requirements.txt
├── pyproject.toml             ← optional extras: [layout], [trocr], [lm]
├── README.md
├── io/                        ← document_loader.py (PDF+image → pages)
├── preprocessing/             ← image_pipeline, deskew, thresholding, denoise, layout_detection
├── ocr_engine/                ← schema, base, paddle, tesseract, trocr, ensemble
├── alignment/                 ← token_matcher, bbox_alignment, string_similarity
├── normalization/             ← arabic_normalizer, noise_filter
├── morphology/                ← root_extractor, pattern_analyzer, camel_adapter
├── lexicon_ingestion/         ← sources, parser, storage, index_builder
├── lexicon_engine/            ← query_engine, candidate_generator, context_scorer, scorer, ranker
├── confidence_engine/         ← scoring, decision, state
├── output/                    ← formatter, json_export, markdown_export, debug_export
├── api/                       ← server.py, routes.py, schemas.py
├── cli/                       ← main.py, commands.py
├── utils/                     ← config.py, logging.py, cache.py
├── eval/                      ← metrics.py, benchmark.py
├── config/
│   ├── profiles.yaml          ← user-editable manuscript profiles (NEW)
├── ocr_engine/
│   ├── ...existing...
│   ├── kraken_backend.py      ← Kraken HTR engine, profile-driven (NEW)
│   ├── profile_loader.py      ← OCRProfile + ProfileManager (NEW)
├── preprocessing/
│   ├── ...existing...
│   ├── adjustments.py         ← brightness/contrast/gamma/stroke/denoise/sharpen (NEW)
├── align/
│   ├── __init__.py            ← new top-level package (NEW)
│   └── openiti.py             ← Passim / rapidfuzz alignment against Ibn al-Awwam (NEW)
├── ui/
│   ├── streamlit_app.py       ← 4-tab Streamlit frontend (NEW, [ui] extra)
│   └── api_client.py          ← single API call module for Streamlit (NEW)
├── tests/
├── data/                      ← lexicon raw + built indexes (gitignored)
└── models/                    ← downloaded/finetuned weights (gitignored)
```

---

## Build order — mandatory, do not skip phases

```
1.  Skeleton: all __init__.py, config.yaml, utils/config.py, utils/logging.py
2.  main.py orchestrator (safe imports — fail with clear message if module missing)
3.  Shared contracts: ocr_engine/schema.py + confidence_engine/state.py  ← build first, never change shape
4.  io/document_loader.py
5.  preprocessing/: denoise → thresholding → deskew → layout_detection → image_pipeline
6.  ocr_engine/: base → paddle_backend → tesseract_backend → trocr_backend
7.  alignment/: string_similarity → bbox_alignment → token_matcher
8.  ocr_engine/ensemble.py (uses alignment)
9.  normalization/: noise_filter → arabic_normalizer
10. lexicon_ingestion/: sources → parser → storage → index_builder
11. morphology/: root_extractor → pattern_analyzer → camel_adapter
12. lexicon_engine/: query_engine → candidate_generator → context_scorer → scorer → ranker
13. confidence_engine/: scoring → decision  (state already done in step 3)
14. output/: formatter → json_export → markdown_export → debug_export
15. api/: server → routes → schemas
16. cli/: main → commands
17. tests/ + eval/
18. README.md

Sprint V2 (after Phase 18):
0-A  WordToken schema extension + config.yaml additions
0-B  config/profiles.yaml + ocr_engine/profile_loader.py
0-C  preprocessing/adjustments.py + image_pipeline.py integration
1    ocr_engine/kraken_backend.py (profile-driven, Muharaf seg + Agapet rec)
2    lexicon_ingestion: ibn_awwam_filaha source + parser + ingest script
3    align/openiti.py (Passim/rapidfuzz post-scoring alignment)
4    output/review_export.py — Kraken baseline coordinate support
5    lexicon_engine/candidate_generator.py — weighted confusion costs
6    api/server.py — add /api/preview, /api/profiles, /api/suggest_profile routes
7    ui/streamlit_app.py + ui/api_client.py
8    Integration smoke-test
```

---

## Shared data contracts — define once, obey everywhere

Use **Pydantic v2**. All models must be JSON-serializable. Build these in step 3 and never change the field names downstream.

```python
class WordToken(BaseModel):
    text: str
    confidence: float           # 0..1
    bbox: tuple[int,int,int,int]  # x, y, w, h — always populated, page-space
    page_index: int
    source: str                 # "paddle"|"tesseract"|"trocr"|"kraken"|"ensemble"
    region_id: str | None = None
    line_id: str | None = None             # Kraken line UUID from segment JSON
    baseline: list[tuple[int, int]] | None = None  # raw baseline points, page-space (Kraken only)

class OCRResult(BaseModel):
    text: str
    words: list[WordToken]
    confidence: float
    page_index: int
    source: str
    raw: dict = {}              # engine-specific payload for debug

class LexiconEntry(BaseModel):
    lemma: str
    root: str | None
    pattern: str | None
    gloss: str
    source: str
    era: str                    # "classical" | "modern"
    domain: str | None
    examples: list[str] = []
    priority: int

class Candidate(BaseModel):
    text: str
    reason: str                 # "spelling_variant"|"normalization"|"root_alt"|"morph_alt"|"identity"
    lexicon_entries: list[LexiconEntry] = []
    features: dict = {}         # lexicon_score, morph_score, ocr_score, context_score
    score: float | None = None

class TokenState(BaseModel):
    original: str
    normalized: str
    normalization_log: list[dict]        # per-step changes (traceability)
    candidates: list[Candidate]
    selected: str
    confidence: float
    sources: list[str]
    decision: str               # "accept"|"accept_with_note"|"uncertain"|"review_required"
    reason_code: str

@dataclass
class AlignmentResult:
    corrected_text: str
    ocr_text: str
    confidence: float            # Levenshtein ratio of best match
    method: str                  # "passim" | "rapidfuzz"
    accepted: bool               # confidence >= config.align.passim.threshold
    match_start: int
    match_end: int
```

Rules for WordToken:
- `bbox` is always populated even for Kraken tokens. Compute from baseline polygon: `x=min_x, y=min_y, w=max_x−min_x, h=max_y−min_y`.
- `baseline` and `line_id` are None for Paddle/Tesseract/TrOCR tokens.

---

## Non-negotiable design rules

1. **Never overwrite a token without storing the original + alternatives.**
2. **Every correction must carry its source(s) and a reason code.**
3. **Lexicon evidence > morphology > model guessing.** Morphology supports; never overrules.
4. **Raw per-engine OCR stays accessible throughout** (needed for debug mode).
5. **Config drives everything.** No magic numbers in code — read from `config.yaml`.
6. **No side effects at import.** No model downloads, no DB connections at import time.
7. **Graceful degradation.** Missing optional dep/model/lexicon → log it, disable that signal, continue.
8. **Every module independently unit-testable** with no heavy dependencies.
9. Classical and modern lexicon evidence stay tagged and separable.
10. `review_required` and `uncertain` tokens are NEVER silently corrected. They go to the review queue and must be explicitly resolved by a human.
11. Feedback never auto-applies. Stored corrections require explicit calibration trigger (/calibrate endpoint or calibrator.py).
12. Profiles are additive configuration. Changing a profile changes output but never changes stored lexicon entries, TokenState records, or feedback data.
13. Alignment accepted results go into aligned_text on the page result only. They never overwrite TokenState.selected.

---

## Pipeline order (canonical — main.py must follow this exactly)

```
input file
 → io/document_loader          (PDF/PNG/JPG/TIFF → page images)
 → preprocessing/image_pipeline
 → ocr_engine/ensemble         (Paddle primary; Tesseract + TrOCR optional)
 → alignment (inside ensemble)
 → normalization/arabic_normalizer + noise_filter
 → morphology/camel_adapter (+ fallbacks)
 → lexicon_engine/query_engine + candidate_generator + context_scorer + scorer + ranker
 → confidence_engine/scoring + decision
 → output/formatter + exporter
```

---

## Confidence model

```python
final = 0.30*ocr_confidence + 0.30*lexicon_score + 0.20*morphology_score + 0.20*context_score
```

Decision thresholds (from `config.decision`):

| Score | Label |
|---|---|
| >= 0.90 | accept |
| 0.70–0.89 | accept_with_note |
| 0.50–0.69 | uncertain |
| < 0.50 | review_required |

---

## Key technical constraints

**OCR engines:**
- PaddleOCR is primary (`lang="ar"`) but quality on classical/Ottoman/Quranic type is limited — treat it as one signal.
- Tesseract requires system packages `tesseract-ocr` + `tesseract-ocr-ara`. Detect absence and fail clearly.
- TrOCR has **no official Arabic model**. Use a community fine-tuned checkpoint configured in `config.yaml`. Its role is **line-level re-recognition** of low-confidence crops, not bbox token extraction. Make it fully optional.

**Normalization:**
- `normalize_text()` returns `(text, change_log)`. The change log per token IS the traceability — there is no lossless global inverse.
- `reverse_normalization_map()` documents the folding rules only.

**Layout detection:**
- Default is pure-OpenCV projection-profile. `layoutparser` (detectron2) is an **optional extra** `[layout]`. Never make it a hard dependency.

**Lexicon sources:**
- OpenITI/Shamela public-domain dumps for Lisān al-ʿArab, Tāj al-ʿArūs, al-Qāmūs, Lane's Lexicon.
- **DO NOT scrape Almaany** (ToS). Keep a disabled stub entry in `sources.py`.
- Ship a tiny synthetic fixture lexicon at `data/lexicons/_fixture/` so the full pipeline runs with zero external data.

**CAMeL Tools:**
- Lazy-load the morphology DB (`camel_data -i morphology-db-msa-r13`).
- If absent → rule-based fallback in `root_extractor.py` + `pattern_analyzer.py`.

**context_scorer.py:**
- This module MUST exist and produce a `float` in [0,1]. Default: deterministic n-gram/co-occurrence.
- Optional masked-LM (AraBERT) behind `[lm]` extra + config flag.
- Without it the confidence formula silently zeroes out a 20% weight — that is not acceptable.

**Profiles:**
- All processing parameters live in config/profiles.yaml, not config.yaml.
- config.yaml only holds: profiles.enabled, profiles_file, active_profile.
- "default" profile is protected — cannot be deleted.
- profile_loader.ProfileManager is instantiated lazily (not at import time).
- Every backend, preprocessing call, and API route that touches processing parameters must accept an OCRProfile, not bare config values.

**Kraken:**
- Kraken backend is optional — if not installed, is_available() returns False, pipeline continues.
- Binarizer choice (nlbin/sauvola/otsu) is backend's job, not image_pipeline.py's.
- nlbin is Kraken-internal and must not be called outside the Kraken workflow.
- General preprocessing (brightness, contrast, gamma, etc.) runs in image_pipeline.py before the backend is called.
- Kraken models download lazily inside process_image(), never in __init__().

**Streamlit UI:**
- Lives in ui/. Optional dependency group [ui] = ["streamlit>=1.35", "Pillow>=10"].
- All API calls go through ui/api_client.py — never inline requests calls.
- API base URL from API_BASE_URL env var, default http://localhost:8000.

**Passim alignment:**
- Alignment is a post-scoring layer only. Never called from lexicon_engine/.
- Called from main.py after confidence_engine/decision.py, before output/formatter.py.
- Results go into aligned_text on the page result. They NEVER overwrite TokenState.selected.
- Fallback to rapidfuzz if Java absent. Skip entirely if both absent.

---

## Running the system

```bash
# CLI
python -m cli.main process input.pdf --mode annotated
python -m cli.main process input.png --mode debug
python -m cli.main batch ./folder/ --mode clean

# API
uvicorn api.server:app --reload
# POST /process  (multipart file + mode param)

# Tests
pytest tests/

# Benchmark
python -m eval.benchmark --data ./data/benchmark/

# Install UI dependencies
pip install ".[ui]" --break-system-packages

# Start Streamlit (in a second terminal)
streamlit run ui/streamlit_app.py

# Run OCR with a named profile
python -m cli.main process input.pdf --mode annotated --profile andalusian_naskh

# Ingest Ibn al-Awwam lexicon (one-time, requires network)
python scripts/ingest_ibn_awwam.py
```

---

## Adding a new lexicon source

1. Add an entry to `lexicon_ingestion/sources.py`.
2. Write a parser adapter in `lexicon_ingestion/parser.py`.
3. Run the ingestion script to populate `data/lexicons/<source>/` and rebuild indexes.
4. Enable in `config.yaml` under `lexicon.sources`.

No other files need changing.

Agricultural domain sources follow the same 4-step process. ibn_awwam_filaha is already added. To add another OpenITI text:
1. Add LexiconSource entry in sources.py with correct OpenITI GitHub raw URL.
2. Reuse parse_openiti_markdown() in parser.py (it handles all OpenITI mARkdown).
3. Run ingest script.
4. Enable in config.yaml.
Note: OpenITI texts produce vocabulary coverage entries only (no gloss/root).

---

## Profile system

Profiles are named sets of processing parameters in config/profiles.yaml.
A profile controls: binarizer choice, seg/rec model handles, N-best count,
RTL setting, device, and all preprocessing adjustments.

Built-in profiles:
- default            — printed Arabic, clean scan
- andalusian_naskh   — Ibn al-Awwam manuscript, Agapet primary
- maghrebi_degraded  — Sauvola binarizer, stroke normalization on
- low_contrast       — high brightness/contrast boost

Users add custom profiles via the Streamlit UI (Tab 1) or by editing
config/profiles.yaml directly. The "default" profile cannot be deleted.

Profile selection in CLI: --profile <name>
Profile selection in API: profile_name form field on all processing endpoints.
Profile creation/editing in UI: Profile Manager tab.
Live preview of profile effects: Live Preview tab (sliders without saving).

---

## Full design reference

See `docs/MASTER_PLAN.md` for the complete annotated architecture, the rationale behind every design decision, the data-acquisition and licensing table, and a full risk register.
