# Arabic OCR + Lexicon-Augmented Intelligence — Master Build Plan

**Audience:** Claude Code (and human reviewer)
**Status:** Consolidated and corrected from three source drafts (`ocr1`, `ocr_2`, `ocr3`).
**Goal:** A production-grade, deterministic-first pipeline that ingests scanned Arabic PDFs/images, runs an OCR ensemble, normalizes noisy output, resolves uncertain words using morphology + multiple Arabic lexicons, and returns corrected Arabic text with full, traceable evidence per token.

---

## 0. How to use this document

1. Read §1–§9 first. They define the architecture, the build order, and the **shared data contracts** every module must obey. Most integration bugs in a project like this come from modules disagreeing on the shape of `WordToken` / `OCRResult` / `TokenState` — so those contracts are fixed here, once, in §9.
2. Build strictly in the order given in §5.
3. Each file's spec in §10 is a contract (inputs → outputs → public functions), not loose prose. Implement to the contract.
4. The non-negotiable design rules in §8 override convenience. When in doubt, prefer the rule.
5. §14 is a ready-to-paste kickoff prompt for a fresh Claude Code session.

---

## 1. System overview

**Pipeline (canonical order):**

```
input file (PDF/PNG/JPG/TIFF)
  → document loading + page splitting
  → preprocessing (per page)
  → OCR ensemble (Paddle + Tesseract + TrOCR verify)
  → cross-engine token alignment + merge
  → conservative Arabic normalization (lossy steps recorded per token)
  → morphological analysis (CAMeL Tools, rule-based fallback)
  → lexicon evidence retrieval (classical > modern, with provenance)
  → candidate generation → scoring → ranking
  → confidence scoring → decision label
  → output formatting (clean | annotated | debug)
```

**Three output modes:**
- `clean` — corrected Arabic text only.
- `annotated` — text + per-token `{original, corrected, confidence, sources, alternatives, decision}`.
- `debug` — full trace: raw per-engine OCR, preprocessing metadata, alignment, normalization deltas, morphology, lexicon candidates, scores, decisions.

**Core philosophy:** deterministic and explainable first; learned/neural calibration later. Lexicon evidence outranks model guessing. Never silently overwrite an uncertain token.

---

## 2. Analysis of the source drafts — what was merged, fixed, and added

The three drafts are consistent on architecture and module names, so the module structure below is theirs. The following are the **corrections and additions** made while consolidating (these are the substantive changes — everything else is faithful reformatting):

**Conflicts resolved**
- **Morphology vs. lexicon ordering.** `ocr1`'s phase list builds morphology *after* lexicon ingestion; `ocr3`'s build order builds morphology *before* lexicon ingestion. Resolved: build **lexicon ingestion → morphology → lexicon engine**. Ingestion only writes static data; morphology is needed *before* the lexicon *engine* because candidate generation and the root index both consume morphology. (See §5.)
- **`reverse_normalization_map()` is not losslessly invertible.** Several normalization steps (alef-variant folding, taa-marbuta folding) are many-to-one and cannot be reversed from the normalized string alone. Corrected design: do **not** promise a global reverse map. Instead each token carries its own `original` alongside `normalized`, and normalization returns a per-step change log. `reverse_normalization_map()` is kept only as a *documentation/debug* helper listing the folding rules, not a real inverse.

**Gaps filled (new files added — marked ➕ in §10)**
- ➕ **Document loader.** All drafts say "accept PDF/PNG/JPG/TIFF, split PDFs into pages" but no module does it, and `main.py` is told not to. Added `io/document_loader.py` (PyMuPDF/pdf2image → list of page images).
- ➕ **Context scorer.** Every draft lists `context_score` / "sentence coherence" as a scoring input, but **no module produces it**. Added `lexicon_engine/context_scorer.py` (deterministic n-gram fallback; optional masked-LM AraBERT scorer behind a flag).
- ➕ **Cache layer.** `config.yaml` has caching options but no cache module. Added `utils/cache.py`.
- ➕ **Evaluation harness.** `ocr1` phase 11 specifies a benchmark set and metrics (CER/WER, root accuracy) but there is no eval code. Added `eval/benchmark.py` + `eval/metrics.py`.
- ➕ **Logging utility.** Referenced everywhere; centralized in `utils/logging.py`.

**Technical caveats made explicit (do not skip — see §13)**
- **PaddleOCR Arabic** quality on classical/Ottoman/Quranic typography is weak; treat it as *a* signal, not ground truth.
- **TrOCR has no official Arabic model.** It must be a community/fine-tuned checkpoint, and its role is **line-level re-recognition of low-confidence regions**, not per-token bbox extraction. The "verifier" framing in the drafts is corrected accordingly.
- **Lexicon data sourcing & licensing** is unspecified in the drafts and is the single biggest real-world risk. §7 defines where each lexicon actually comes from and which to drop. **Almaany must not be scraped.**
- `layoutparser` pulls heavy/fragile deps (detectron2). It is moved to an optional extra with a pure-OpenCV fallback as default.

---

## 3. Corrected architecture & module map

| Layer | Package | Responsibility |
|---|---|---|
| Ingestion | `io/` | Load files, split PDFs to page images, classify page quality |
| Preprocess | `preprocessing/` | grayscale, contrast, denoise, deskew, threshold, optional layout |
| OCR | `ocr_engine/` | Backend interface + Paddle/Tesseract/TrOCR + ensemble merge |
| Alignment | `alignment/` | bbox IoU matching, fuzzy text matching, token grouping |
| Normalization | `normalization/` | conservative Arabic folding + noise removal (change-logged) |
| Morphology | `morphology/` | root extraction, wazn detection, CAMeL adapter |
| Lexicon data | `lexicon_ingestion/` | sources registry, parser, index builder, storage |
| Lexicon engine | `lexicon_engine/` | query, candidate gen, **context scorer**, scorer, ranker |
| Confidence | `confidence_engine/` | weighted scoring, decision labels, per-token state |
| Output | `output/` | formatter + json/markdown/debug exporters |
| Interfaces | `api/`, `cli/` | FastAPI server + Typer/argparse CLI |
| Support | `utils/` | config loader, logging, cache |
| Quality | `tests/`, `eval/` | unit tests + benchmark harness |
| Orchestration | `main.py`, `config.yaml` | wire everything; single source of pipeline order |

---

## 4. Repository tree

```
ocr-arabic-system/
├── main.py
├── config.yaml
├── requirements.txt
├── pyproject.toml            # optional extras: [layout], [trocr], [lm]
├── README.md
├── io/
│   ├── __init__.py
│   └── document_loader.py    ➕
├── preprocessing/
│   ├── __init__.py
│   ├── image_pipeline.py
│   ├── deskew.py
│   ├── thresholding.py
│   ├── denoise.py
│   └── layout_detection.py
├── ocr_engine/
│   ├── __init__.py
│   ├── schema.py
│   ├── base.py
│   ├── paddle_backend.py
│   ├── tesseract_backend.py
│   ├── trocr_backend.py
│   └── ensemble.py
├── alignment/
│   ├── __init__.py
│   ├── token_matcher.py
│   ├── bbox_alignment.py
│   └── string_similarity.py
├── normalization/
│   ├── __init__.py
│   ├── arabic_normalizer.py
│   └── noise_filter.py
├── morphology/
│   ├── __init__.py
│   ├── root_extractor.py
│   ├── pattern_analyzer.py
│   └── camel_adapter.py
├── lexicon_ingestion/
│   ├── __init__.py
│   ├── sources.py
│   ├── parser.py
│   ├── index_builder.py
│   └── storage.py
├── lexicon_engine/
│   ├── __init__.py
│   ├── query_engine.py
│   ├── candidate_generator.py
│   ├── context_scorer.py     ➕
│   ├── scorer.py
│   └── ranker.py
├── confidence_engine/
│   ├── __init__.py
│   ├── scoring.py
│   ├── decision.py
│   └── state.py
├── output/
│   ├── __init__.py
│   ├── formatter.py
│   ├── json_export.py
│   ├── markdown_export.py
│   └── debug_export.py
├── api/
│   ├── __init__.py
│   ├── server.py
│   ├── routes.py
│   └── schemas.py
├── cli/
│   ├── __init__.py
│   ├── main.py
│   └── commands.py
├── utils/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py            ➕
│   └── cache.py              ➕
├── eval/                     ➕
│   ├── __init__.py
│   ├── benchmark.py
│   └── metrics.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_ocr.py
│   ├── test_alignment.py
│   ├── test_normalization.py
│   ├── test_lexicon.py
│   ├── test_morphology.py
│   └── test_pipeline.py
├── data/                     # lexicon raw + built indexes (gitignored)
└── models/                   # downloaded/finetuned OCR + LM weights (gitignored)
```

---

## 5. Canonical build order (mandatory)

Build and verify each phase before starting the next. Each phase must have passing tests or a runnable smoke check.

1. **Skeleton** — package dirs, all `__init__.py`, `config.yaml`, `utils/config.py`, `utils/logging.py`.
2. **`main.py` orchestrator** — with *safe imports* and stub-friendly wiring (modules may not exist yet; fail with a clear message naming the missing module).
3. **Shared contracts** — `ocr_engine/schema.py` and `confidence_engine/state.py` (everything downstream depends on these; build them early). See §9.
4. **Document loader** — `io/document_loader.py`.
5. **Preprocessing** — `denoise` → `thresholding` → `deskew` → `layout_detection` → `image_pipeline` orchestrator.
6. **OCR backends** — `base.py` → `paddle_backend` → `tesseract_backend` → `trocr_backend`. Each must fail gracefully if its dependency/model is absent.
7. **Alignment** — `string_similarity` → `bbox_alignment` → `token_matcher`.
8. **OCR ensemble** — `ensemble.py` (consumes alignment).
9. **Normalization** — `noise_filter` → `arabic_normalizer`.
10. **Lexicon ingestion** — `sources` → `parser` → `storage` → `index_builder`. (Static data; produces `data/` indexes.)
11. **Morphology** — `root_extractor` → `pattern_analyzer` → `camel_adapter`.
12. **Lexicon engine** — `query_engine` → `candidate_generator` → `context_scorer` → `scorer` → `ranker`.
13. **Confidence engine** — `scoring` → `decision` (state already built in step 3).
14. **Output** — `formatter` → `json_export` → `markdown_export` → `debug_export`.
15. **API + CLI** — `api/*`, `cli/*`.
16. **Tests + eval harness** — fill `tests/`, build `eval/`.
17. **README** — write last, when the surface is stable.
18. *(Optional, post-baseline)* semantic reconstruction / missing-word restoration. Do **not** build before the baseline is stable.

---

## 6. Dependencies & environment

`requirements.txt` (core, conservative pins — verify latest compatible at build time):

```
# image + io
opencv-python-headless>=4.9
pillow>=10.0
numpy>=1.26
pymupdf>=1.24          # PDF -> images (fitz); preferred over pdf2image (no poppler dep)

# ocr
paddleocr>=2.7
paddlepaddle>=2.6      # CPU build by default
pytesseract>=0.3.10    # requires system `tesseract-ocr` + `tesseract-ocr-ara`
transformers>=4.40     # TrOCR (optional model)
torch>=2.2             # TrOCR backend only

# arabic nlp
camel-tools>=1.5       # run `camel_data -i morphology-db-msa-r13` after install
python-Levenshtein>=0.25

# api + cli
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
typer>=0.12

# utils + tests
pyyaml>=6.0
pydantic>=2.6
diskcache>=5.6
pytest>=8.0
```

**Optional extras (keep out of core to avoid dependency hell):**
- `[layout]` → `layoutparser` (+ detectron2). Heavy/fragile; the default layout detector is pure OpenCV projection-profile, used unless this extra is installed.
- `[trocr]` → a specific Arabic TrOCR checkpoint (community/fine-tuned; pin the model id in config).
- `[lm]` → `arabert` / a small KenLM model for the optional masked-LM context scorer.

**System packages:** `tesseract-ocr`, `tesseract-ocr-ara`. Document these in the README; the Tesseract backend must detect their absence and fail with a clear message.

**Determinism caveat:** neural backends (Paddle/TrOCR) are not bit-reproducible across hardware. Set seeds where possible and document that *debug determinism* applies to the deterministic stages (alignment, normalization, lexicon, scoring, decision), not raw neural OCR.

---

## 7. Data acquisition & licensing (critical — drafts omitted this)

The lexicon layer is the heart of the system and the biggest real-world risk. Define provenance and format up front. Store everything under `data/lexicons/<source>/`.

| Source | Era | Realistic acquisition | License posture | Action |
|---|---|---|---|---|
| Lane's Lexicon (Arabic–English) | Classical | Public-domain digitizations (e.g. StudyQuran / Perseus / community JSON) | Public domain | Include |
| Lisān al-ʿArab | Classical | OpenITI / Shamela corpus dumps (structured text) | Mostly PD text; check corpus license | Include |
| Tāj al-ʿArūs | Classical | OpenITI / Shamela | PD text; check corpus license | Include |
| al-Qāmūs al-Muḥīṭ | Classical | OpenITI / Shamela | PD text; check corpus license | Include |
| Arabic WordNet | Modern | Official AWN distribution | Research license — review terms | Include if license permits |
| Almaany | Modern | **Website only, no open dataset** | ToS prohibits scraping | **Exclude.** Do not scrape. Leave a disabled stub `source` entry. |

**Rules:**
- `sources.py` declares each source with `{name, era: classical|modern, priority, domain, license, enabled, path}`. Classical sources get higher `priority` than modern (per the brief).
- The parser converts each raw source into the canonical `LexiconEntry` (§9) via a per-source adapter, so adding a source later means adding one adapter — never touching the engine.
- Ship a tiny **synthetic fixture lexicon** (`data/lexicons/_fixture/`) so the whole pipeline and tests run with **zero** external data. Real sources are opt-in via config.

---

## 8. Global design rules (non-negotiable)

1. Never overwrite a token without storing the original and the alternatives.
2. Every correction must carry its source(s) and a reason code.
3. All decisions must be explainable and deterministic in the deterministic stages.
4. **Lexicon evidence > morphology > model guessing.** Morphology supports correction; it never overrules lexicon evidence.
5. Keep raw per-engine OCR output accessible through the whole pipeline (needed for debug mode).
6. Classical and modern lexicon evidence stay tagged and separable, but are mergeable.
7. Every module is independently importable and unit-testable, with no import-time side effects (no model downloads at import).
8. Config drives behavior; no magic numbers in code — read weights/thresholds/toggles from `config.yaml`.
9. Graceful degradation: a missing optional backend/model/lexicon disables that signal and logs it; it never crashes the pipeline.

---

## 9. Shared data contracts (build these once, obey everywhere)

Use Pydantic v2 models (JSON-serializable, validated). These are the integration backbone.

**`WordToken`** — one recognized word.
```
text: str
confidence: float            # 0..1
bbox: tuple[int,int,int,int] # x, y, w, h  (top-left origin)
page_index: int
source: str                  # "paddle" | "tesseract" | "trocr" | "ensemble"
```

**`OCRResult`** — one page from one engine (or the merged ensemble).
```
text: str
words: list[WordToken]
confidence: float            # page-level aggregate, 0..1
page_index: int
source: str
raw: dict = {}               # engine-specific raw payload (kept for debug)
```

**`LexiconEntry`** — one normalized dictionary record.
```
lemma: str
root: str | None
pattern: str | None          # wazn
gloss: str
source: str
era: "classical" | "modern"
domain: str | None
examples: list[str] = []
priority: int                # from sources.py
```

**`Candidate`** — one correction proposal for a token.
```
text: str
reason: str                  # "spelling_variant" | "normalization" | "root_alt" | "morph_alt" | "identity"
lexicon_entries: list[LexiconEntry] = []
features: dict = {}          # lexicon_score, morph_score, ocr_score, context_score (filled by scorer)
score: float | None = None
```

**`TokenState`** — the per-token record that flows the whole pipeline and serializes into debug output.
```
original: str
normalized: str
normalization_log: list[dict]      # per-step changes (the "reversible" trace)
candidates: list[Candidate]
selected: str
confidence: float
sources: list[str]
decision: "accept" | "accept_with_note" | "uncertain" | "review_required"
reason_code: str
```

---

## 10. Per-file specification

Each entry: **purpose → public API → must / must-not.** Keep modules dependency-light and side-effect-free at import.

### Orchestration & config
- **`main.py`** — top-level orchestrator. Public: `process_file(file_path: str, mode: str = "clean") -> dict`; `run_pipeline(pages, mode)`. Loads config, runs the §1 order, keeps raw + intermediate artifacts for debug. **Must not** implement OCR/preprocess/lexicon logic — orchestrate only. Safe imports: if a module is missing, raise a clear error naming it. `if __name__ == "__main__"` dispatches to a minimal local run or the CLI.
- **`config.yaml`** — nested, conservative defaults. Sections: `paths` (data/models/cache), `preprocessing` (per-step on/off + params), `ocr` (engine toggles, ensemble weights; Paddle primary, Tesseract+TrOCR optional fallbacks), `normalization` (per-rule toggles, mode), `morphology`, `lexicon` (per-source priority; classical > modern), `scoring` (weights for ocr/lexicon/morphology/context), `decision` (thresholds), `output` (default mode), `cache`, `logging`.
- **`utils/config.py`** — load/validate `config.yaml` into a typed object; expose dotted access + defaults.
- **`utils/logging.py`** ➕ — central logger factory; debug mode bumps verbosity; structured (key=value) logs.
- **`utils/cache.py`** ➕ — `diskcache`-backed get/set with namespacing for OCR results and lexicon queries; keyed by content hash + config hash.

### Ingestion
- **`io/document_loader.py`** ➕ — Public: `load_document(path) -> list[PageImage]` where each page is `{image: np.ndarray, page_index, dpi, source_path}`. PDFs via PyMuPDF (render at configurable DPI, default 300); images loaded directly. `classify_page(image) -> dict` flags scanned/low-res/skewed/multi-column heuristically. Supports PDF/PNG/JPG/TIFF.

### Preprocessing (`preprocessing/`)
- **`image_pipeline.py`** — Public: `preprocess_image(image, config) -> (image, metadata)`. Runs enabled steps in order: grayscale → contrast → denoise → deskew → threshold → optional layout. Returns processed image + metadata listing steps applied. On any step failure, log and fall back to the pre-step image (never crash).
- **`denoise.py`** — `denoise(image)`, `median_filter(image)`. OpenCV. **Must not** over-smooth (preserve Arabic dots/diacritics). Deterministic.
- **`thresholding.py`** — `apply_clahe(image)`, `adaptive_binarization(image)`. Prefer adaptive over global; safe for faint text.
- **`deskew.py`** — `detect_skew(image) -> float`, `correct_skew(image, angle)`. Hough-based (or projection-profile) angle estimate; near-zero skew is a no-op; preserve quality on rotation.
- **`layout_detection.py`** — `detect_layout(image, config=None) -> regions`. Default pure-OpenCV projection-profile detection of columns/margins/footnotes/verse blocks; use `layoutparser` only if the `[layout]` extra is installed. Returns structured regions (no OCR here). Safe to disable.

### OCR engine (`ocr_engine/`)
- **`schema.py`** — defines `WordToken`, `OCRResult` (§9). JSON-serializable.
- **`base.py`** — abstract `OCRBackend` with `extract(self, image, page_index: int = 0) -> OCRResult`. No backend-specific logic.
- **`paddle_backend.py`** — PaddleOCR Arabic (`lang="ar"`), primary. Returns text + word tokens (bbox + confidence). Lazy-load model on first use; fail gracefully if PaddleOCR/paddlepaddle missing. Extraction only.
- **`tesseract_backend.py`** — Tesseract with `lang="ara"`, fallback. Word tokens + confidences from `image_to_data`. Detect missing binary/lang pack and fail with a clear message.
- **`trocr_backend.py`** — **line-level** TrOCR re-recognition for low-confidence regions, used as a verifier signal, **not** a primary token extractor. Public: `recognize_line(crop) -> (text, score)`; optionally `extract(...)` returning a coarse `OCRResult`. Model id from config (`[trocr]` extra). Handle missing model/transformers/torch gracefully.
- **`ensemble.py`** — Public: `run_ensemble(image, page_index, config) -> OCRResult`. Steps: run enabled backends → align tokens (via `alignment/`) → per-token weighted vote/selection using config weights → preserve all raw outputs in `raw`. Deterministic, explainable; produces merged tokens + confidence map + source attribution. No preprocessing/lexicon/morphology here.

### Alignment (`alignment/`)
- **`string_similarity.py`** — `similarity(a, b) -> float` (normalized edit distance / Levenshtein ratio), robust to noisy Arabic. Dependency-light.
- **`bbox_alignment.py`** — `iou_match(box1, box2) -> float`, `align_by_bbox(tokens_a, tokens_b) -> pairs`. General/reusable; no OCR dependency.
- **`token_matcher.py`** — `match_tokens_by_text(...)` + grouping that combines bbox + string similarity into token clusters across engines. Deterministic.

### Normalization (`normalization/`)
- **`arabic_normalizer.py`** — `normalize_text(text, config=None) -> (text, change_log)`. Configurable rules: alef variants (أ/إ/آ → ا), alef-maqsura↔yaa (ى/ي), taa-marbuta↔haa (ة/ه, configurable), hamza handling, tatweel (ـ) removal, punctuation cleanup, diacritics by mode. Records a per-change log for traceability. `reverse_normalization_map()` returns the documented folding rules (reference only — **not** a lossless inverse; see §2).
- **`noise_filter.py`** — `clean_noise(text) -> (text, change_log)`. Remove stray symbols/garbage glyphs, fix broken spacing/ligatures. Conservative; never destroys valid Arabic.

### Morphology (`morphology/`)
- **`root_extractor.py`** — `extract_root(word) -> list[RootCandidate]`. Trilateral-first with graceful fallback; keeps multiple candidates. Explainable. Assists, never overrules lexicon.
- **`pattern_analyzer.py`** — `detect_pattern(word, root) -> {pattern, confidence}`. Wazn detection; works for classical + modern. No OCR/API deps.
- **`camel_adapter.py`** — `analyze(word) -> {lemma, root, pos, pattern?}`. Wraps CAMeL Tools; lazy DB load; if CAMeL/DB missing, return `None`/empty and let rule-based fallback take over. Isolated.

### Lexicon ingestion (`lexicon_ingestion/`)
- **`sources.py`** — declarative registry of sources with the fields in §7. Purely declarative.
- **`parser.py`** — per-source adapters → canonical `LexiconEntry`. Modular (one adapter per source). No DB logic here.
- **`storage.py`** — durable local store: SQLite (preferred) or JSONL. `save_entries`, `load_entries`, `clear`/`rebuild`. Independent from the query layer; migration-friendly.
- **`index_builder.py`** — build indexes by lemma, root, pattern, and approximate match. Lightweight now; FAISS-ready later. Output suited for fast runtime lookup. No OCR/scoring.

### Lexicon engine (`lexicon_engine/`)
- **`query_engine.py`** — `query(word, context=None) -> list[LexiconEntry]` across enabled sources, via the index/storage layer (not raw scanning). Every result carries source + era. No final scoring here.
- **`candidate_generator.py`** — `generate(token, morphology, config) -> list[Candidate]`. Sources of candidates: identity, spelling variants, normalization variants, root-based alternatives, morphology-informed alternatives. Conservative; each candidate gets a `reason`.
- **`context_scorer.py`** ➕ — `context_score(candidate_text, left_context, right_context) -> float`. Default: deterministic n-gram/co-occurrence over the lexicon corpus. Optional masked-LM (AraBERT) behind the `[lm]` extra/config flag. This is the module that finally **produces** the `context_score` the confidence engine consumes.
- **`scorer.py`** — `score(candidate, context, ocr_conf, config) -> Candidate` filling `features` (lexicon_score, morph_score, ocr_score, context_score) and a transparent weighted `score`. Deterministic; weights from config. Returns score + explanation.
- **`ranker.py`** — `rank(candidates) -> RankedResult` (best + alternatives, full ordering preserved). Ranking only — no lookup/feature extraction.

### Confidence engine (`confidence_engine/`)
- **`scoring.py`** — `final_confidence(features, config) -> float` in [0,1], configurable weights (defaults in §11). Transparent formula; does **not** decide the action.
- **`decision.py`** — `decide(confidence, config) -> (label, reason_code)` with labels `accept | accept_with_note | uncertain | review_required` from config thresholds. Deterministic.
- **`state.py`** — `TokenState` (§9) construction/serialization; lightweight; flows through the pipeline; JSON for debug.

### Output (`output/`)
- **`formatter.py`** — `format(states, text, mode) -> dict` for `clean | annotated | debug`. Separates formatting from business logic.
- **`json_export.py`** — stable, deterministic JSON (text, confidence map, corrections, sources, debug metadata when present). Suitable for downstream/Claude Code consumption.
- **`markdown_export.py`** — compact annotated markdown: original | corrected | confidence | sources. For manual review.
- **`debug_export.py`** — full structured trace: raw per-engine OCR, preprocessing metadata, alignment, normalization deltas, morphology, lexicon candidates, scores, decisions. Verbose; shows where each correction came from.

### API (`api/`)
- **`server.py`** — FastAPI app entry; load config + init pipeline once at startup; simple startup/shutdown. No business logic.
- **`routes.py`** — `POST /process` accepting a file upload + `mode` (clean/annotated/debug); thin handler delegating to the pipeline; clean error responses. Plus `GET /health`.
- **`schemas.py`** — Pydantic request/response models for upload metadata, mode, and responses (clean text, confidence map, corrections, lexicon sources, optional debug metadata).

### CLI (`cli/`)
- **`main.py`** — CLI entry (Typer or argparse). Commands: `process`, `debug`, `batch`. Clean arg parsing; human-readable minimal output. Calls shared orchestration.
- **`commands.py`** — handlers for single-file, annotated, debug, and batch processing. Separate from arg parsing; call pipeline + formatter; structured errors. Example: `ocr-arabic process input.pdf --mode annotated`.

### Utilities & quality
- **`utils/*`** — see config/logging/cache above.
- **`eval/metrics.py`** ➕ — `cer`, `wer`, `root_accuracy`, `unresolved_rate`, `sentence_coherence`.
- **`eval/benchmark.py`** ➕ — run the pipeline over a fixed benchmark set (clean print, low-res scans, skewed, Ottoman-style, Quranic typography, mixed Arabic/Latin, noisy historical) and report metrics + a summary table.
- **`tests/*`** — deterministic unit tests with synthetic fixtures and mocked engines (no heavy model downloads): preprocessing types/shapes; OCR schema + ensemble merge preserves text/confidence/source; alignment correctness; normalization conservatism + change-log; lexicon query/candidate/rank + provenance; morphology with CAMeL-absent fallback; end-to-end pipeline in all three modes respecting the §1 order.
- **`__init__.py`** — minimal per package; export the most useful public symbols (e.g. `OCRBackend`, `OCRResult`, `preprocess_image`, `normalize_text`, `query`, `final_confidence`).

---

## 11. Confidence model

Default weighted formula (override via `config.scoring`):

```
final = 0.30*ocr_confidence + 0.30*lexicon_score + 0.20*morphology_score + 0.20*context_score
```

Default decision thresholds (`config.decision`):

| Range | Label |
|---|---|
| `>= 0.90` | accept |
| `0.70 – 0.89` | accept_with_note |
| `0.50 – 0.69` | uncertain |
| `< 0.50` | review_required |

Keep transparent first; learned calibration of weights/thresholds is a later upgrade.

---

## 12. Testing & evaluation

- **Unit:** every module, deterministic, fixture-based, engine-mocked. Target: each module passes in isolation with all optional deps absent.
- **Integration:** `tests/test_pipeline.py` runs end-to-end on the synthetic fixture lexicon + a tiny bundled image, asserting structured output in all three modes and correct stage ordering.
- **Benchmark:** `eval/benchmark.py` over the seven page categories above; report CER/WER, corrected-word accuracy, root accuracy, unresolved-token rate. Use it to tune scoring weights — do not hand-tune blindly.

---

## 13. Known risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| PaddleOCR Arabic weak on classical/Ottoman/Quranic type | Low primary accuracy | Treat Paddle as one signal; ensemble + lexicon correction; allow per-job engine weighting |
| No official Arabic TrOCR model | Verifier may be unavailable | Make TrOCR fully optional; pin a community checkpoint in config; degrade gracefully |
| Lexicon licensing / Almaany ToS | Legal/compliance | Use OpenITI/PD sources; **exclude Almaany scraping**; record license per source in `sources.py` |
| `context_score` had no producer | Silent zeros, mis-scoring | Added `context_scorer.py`; default deterministic n-gram so the signal always exists |
| `layoutparser`/detectron2 install pain | Broken environments | Optional `[layout]` extra; pure-OpenCV default |
| CAMeL Tools data download + weight | First-run failures | Lazy load; rule-based fallback; documented `camel_data` step |
| Neural non-determinism | "Deterministic debug" overpromised | Scope determinism to non-neural stages; set seeds; document clearly |
| Lossy normalization "reverse map" | False expectation | Per-token original + change log; no global inverse promised |
| No file loader in original plan | PDFs can't be processed | Added `io/document_loader.py` |

---

## 14. Ready-to-paste Claude Code kickoff prompt

> You are implementing a **modular Arabic OCR + Lexicon-Augmented Intelligence system** from a master plan I will follow file-by-file.
>
> **Goal:** ingest scanned Arabic PDFs/images, run an OCR ensemble (PaddleOCR primary; Tesseract + optional TrOCR fallbacks), normalize noisy output conservatively and reversibly-by-trace, resolve uncertain words using morphology (CAMeL Tools + rule fallback) and multiple Arabic lexicons (classical > modern), and return corrected Arabic text with full per-token provenance. Provide `clean`, `annotated`, and `debug` output modes, plus FastAPI and CLI interfaces.
>
> **Hard rules (non-negotiable):** never overwrite a token without keeping the original + alternatives; every correction carries source(s) + reason code; lexicon evidence > morphology > model guessing; raw per-engine OCR stays accessible; deterministic + explainable in the non-neural stages; config-driven (no magic numbers); modules side-effect-free at import; missing optional deps disable a signal, never crash.
>
> **Shared contracts (build first, obey everywhere):** Pydantic models `WordToken`, `OCRResult`, `LexiconEntry`, `Candidate`, `TokenState` exactly as specified in the plan.
>
> **Build strictly in this order:** skeleton+config → `main.py` (safe imports) → schema+state → `io/document_loader` → preprocessing → OCR backends → alignment → ensemble → normalization → lexicon ingestion → morphology → lexicon engine (incl. context_scorer) → confidence engine → output → API+CLI → tests+eval → README.
>
> **Important corrections to honor:** include a PDF/image loader; produce `context_score` via a real module; do not promise a lossless normalization inverse (use per-token original + change log); TrOCR is optional line-level verification, not bbox extraction; do **not** scrape Almaany — use OpenITI/public-domain lexicon sources and ship a synthetic fixture lexicon so everything runs with zero external data; keep `layoutparser` optional with a pure-OpenCV default.
>
> Start with Phase 1 (skeleton, `config.yaml`, `utils/config.py`, `utils/logging.py`) and the shared Pydantic contracts. Show me each file, wait for confirmation, then proceed to the next phase. Ask before adding any dependency outside the agreed `requirements.txt`.

---

*End of master plan.*
