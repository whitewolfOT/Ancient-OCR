# Session State — Ancient OCR

## Current state
- 262 tests passing
- Branch: claude/adoring-carson-UqiLp

## Frontend (frontend/)
- React app with Vite + Tailwind CSS
- 7 components built: UploadZone, PageSidebar, PageViewer, PreprocessingControls, GroundTruthPanel, WorkflowBar, SimilarPagesPanel
- 6 backend endpoints: upload, pages list, preview, ground truth (GET+POST), apply cluster settings, page image
- pHash clustering with connected components + medoid representative
- Debounced preprocessing preview (300ms, stale response guard via reqCounter)
- RTL Arabic ground truth textarea
- data/documents/documents.db for document state
- Integration test passed — all 6 endpoints verified on real image

## What the frontend does NOT do yet
- WorkflowBar steps not navigable — display only (step advances automatically when OCR runs)
- SimilarPagesPanel only useful on multi-page documents
- No export functionality

## Lexicon DB (data/lexicons/lexicons.db)
- Total: 187,451 entries
- _fixture: 80 (synthetic, priority 1)
- quranic_corpus: 4,635 (real data — mustafa0x/quran-morphology, 130k lines)
- lanes: 39,950 (live, TEI XML parsed)
- khorsi_roots: 142,786 (live, SQL parsed from Tāj al-ʿArūs extraction)
- qamus: 0 (CLARIN URL 403 — needs manual browser download)
- arabic_wordnet: 0 (enabled=False — globalwordnet.org 403 in container)

## What works right now
- Degradation detection: classify_page() returns low_contrast, faded_ink, high_noise, bleed_through flags
- suggest_settings_from_degradation() returns recommended CLAHE/denoise/binarization from flags
- Preview endpoint returns degradation_flags + suggested_settings in response
- Multi-hypothesis OCR: PaddleOCR confusion-pair alternatives (edit-distance-1) stored in OCRResult.raw["paddle_alternatives"]
- Confusion pairs wired into candidate_generator as "ocr_alternative" candidates
- OCR endpoint: POST /pages/{page_id}/ocr runs pipeline, saves to ocr_results table, returns token decisions+bboxes
- GET /pages/{page_id}/ocr returns saved OCR result
- PageViewer: "Run OCR" button, SVG confidence heatmap overlay (4 decision colors), show/hide heatmap toggle
- WorkflowBar step advances to "review" when OCR results are present
- Full pipeline end to end on real Arabic scans with Tesseract
- Four-tier token gate: stopword → cache → OCR confidence → full resolution
- Stopword filter ~500 words
- Noise filter reduces review queue from ~200 to ~70 on real scans
- DPI normalization upscales low-res inputs before OCR
- morph_score uses real root agreement
- Left context wired into scoring loop (up to 3 previous resolved tokens)
- Quranic bigram corpus active (130k rows, verse co-occurrence)
- Three-tier context scoring: bigram hit > unigram presence > neutral 0.5
- Tier 3 non-Arabic script check (digits/punctuation no longer accepted as ocr_confident)
- review_queue embedded in annotated output
- Khorsi + Lane's contributing multi-source hits on real scans

## Real scan baseline (data/test_images/sample_classical.jpg, Tesseract only)
- 40 accept / 14 accept_with_note / 38 uncertain / 25 review_required / 219 rejected
- review_queue: 63 flagged
- Root cause of rejections: Tesseract fragments vocalized Arabic at glyph level
- Fix: PaddleOCR (models need local download ~400MB)

## What does NOT work yet
- PaddleOCR models not downloaded (403 in container) — run locally to activate
- CAMeL Tools morphology DB not installed — rule-based fallback only
- Qāmūs LMF needs manual browser download from CLARIN
- Arabic WordNet — parser stub written (enabled=False), data needs manual download from globalwordnet.org (403 in container)
- HuggingFace upload configured: `upload-lexicons` CLI command available; set HF_TOKEN env var and --repo-id to activate
- Right context not wired in pipeline

## Known issues (documented, do not fix without discussion)
- Tesseract fragments vocalized Arabic — PaddleOCR is the fix
- Khorsi has no gloss field — root/lemma coverage only

## Architecture decisions (do not revisit without reason)
- Model A: pre-built lexicons.db on HuggingFace, not raw sources at runtime
- Single DB path: data/lexicons/lexicons.db
- ingest_all_enabled() is local dev convenience only
- ensure_lexicons_db() is the runtime path
- Khorsi priority=5, Lane's priority=9, Quranic priority=10, fixture priority=1
- taa_marbuta normalization OFF by default
- context_scorer: unseen bigrams are no-evidence (not penalised) — Quranic corpus is domain-specific, absence means out-of-domain not incorrectness

## Next session priorities (in order)
1. Run locally with PaddleOCR — this is the single highest-impact remaining step, requires local machine only
2. Install CAMeL Tools locally (`camel_data -i morphology-db-msa-r13`) — improves morphology scoring
3. Download Qāmūs LMF manually from CLARIN browser and place in `data/lexicons/qamus/` — adds ~25k structured classical entries
4. Arabic WordNet — parser stub written, data needs manual download from globalwordnet.org
5. HuggingFace upload — configure `hf_repo_id` and `HF_TOKEN` once lexicons are stable

## Commands
pytest tests/ -x
python -m cli.main process data/test_images/sample_classical.jpg --mode annotated
python -m cli.main upload-lexicons --repo-id username/ancient-ocr-lexicons
