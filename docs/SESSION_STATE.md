# Session State — Ancient OCR

## What works right now
- Full pipeline runs end to end on real Arabic scans with Tesseract
- 174 tests passing
- Lexicons: Lane's (39,950) + Khorsi (142,786) + Quranic corpus (pending real data) + fixture (80)
- Noise filter reduces review queue from ~200 to ~70 tokens on real scans
- DPI normalization upscales low-res inputs before OCR
- Stopword filter (~500 words) bypasses common particles instantly
- Four-tier token gate: stopword → cache → OCR confidence → full resolution
- PaddleOCR 3.x backend correct but cannot download models in this container
- morph_score uses real root agreement, not fixed values
- context_scorer in unigram-only mode (0.6 known lemma / 0.5 unknown)
- review_queue embedded in annotated output

## What does NOT work yet
- PaddleOCR needs model download (~400MB) — run locally to activate
- CAMeL Tools morphology DB not installed — rule-based fallback only
- Quranic corpus real data 403 in container — needs manual download from corpus.quran.com
- Qāmūs LMF CLARIN URL 403 — needs manual browser download
- Lisān/Tāj deprioritized — noisy prose, not plug-and-play
- Context scorer bigram model empty — no Arabic phrase corpus yet
- Right context in pipeline not wired (left context TODO)
- HuggingFace upload not configured (no HF_TOKEN, no repo_id in config)

## Known issues
- Tier 3 (OCR confidence >= 0.90) accepts noise digits unconditionally — documented
- Tesseract fragments vocalized Arabic at glyph level — PaddleOCR fixes this
- Khorsi source has no gloss field — root/lemma coverage only
- Quranic corpus has no English gloss — domain evidence only

## Architecture decisions made (do not revisit without reason)
- Model A: pre-built lexicons.db on HuggingFace, not raw sources at runtime
- Single DB path: data/lexicons/lexicons.db used by both storage.py and downloader.py
- ingest_all_enabled() is local dev convenience only, not runtime path
- ensure_lexicons_db() is the runtime path
- Khorsi priority=5 (unverified extraction), Lane's priority=9, Quranic priority=10
- taa_marbuta normalization OFF by default
- context_scorer returns 0.5 neutral when bigrams empty (not 0.0)
- Lisān/Tāj treated as long-term research task, not near-term ingestion

## Next session priorities (in order)
1. Run tests + merged DB build from current session (Phases 5-8 incomplete)
2. Get real data into the system locally: PaddleOCR models + CAMeL data + Quranic corpus TSV + Qāmūs XML
3. Re-run real scan after PaddleOCR active — get baseline quality measurement
4. Arabic WordNet (CC BY 4.0) — clean structured source, worth adding
5. HuggingFace upload setup — configure repo_id + HF_TOKEN for runtime distribution

## Branch
claude/adoring-carson-UqiLp

## Test command
pytest tests/ -x

## Real scan test
python -m cli.main process data/test_images/sample_classical.jpg --mode annotated
