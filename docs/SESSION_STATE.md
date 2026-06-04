# Session State — Ancient OCR

## Current state
- 198 tests passing
- Branch: claude/adoring-carson-UqiLp

## Lexicon DB (data/lexicons/lexicons.db)
- Total: 187,451 entries
- _fixture: 80 (synthetic, priority 1)
- quranic_corpus: 4,635 (real data — mustafa0x/quran-morphology, 130k lines)
- lanes: 39,950 (live, TEI XML parsed)
- khorsi_roots: 142,786 (live, SQL parsed from Tāj al-ʿArūs extraction)
- qamus: 0 (CLARIN URL 403 — needs manual browser download)
- arabic_wordnet: 0 (enabled=False — globalwordnet.org 403 in container)

## What works right now
- Full pipeline end to end on real Arabic scans with Tesseract
- Four-tier token gate: stopword → cache → OCR confidence → full resolution
- Stopword filter ~500 words
- Noise filter reduces review queue from ~200 to ~70 on real scans
- DPI normalization upscales low-res inputs before OCR
- morph_score uses real root agreement
- context_scorer unigram-only mode (0.6 known lemma / 0.5 unknown)
- review_queue embedded in annotated output
- Khorsi + Lane's contributing multi-source hits on real scans

## Real scan baseline (data/test_images/sample_classical.jpg, Tesseract only)
- 40 accept / 7 accept_with_note / 44 uncertain / 26 review_required / 219 rejected
- Root cause of rejections: Tesseract fragments vocalized Arabic at glyph level
- Fix: PaddleOCR (models need local download ~400MB)

## What does NOT work yet
- PaddleOCR models not downloaded (403 in container) — run locally to activate
- CAMeL Tools morphology DB not installed — rule-based fallback only
- Quranic corpus fetcher updated to use mustafa0x/quran-morphology (Arabic Unicode, 130k lines, 4635 entries) — parser updated for new format; real file already in data/lexicons/quranic_corpus/quran-morphology.txt
- Qāmūs LMF needs manual browser download from CLARIN
- Arabic WordNet — parser stub written (enabled=False), data needs manual download from globalwordnet.org (403 in container)
- HuggingFace upload configured: `upload-lexicons` CLI command available; set HF_TOKEN env var and --repo-id to activate
- Context scorer bigram model empty — no Arabic phrase corpus
- Right context not wired in pipeline

## Known issues (documented, do not fix without discussion)
- Tier 3 accepts noise digits unconditionally (conf >= 0.90 bypasses lexicon)
- Tesseract fragments vocalized Arabic — PaddleOCR is the fix
- Khorsi has no gloss field — root/lemma coverage only
- Quranic corpus synthetic only until real TSV placed in data/lexicons/quranic_corpus/

## Architecture decisions (do not revisit without reason)
- Model A: pre-built lexicons.db on HuggingFace, not raw sources at runtime
- Single DB path: data/lexicons/lexicons.db
- ingest_all_enabled() is local dev convenience only
- ensure_lexicons_db() is the runtime path
- Khorsi priority=5, Lane's priority=9, Quranic priority=10, fixture priority=1
- taa_marbuta normalization OFF by default
- context_scorer returns 0.5 neutral when bigrams empty

## Next session priorities (in order)
1. Get real data locally: PaddleOCR models + CAMeL data + Quranic corpus TSV + Qāmūs XML
2. Re-run real scan after PaddleOCR active — get true baseline quality measurement
3. Arabic WordNet (CC BY 4.0) — clean structured source, add next
4. HuggingFace upload setup — configure repo_id + HF_TOKEN
5. Wire right context in pipeline once bigram corpus exists

## Commands
pytest tests/ -x
python -m cli.main process data/test_images/sample_classical.jpg --mode annotated
python -m cli.main upload-lexicons --repo-id username/ancient-ocr-lexicons
