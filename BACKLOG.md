# Ancient Arabic OCR Platform — Full Backlog
## What's built, what's missing, what to build next

---

## BUILT AND WORKING

| Component | Status | Notes |
|---|---|---|
| Kraken OCR pipeline | ✓ | Line segmentation + recognition |
| Profile system | ✓ | 4 profiles, YAML-driven |
| Preprocessing (CLAHE, denoise, deskew, gamma, contrast) | ✓ | Profile-driven |
| Sauvola + nlbin + Otsu binarization | ✓ | Per-profile |
| Weighted confusion costs | ✓ | Config-driven Arabic pairs |
| CAMeL Tools morphology | ✓ | Installed, wired |
| Lexicon (Lane's, Khorsi, Quranic, 187k entries) | ✓ | Working |
| Confidence scoring + HITL | ✓ | 382 tests passing |
| FastAPI backend | ✓ | All routes working |
| React frontend | ✓ | Expert tool |
| Line review interface | ✓ | Sprint 1 complete |
| Drawing/annotation tool | ✓ | On manuscript directly |
| Training pair export | ✓ | PNG + .gt.txt Kraken format |
| Synthetic data generator | ✓ | 10k samples, separate repo |
| Passim/rapidfuzz alignment | ✓ | Post-scoring |
| Agricultural lexicon (ibn awwam) | ✓ | Deferred (no source text yet) |
| N-best candidate display | ✓ | UI built, single model only |
| Character confidence display | ✓ | UI built |

---

## PLANNED BUT NOT BUILT

### HIGH PRIORITY — directly improves OCR accuracy

**1. Transkribus integration**
- What: Send line images to Transkribus API, get recognition results back
- Why: Transkribus has pre-trained models for difficult Arabic scripts (Maghrebi, Andalusian) trained on thousands of pages — better than Muharaf for some manuscript styles
- How: REST API call per line, merge result with Kraken output as third engine
- Effort: Low
- Claude Code instruction needed: Yes

**2. Vision model post-correction (Claude/GPT-4V)**
- What: Lines below 0.85 confidence → send image + OCR text to vision API → corrected text as better starting point for human review
- Why: Vision models know classical Arabic vocabulary and can fix garbled words from context even without knowing the specific scribe's hand
- How: Optional pipeline step after Kraken, before review queue. Uses Anthropic API (already available in artifacts)
- Effort: Low
- Claude Code instruction needed: Yes

**3. Bleed-through removal**
- What: Text from reverse side of page shows through parchment, confuses OCR
- Why: Major source of errors on thin parchment manuscripts
- How: cv2 background subtraction + frequency domain filtering
- Effort: Low
- Claude Code instruction needed: Yes

**4. Multispectral channel extraction**
- What: Extract R, G, B channels separately, process each, combine best
- Why: Faded ink often shows more clearly in one channel than others
- How: Split BGR channels, run OCR on each, pick highest confidence
- Effort: Low
- Claude Code instruction needed: Yes

**5. N-best from multiple models with voting**
- What: Run Kraken with both Muharaf AND Agapet models, vote on best result per line
- Why: Different models make different errors — ensemble reduces overall error rate
- How: Secondary model slot already exists in KrakenBackend, needs activation + voting logic
- Effort: Low (slot exists, just needs wiring)
- Claude Code instruction needed: Yes

---

### MEDIUM PRIORITY — improves workflow and scalability

**6. Crowd platform (Sprint 4)**
- What: Public web interface for anyone to contribute corrections
- Why: Main bottleneck is Arabic readers — crowd solves this at scale
- How: Simple React app, task queue, majority vote verification, no login required initially
- Effort: Medium (2-3 sessions)
- Claude Code instruction: PLATFORM_PLAN.md Sprint 4

**7. Language model post-correction**
- What: After OCR, flag words not in classical Arabic, suggest corrections
- Why: Catches substitution errors that produce nonsense words
- How: CAMeL Tools already installed — run morphological analysis, flag unrecognized forms
- Effort: Low
- Claude Code instruction needed: Yes

**8. Active learning queue**
- What: System identifies which lines will benefit most from human review
- Why: Not all lines need equal attention — focus human effort where it matters
- How: Confidence variance, character-level uncertainty, line complexity score
- Effort: Low
- Claude Code instruction needed: Yes

**9. Transkribus model fine-tuning**
- What: Fine-tune a Transkribus model on your corrections (not just Kraken)
- Why: Transkribus has better tooling for Arabic manuscript fine-tuning
- Effort: Medium (requires Transkribus account + their training pipeline)

---

### LOWER PRIORITY — future platform features

**10. CycleGAN style transfer for synthetic data**
- What: Train a model to convert clean Arabic text images into manuscript-style images
- Why: More realistic synthetic training data than the current augmentation pipeline
- Effort: High (weeks of GPU training)
- When: After crowd platform is running and generating real corrections

**11. Corpus search across manuscripts**
- What: Full-text search across all transcribed manuscripts
- Effort: Medium

**12. TEI XML export**
- What: Standard academic format for manuscript transcriptions
- Effort: Low

**13. IIIF integration**
- What: Standard manuscript image protocol used by all major libraries
- Effort: Medium

**14. Per-manuscript model versioning**
- What: Track which model version produced each transcription
- Effort: Low

---

## RECOMMENDED BUILD ORDER (next sessions)

### Session 1 — Quick wins that immediately improve accuracy
Build items 3, 4, 5 together (bleed-through removal, multispectral, N-best voting).
These are all preprocessing/pipeline changes, no UI needed.
Expected accuracy improvement: 5-10% on difficult pages.

### Session 2 — Vision model post-correction (item 2)
Add Claude/GPT-4V as optional post-correction step.
Expected improvement: significant on low-confidence lines.

### Session 3 — Transkribus integration (item 1)
Add as third OCR engine alongside Kraken.
Expected improvement: best results on manuscript styles Muharaf wasn't trained on.

### Session 4 — Crowd platform (item 6)
Sprint 4 from PLATFORM_PLAN.md.
This solves the human bottleneck permanently.

### Session 5 — Language model post-correction (item 7)
CAMeL Tools integration for classical Arabic vocabulary checking.

---

## HOW TO USE THIS DOCUMENT

When budget resets, paste into Claude Code:

For Session 1:
```
Read BACKLOG.md items 3, 4, and 5. Implement all three as preprocessing 
pipeline steps. Each is optional and config-driven (enabled/disabled in 
config.yaml). Token-efficient mode. Start with item 5 (N-best voting — 
slot already exists). Wait for confirmation at each item.
```

For Session 2:
```
Read BACKLOG.md item 2. Add vision model post-correction as an optional 
pipeline step. Use the Anthropic API (already available). Lines below 
config threshold get sent to claude-sonnet-4-6 with the line image + 
OCR text. Corrected result stored as candidate in the review queue. 
Token-efficient mode.
```
