# Ancient Arabic OCR Platform — Full Roadmap
## From Single Manuscript to Complete Classical Arabic Corpus

---

## Vision

A distributed human-AI platform that digitizes the classical Arabic scholarly
corpus — Ibn Sina, Ibn Rushd, al-Kindi, al-Biruni, al-Ghazali, and hundreds
more — through a self-improving loop where AI handles the easy parts and humans
(both experts and crowd) handle the rest. Every correction makes the AI better.
Every improvement means less human work for the next manuscript.

---

## How the system works (complete picture)

```
NEW MANUSCRIPT UPLOADED
         ↓
AUTO-PROCESSING (no human needed)
  • Deskew, enhance, binarize
  • Detect pages, margins, columns
  • Segment into lines (Kraken blla — already works)
  • Run OCR with closest existing model
  • Score every line by confidence
         ↓
TRIAGE (automatic)
  • Confidence ≥ 0.95 → auto-accepted, no review needed
  • Confidence 0.80–0.95 → Crowd queue
  • Confidence < 0.80 → Expert queue
  • Completely unreadable → Flag for manual segmentation
         ↓
TWO REVIEW CHANNELS (parallel)

EXPERT CHANNEL                    CROWD CHANNEL
Scholars, paleographers           Anyone — students, enthusiasts,
Full page view                    paid micro-workers
All tools (annotation,            One line or one word at a time
drawing, bbox editing)            Simple text field
Highest trust weight              Multiple people per item
Paid or volunteer                 Majority vote = accepted
                                  Points + micro-payment

         ↓
VERIFIED TRANSCRIPTION
  • Accepted by expert OR
  • Agreed by 3+ crowd contributors OR
  • Auto-accepted (high confidence)
         ↓
TRAINING DATA LOOP
  • Verified lines → fine-tune current model
  • New model → re-run on remaining items
  • Confidence rises → less human work needed
  • Cycle continues automatically
         ↓
CORPUS OUTPUT
  • Searchable full text
  • Image-text alignment
  • TEI XML export
  • API for researchers
  • Open access publication
```

---

## Sprint Roadmap

### SPRINT 1 — Line Review Interface (NEXT, 1 session)
**Goal:** Expert correction tool + first training dataset

What gets built:
- Line-by-line correction interface (keyboard-driven, fast)
- API endpoints saving corrections to disk
- Export in Kraken training format (.png + .gt.txt pairs)
- Progress tracking per page

Outcome: After 2 hours of correction work, 162 verified lines from 9 pages.
That's enough to fine-tune the model for the first time.

---

### SPRINT 2 — First Fine-Tuning Loop (1 session + 3 hours Colab)
**Goal:** Improve model accuracy from ~70% to ~85% on this manuscript

What gets built:
- Google Colab notebook for Kraken fine-tuning (reusable for every manuscript)
- Model version management in the pipeline
- Before/after CER comparison tool
- Auto-re-run OCR after new model is installed

Steps:
1. Export training pairs from Sprint 1
2. Upload to Colab, run `ketos train` on free T4 GPU
3. Download new .mlmodel, drop into models/kraken/
4. Re-run OCR on same 9 pages
5. Compare confidence scores

Outcome: Second-pass correction takes 30 minutes instead of 2 hours.

---

### SPRINT 3 — Multi-Manuscript Pipeline (1-2 sessions)
**Goal:** Process any new manuscript without code changes

What gets built:
- Manuscript registry (upload + metadata: title, author, era, script style)
- Auto-script-detection (Naskh, Maghrebi, Nastaliq, etc.)
- Model selection based on script type
- Per-manuscript profile auto-configuration
- Manuscript dashboard showing status of each

New manuscript workflow:
1. Upload scans (PDF or image folder)
2. System detects script style, selects model
3. Auto-OCR runs
4. Shows confidence distribution
5. Routes to correction queues

---

### SPRINT 4 — Crowd Platform (2-3 sessions)
**Goal:** Anyone can contribute corrections, not just experts

What gets built:
- Separate crowd-facing web app (simple, mobile-friendly)
- Account system with trust scores
- Task queue: show one line, edit text, submit
- Verification system: same line to 3 people, majority wins
- Seeded ground truth tasks (known correct answers) to calibrate trust
- Points system and leaderboard
- Basic accuracy tracking per contributor

Crowd interface (as simple as possible):
```
┌─────────────────────────────────────┐
│  Help transcribe historical Arabic  │
│  manuscripts                        │
│                                     │
│  [Line image shown here]            │
│                                     │
│  What does this line say?           │
│  ┌─────────────────────────────┐    │
│  │ من الخضرة...                │    │
│  └─────────────────────────────┘    │
│                                     │
│  [Submit]  [Can't read this]        │
│                                     │
│  You've transcribed 47 lines        │
│  ████████░░  Accuracy: 89%         │
└─────────────────────────────────────┘
```

---

### SPRINT 5 — Payment and Incentives (1 session)
**Goal:** Sustainable contributor ecosystem

What gets built:
- Stripe integration for micro-payments
- Payment tiers: per line, per page, accuracy bonuses
- Withdrawal system (PayPal, bank transfer)
- Fraud prevention (bot detection, quality gates)
- Contributor dashboard showing earnings

Economics:
- Academic institutions pay per manuscript processed
- Revenue shared with contributors
- Expert reviewers paid higher rate than crowd
- Crowd contributors earn $0.01-0.05 per verified line

---

### SPRINT 6 — Corpus Management (1-2 sessions)
**Goal:** The output is useful and accessible

What gets built:
- Full-text search across all transcribed manuscripts
- Image-text parallel viewer (click text → see manuscript region)
- TEI XML export (standard academic format)
- REST API for external researchers
- Manuscript metadata (author, date, library, call number)
- Public-facing reading interface

---

### SPRINT 7 — Scale and Quality (ongoing)
**Goal:** Accuracy approaches human expert level

What gets built:
- Active learning: automatically identify which lines need expert review
- Confidence calibration: model's stated confidence matches real accuracy
- Cross-manuscript validation: words appearing in multiple manuscripts
- Arabic NLP integration: grammar checking flags likely OCR errors
- Diacritic restoration: add tashkeel to undiacritized text

---

## Manuscript Priority List

Based on scholarly impact, scan availability, and script similarity to
the Muharaf training data:

| Priority | Author | Work | Script | Scans Available |
|---|---|---|---|---|
| 1 | Ibn al-Awwam | Kitāb al-Filāḥa | Andalusian Naskh | Current test ✓ |
| 2 | Ibn Sina | Canon of Medicine | Eastern Naskh | Princeton, BnF |
| 3 | Ibn al-Haytham | Book of Optics | Abbasid Naskh | Süleymaniye |
| 4 | Ibn Khaldun | Muqaddimah | Maghrebi | BnF, multiple |
| 5 | al-Biruni | Al-Qanun | Eastern Naskh | Multiple |
| 6 | Ibn Rushd | Commentaries | Andalusian | Escorial |
| 7 | al-Ghazali | Ihya | Various | Widespread |
| 8 | al-Kindi | Philosophical works | Abbasid | Istanbul |

Each new manuscript style adds a new model. By manuscript 5, the system
handles most Naskh variants with minimal fine-tuning needed.

---

## Technology Stack (final state)

```
BACKEND
  FastAPI (current) — OCR pipeline, corrections API
  Kraken — segmentation + recognition (current)
  PostgreSQL — replaces file-based storage at scale
  Redis — task queue for crowd review
  Celery — background OCR jobs
  S3/R2 — manuscript image storage

FRONTEND  
  React (current) — expert interface
  React Native or PWA — crowd mobile interface
  
AI/ML
  Kraken — primary HTR engine (current)
  CAMeL Tools — Arabic morphology (current)
  Custom fine-tuned models per manuscript style

INFRASTRUCTURE
  Docker — containerization
  Cloudflare — CDN for manuscript images
  GitHub Actions — CI/CD

INTEGRATIONS
  Stripe — payments
  PayPal — payouts
  IIIF — standard manuscript image protocol
  Zenodo — DOI for published transcriptions
```

---

## What already exists (nothing is wasted)

| Component | Status | Used in |
|---|---|---|
| Kraken OCR pipeline | ✓ Working | All sprints |
| Profile system | ✓ Working | Multi-manuscript (Sprint 3) |
| Confidence scoring | ✓ Working | Triage routing |
| HITL review interface | ✓ Working | Expert channel (extended) |
| Annotation/drawing tool | ✓ Working | Expert channel |
| Training pairs system | ✓ Working | Fine-tuning loop (Sprint 2) |
| FastAPI backend | ✓ Working | Extended each sprint |
| React frontend | ✓ Working | New views added |
| Weighted confusion costs | ✓ Working | Candidate generation |
| Passim alignment | ✓ Working | Cross-manuscript validation |
| Agricultural lexicon | ✓ Working | Domain vocabulary |
| Synthetic data generator | ✓ Working | Training augmentation |

---

## Grant and Funding Opportunities

These organizations actively fund exactly this type of project:

- **NEH (National Endowment for the Humanities)** — Digital humanities grants
- **Mellon Foundation** — Scholarly communications + manuscript digitization  
- **ISIF (Internet Society Foundation)** — Open access digital infrastructure
- **DARPA LORELEI** — Low-resource language technology
- **Qatar Foundation** — Arabic cultural heritage specifically
- **King Abdulaziz Foundation** — Arabic manuscript preservation
- **British Library Endangered Archives Programme** — Digitization grants

A working demo with even 50 verified lines is enough for a grant application.
Sprint 1 + Sprint 2 = fundable proof of concept.

---

## The self-improving loop (how it compounds)

```
Manuscript 1 (current): 70% accuracy, 2 hours correction per page
        ↓ fine-tune
Manuscript 1 second pass: 85% accuracy, 30 min correction per page
        ↓ fine-tune  
Manuscript 1 third pass: 93% accuracy, 5 min correction per page
        ↓ model published as "Andalusian Naskh v1"

Manuscript 2 (Ibn Sina): starts at 75% (transfer learning benefit)
        ↓ fine-tune on 20 pages
Manuscript 2: 88% accuracy after first fine-tune
        ↓ model published as "Eastern Naskh v1"

By manuscript 10: new manuscripts reach 85% accuracy with NO fine-tuning
because the base model has learned enough Arabic manuscript variation.

Crowd contribution requirement drops from 3 hours/page to 20 minutes/page.
Expert review needed only for truly ambiguous passages.
```

---

## Immediate next actions

1. Wait for budget reset
2. Run Sprint 1 (line review interface) — 1 session
3. Use the interface to correct all 162 lines across 9 pages — 2 hours work
4. Run Sprint 2 (Colab fine-tuning) — 1 session + 3 hours GPU time
5. Measure accuracy improvement
6. Apply for one grant with working demo
