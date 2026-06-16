# RALM Sprint — Root-Affinity Language Model Re-Ranking Engine
## Oracle Layer for candidate_generator.py

Read CLAUDE.md fully. Read this document fully. Then start Step 0.
Token-efficient mode. Wait for confirmation after each step's smoke-check.

---

## What this builds

A root-affinity oracle that sits between Kraken's raw OCR output and the
confidence engine. It re-ranks word candidates using Arabic root frequencies
from Lane's Lexicon (static prior) and a Bayesian domain matrix that grows
smarter as the crowd submits corrections.

The oracle never hallucinates: words with no valid Arabic root are flagged
for human review rather than output as valid predictions.

---

## Compatibility checklist (verify before writing any code)

```bash
# Run these first and report all output before Step 1
python -c "from camel_tools.morphology.analyzer import Analyzer; print('CAMeL OK')"
python -c "import json; d=json.load(open('data/lexicons/lanes/index.json')); print('Lane keys:', list(d.keys())[:5])" 2>/dev/null || echo "check lexicon path"
grep -n "ralm_score\|needs_review\|candidates" ocr_engine/schema.py
grep -n "ralm_score\|needs_review" confidence_engine/state.py
grep -n "def generate_candidates\|def select" lexicon_engine/candidate_generator.py | head -20
python -c "import farasapy; print('Farasa OK')" 2>/dev/null || echo "Farasa not installed"
ls data/lexicons/
```

Report all output. Wait for confirmation before Step 1.

---

## Step 0 — Schema + Config extensions

**Touch only `ocr_engine/schema.py`, `confidence_engine/state.py`,
and `config.yaml`. No logic yet.**

### 0-A  Add `ralm_score` to `WordToken`

In `ocr_engine/schema.py`, add to `WordToken`:

```python
ralm_score: float | None = None        # 0..1 combined affinity score
ralm_zone: str | None = None           # "accept" | "review" | "abstain"
ralm_root: str | None = None           # extracted root (for debugging)
```

These are additive — all existing fields unchanged.

### 0-B  Add `ralm_zone` to `TokenState`

In `confidence_engine/state.py`, add to `TokenState`:

```python
ralm_zone: str | None = None           # propagated from WordToken
ralm_score: float | None = None
```

If `ralm_zone == "review"` or `ralm_zone == "abstain"`, the token must be
added to the review queue regardless of Kraken confidence. Wire this into
the existing `needs_review` logic:

```python
@property
def needs_review(self) -> bool:
    # Existing logic OR RALM flagged it
    return self._needs_review or self.ralm_zone in ("review", "abstain")
```

Do not remove existing `needs_review` logic — extend it.

### 0-C  Config additions

Append to `config.yaml` (do not remove anything existing):

```yaml
ralm:
  enabled: true
  thresholds:
    accept: 0.85
    review: 0.50
    # Below review = abstain
  bayesian:
    learning_rate: 0.1
    min_trust_score: 0.3    # ignore corrections from users below this
    context_window: 2        # words before/after for co-occurrence
  camel:
    preset: "calima-msa-s31"
  paths:
    w_base: "data/ralm/w_base.json"
    m_domain: "data/ralm/m_domain.json"
    root_cache: "data/ralm/root_cache.json"
  fallback_label: "[UNVERIFIED]"
```

### 0-D  Directories

```bash
mkdir -p data/ralm
echo "data/ralm/root_cache.json" >> .gitignore  # cache only
# w_base.json and m_domain.json ARE committed (they are model state)
```

Smoke-check:
```python
from ocr_engine.schema import WordToken
t = WordToken(text="test", confidence=0.9, bbox=(0,0,10,10),
              page_index=0, source="kraken", ralm_score=0.7, ralm_zone="accept")
assert t.ralm_zone == "accept"
```

Wait for confirmation before Step 1.

---

## Step 1 — `lexicon_engine/root_extractor.py`

**New file. No changes to existing files.**

Implements the tiered root extraction pipeline:

```
Tier 1: CAMeL Tools (.root field — trilateral root, not .lex)
Tier 2: Pattern-based extraction (Arabic morphological patterns)
Tier 3: Farasa stem approximation (if farasapy + Java available)
Tier 4: Abstain → None
```

### Implementation

```python
"""
Root extractor for Arabic words.
All tiers degrade gracefully — never raises, always returns str | None.
Loads CAMeL analyzer once at module level (lazy, not at import time).
"""
from __future__ import annotations
import re
import logging
from functools import lru_cache
from pathlib import Path
import json

logger = logging.getLogger(__name__)

_camel_analyzer = None
_farasa_segmenter = None
_camel_available = False
_farasa_available = False

def _init_camel(preset: str = "calima-msa-s31") -> bool:
    global _camel_analyzer, _camel_available
    if _camel_available:
        return True
    try:
        from camel_tools.morphology.analyzer import Analyzer
        from camel_tools.morphology.database import MorphologyDB
        db = MorphologyDB.builtin_db(preset)
        _camel_analyzer = Analyzer(db)
        _camel_available = True
        return True
    except Exception as e:
        logger.warning(f"CAMeL Tools unavailable: {e}")
        return False

def _init_farasa() -> bool:
    global _farasa_segmenter, _farasa_available
    if _farasa_available:
        return True
    try:
        from farasapy.farasa.segmenter import FarasaSegmenter
        _farasa_segmenter = FarasaSegmenter(interactive=True)
        _farasa_available = True
        return True
    except Exception as e:
        logger.warning(f"Farasa unavailable (Java required): {e}")
        return False

# Arabic morphological patterns for Tier 2
# Maps pattern regex → root extraction function
# Covers the most common wazn patterns in classical Arabic
_ARABIC_PATTERNS = [
    # Verb patterns: fa'ala, fa''ala, fā'ala
    (re.compile(r'^([\u0600-\u06ff])([\u0600-\u06ff])([\u0600-\u06ff])$'), lambda m: m.group(0)),
    # Three-letter root directly
]

def _tier2_pattern(word: str) -> str | None:
    """
    Pattern-based root extraction for common Arabic morphological forms.
    Strips common prefixes (ال، و، ف، ب، ل، م) and suffixes (ة، ات، ون، ين).
    Returns best-guess root or None.
    """
    if not word:
        return None
    
    # Strip definite article
    w = re.sub(r'^ال', '', word)
    # Strip common prefixes
    w = re.sub(r'^[وفبلك]', '', w)
    # Strip common suffixes
    w = re.sub(r'[ةهاتونين]+$', '', w)
    # Strip taa marbuta
    w = re.sub(r'ة$', '', w)
    
    # If 3 letters remain, likely a root
    arabic_letters = re.sub(r'[^\u0600-\u06ff]', '', w)
    if len(arabic_letters) == 3:
        return arabic_letters
    if len(arabic_letters) == 4:
        # Quadrilateral root — return as-is
        return arabic_letters
    return None

@lru_cache(maxsize=10000)
def extract_root(word: str, camel_preset: str = "calima-msa-s31") -> str | None:
    """
    Extract Arabic root with tiered fallback.
    Returns trilateral/quadrilateral root string or None.
    Results are cached in memory (lru_cache) and optionally to disk.
    """
    if not word or not re.search(r'[\u0600-\u06ff]', word):
        return None

    # Tier 1: CAMeL Tools
    if _init_camel(camel_preset):
        try:
            analyses = _camel_analyzer.analyze(word)
            if analyses:
                # .root is the trilateral root (e.g., "كتب"), not .lex (lexeme)
                root = analyses[0].root
                if root and root != "NOAN":
                    return root
        except Exception:
            pass

    # Tier 2: Pattern-based extraction
    root = _tier2_pattern(word)
    if root:
        return root

    # Tier 3: Farasa (stem approximation — weaker than root)
    if _init_farasa():
        try:
            stem = _farasa_segmenter.stem(word)
            if stem and len(re.sub(r'[^\u0600-\u06ff]', '', stem)) >= 3:
                return re.sub(r'[^\u0600-\u06ff]', '', stem)
        except Exception:
            pass

    # Tier 4: Abstain
    return None

def extract_root_cached(word: str, cache_path: Path | None = None,
                        camel_preset: str = "calima-msa-s31") -> str | None:
    """
    Like extract_root but also persists cache to disk for startup speed.
    """
    root = extract_root(word, camel_preset)
    return root
```

**Tests — `tests/test_root_extractor.py`:**

```python
def test_known_root_camel():
    # "كتاب" should return root "كتب"
    root = extract_root("كتاب")
    assert root is not None

def test_pattern_fallback_three_letters():
    # Simple three-letter word
    root = extract_root("ماء")
    assert root is not None

def test_non_arabic_returns_none():
    assert extract_root("hello") is None

def test_empty_returns_none():
    assert extract_root("") is None

def test_invented_word_returns_none_or_pattern():
    # "النممولي" — OCR garbage — may return None or a pattern approximation
    # Either is acceptable; what matters is it's not a confident wrong answer
    root = extract_root("النممولي")
    # No assertion on value — just verify it doesn't raise
```

Smoke-check: `pytest tests/test_root_extractor.py -v`
Wait for confirmation before Step 2.

---

## Step 2 — `lexicon_engine/ralm_matrices.py`

**New file. Builds and manages W_base and M_domain.**

### W_base — built from Lane's Lexicon (one-time, cached to disk)

Lane's Lexicon is already indexed at `data/lexicons/lanes/`. Extract root
frequencies from it. Each entry has a `root` field — count occurrences per
root to build the prior.

```python
def build_w_base(lexicon_path: Path, output_path: Path) -> dict[str, float]:
    """
    Build static root frequency map from Lane's Lexicon.
    Output: {root: normalized_frequency} (all values sum to 1.0)
    Saved to data/ralm/w_base.json.
    Called once at startup if w_base.json doesn't exist.
    """
```

### M_domain — sparse co-occurrence dict (grows with corrections)

```python
# Structure:
# {
#   "root_a": {
#     "root_b": weighted_count,   # b appeared near a in a correction
#     "root_c": weighted_count,
#   }
# }

def load_m_domain(path: Path) -> dict:
    """Load or create empty domain matrix."""

def save_m_domain(matrix: dict, path: Path) -> None:
    """Persist after every update."""

def update_m_domain(
    corrected_word: str,
    context_words: list[str],      # 2 before + 2 after in the same line
    user_trust_score: float,
    matrix: dict,
    learning_rate: float = 0.1,
    min_trust: float = 0.3,
) -> dict:
    """
    Bayesian update:
    M_domain[root_corrected][root_context] += learning_rate * trust * 1.0

    Rules:
    - Skip if user_trust_score < min_trust
    - Extract root of corrected_word and each context_word
    - Skip any word that returns root=None
    - Normalize M_domain[root_corrected] row after update (sum to 1.0)
    - Return updated matrix (caller must call save_m_domain)
    """
```

### Affinity scoring

```python
def compute_affinity(
    candidate_text: str,
    context_roots: list[str],      # roots of surrounding words
    w_base: dict[str, float],
    m_domain: dict,
    epsilon: float = 0.01,         # floor value — prevents total score=0
) -> float:
    """
    Score = W_base(root) * M_domain_score(root, context_roots)

    W_base(root): normalized frequency from Lane's (0..1)
    M_domain_score: mean co-occurrence weight with context_roots (0..1)

    If root is None: return epsilon (not 0 — allows Kraken-only fallback)
    If root not in W_base: use epsilon for W_base component
    Normalize: clamp final score to [0..1]
    """
```

**Tests — `tests/test_ralm_matrices.py`:**

```python
def test_w_base_loads_or_builds(tmp_path):
    # If w_base.json absent, build_w_base runs and produces non-empty dict

def test_w_base_normalized():
    # All values in [0,1], sum approximately 1.0

def test_update_below_min_trust_ignored():
    matrix = {}
    result = update_m_domain("ماء", ["أرض"], user_trust_score=0.1,
                              matrix=matrix, min_trust=0.3)
    assert result == {}  # no update

def test_update_adds_entry():
    matrix = {}
    result = update_m_domain("ماء", ["أرض"], user_trust_score=0.9,
                              matrix=matrix, learning_rate=0.1, min_trust=0.3)
    # Should have an entry for root of "ماء"
    assert len(result) > 0

def test_affinity_invented_word_returns_epsilon():
    score = compute_affinity("النممولي", [], w_base={}, m_domain={})
    assert score <= 0.05  # epsilon or near-zero
```

Smoke-check: `pytest tests/test_ralm_matrices.py -v`
Wait for confirmation before Step 3.

---

## Step 3 — `lexicon_engine/ralm_oracle.py`

**New file. The main oracle — integrates root extractor + matrices.**

```python
"""
RALM Oracle: Re-ranks WordToken candidates using root-affinity scores.
Called after ensemble.py, before confidence_engine/decision.py.
Non-blocking: if RALM disabled or fails, original candidates returned unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
from ocr_engine.schema import WordToken

logger = logging.getLogger(__name__)

@dataclass
class RALMResult:
    token: WordToken             # updated with ralm_score, ralm_zone, ralm_root
    zone: str                    # "accept" | "review" | "abstain"
    score: float
    root: str | None
    fallback_used: bool          # True if all candidates abstained → UNVERIFIED


class RALMOracle:
    """
    Singleton — initialized once, reused across all pipeline calls.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = config.ralm.enabled
        self._w_base = None
        self._m_domain = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        try:
            from lexicon_engine.ralm_matrices import (
                build_w_base, load_m_domain
            )
            w_base_path = Path(self.config.ralm.paths.w_base)
            m_domain_path = Path(self.config.ralm.paths.m_domain)
            
            if not w_base_path.exists():
                logger.info("Building W_base from Lane's Lexicon...")
                # Find Lane's lexicon path from config
                lanes_path = Path("data/lexicons/lanes")
                self._w_base = build_w_base(lanes_path, w_base_path)
            else:
                import json
                self._w_base = json.loads(w_base_path.read_text())
            
            self._m_domain = load_m_domain(m_domain_path)
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"RALM init failed: {e}. Disabled for this run.")
            return False

    def score_token(self, token: WordToken, context_words: list[WordToken]) -> RALMResult:
        """
        Score a single WordToken against the oracle.
        context_words: up to 2 tokens before and 2 after in reading order.
        """
        if not self.enabled or not self._ensure_initialized():
            # Passthrough: return token unchanged
            return RALMResult(token=token, zone="accept", score=1.0,
                              root=None, fallback_used=False)

        from lexicon_engine.root_extractor import extract_root
        from lexicon_engine.ralm_matrices import compute_affinity

        thresholds = self.config.ralm.thresholds
        fallback_label = self.config.ralm.fallback_label

        # Get context roots
        context_roots = []
        for w in context_words:
            r = extract_root(w.text)
            if r:
                context_roots.append(r)

        # Score each candidate
        candidates = token.candidates or [{"text": token.text,
                                           "confidence": token.confidence}]
        
        best_score = 0.0
        best_candidate = candidates[0]
        best_root = None

        for cand in candidates:
            root = extract_root(cand["text"],
                                self.config.ralm.camel.preset)
            kraken_conf = float(cand.get("confidence", 0.5))
            affinity = compute_affinity(cand["text"], context_roots,
                                        self._w_base, self._m_domain)
            combined = kraken_conf * affinity

            if combined > best_score:
                best_score = combined
                best_candidate = cand
                best_root = root

        # Determine zone
        accept_t = thresholds.accept
        review_t = thresholds.review
        fallback_used = False

        if best_score >= accept_t:
            zone = "accept"
        elif best_score >= review_t:
            zone = "review"
        else:
            zone = "abstain"
            # Fallback: use highest Kraken confidence candidate, mark UNVERIFIED
            best_candidate = max(candidates,
                                  key=lambda c: c.get("confidence", 0))
            fallback_used = True

        # Update token
        updated = token.model_copy(update={
            "ralm_score": best_score,
            "ralm_zone": zone,
            "ralm_root": best_root,
            "text": (fallback_label + " " + best_candidate["text"])
                     if fallback_used and zone == "abstain"
                     else best_candidate["text"],
        })

        return RALMResult(token=updated, zone=zone, score=best_score,
                          root=best_root, fallback_used=fallback_used)

    def score_page(self, tokens: list[WordToken]) -> list[WordToken]:
        """
        Score all tokens on a page with context windows.
        Context: 2 tokens before + 2 tokens after in token list order.
        Returns list of updated WordTokens.
        """
        if not self.enabled:
            return tokens

        window = self.config.ralm.bayesian.context_window
        results = []
        for i, token in enumerate(tokens):
            context = tokens[max(0, i-window):i] + tokens[i+1:i+1+window]
            result = self.score_token(token, context)
            results.append(result.token)
        return results

    def update(self, corrected_word: str, context_words: list[str],
               user_trust_score: float) -> None:
        """
        Update domain matrix from a crowd correction.
        Called by the feedback endpoint in routes.py.
        Persists immediately to disk.
        """
        if not self.enabled or not self._ensure_initialized():
            return
        from lexicon_engine.ralm_matrices import update_m_domain, save_m_domain
        self._m_domain = update_m_domain(
            corrected_word=corrected_word,
            context_words=context_words,
            user_trust_score=user_trust_score,
            matrix=self._m_domain,
            learning_rate=self.config.ralm.bayesian.learning_rate,
            min_trust=self.config.ralm.bayesian.min_trust_score,
        )
        save_m_domain(self._m_domain, Path(self.config.ralm.paths.m_domain))
```

**Tests — `tests/test_ralm_oracle.py`:**

```python
def test_oracle_disabled_passthrough(config_with_ralm_disabled):
    # When ralm.enabled=false, tokens unchanged

def test_invented_word_goes_to_abstain(oracle, mock_token):
    # "النممولي" → zone="abstain", fallback_used=True

def test_known_agricultural_word_accepts(oracle, mock_token):
    # "الماء" with high Kraken confidence → zone="accept"

def test_context_window_correct_size(oracle, token_list):
    # 10 tokens: verify each gets exactly min(2, available) context tokens

def test_update_persists_to_disk(oracle, tmp_path):
    # After update(), m_domain.json exists and contains the root

def test_score_page_returns_same_count(oracle, token_list):
    assert len(oracle.score_page(token_list)) == len(token_list)
```

Smoke-check: `pytest tests/test_ralm_oracle.py -v`
Wait for confirmation before Step 4.

---

## Step 4 — Wire into pipeline

**Touch `main.py` and `api/routes.py` only. No other changes.**

### 4-A  `main.py` — RALM after ensemble, before confidence engine

```python
# In run_pipeline(), after ensemble step produces page_ocr (OCRResult):

# RALM Oracle (after ensemble, before confidence engine)
if config.ralm.enabled:
    oracle = get_ralm_oracle(config)  # singleton getter
    page_ocr.words = oracle.score_page(page_ocr.words)
```

Add singleton getter near top of main.py:

```python
_ralm_oracle = None

def get_ralm_oracle(config):
    global _ralm_oracle
    if _ralm_oracle is None:
        from lexicon_engine.ralm_oracle import RALMOracle
        _ralm_oracle = RALMOracle(config)
    return _ralm_oracle
```

### 4-B  `api/routes.py` — RALM feedback endpoint

Add to the existing `submit_correction` route (extend, don't replace):

```python
# After saving correction to feedback_store:
if config.ralm.enabled:
    oracle = get_ralm_oracle(config)
    # Get context from the correction request
    context_words = body.get("context_words", [])  # list of surrounding word texts
    trust_score = body.get("user_trust_score", 0.5)  # from user session
    oracle.update(
        corrected_word=corrected_text,
        context_words=context_words,
        user_trust_score=trust_score,
    )
```

Update the correction request schema in `api/schemas.py` to add:

```python
context_words: list[str] = []       # surrounding word texts (2 before + 2 after)
user_trust_score: float = 0.5       # 0..1, default 0.5 for new users
```

---

## Step 5 — Smoke-test

```bash
# Full test suite
pytest tests/ -x -q 2>&1 | tail -10

# RALM enabled end-to-end on one page
python -c "
from utils.config import get_config
from main import get_ralm_oracle
from ocr_engine.schema import WordToken

config = get_config()
oracle = get_ralm_oracle(config)

# Test invented word
t1 = WordToken(text='النممولي', confidence=0.6, bbox=(0,0,10,10),
               page_index=0, source='kraken',
               candidates=[{'text': 'النممولي', 'confidence': 0.6}])
result = oracle.score_token(t1, [])
print(f'Invented word: zone={result.zone} score={result.score:.3f} fallback={result.fallback_used}')

# Test known word
t2 = WordToken(text='الماء', confidence=0.95, bbox=(0,0,10,10),
               page_index=0, source='kraken',
               candidates=[{'text': 'الماء', 'confidence': 0.95}])
result2 = oracle.score_token(t2, [])
print(f'Known word: zone={result2.zone} score={result2.score:.3f}')
"

# Run pipeline on 1.jpg with RALM enabled
python -m cli.main process data/test_images/1.jpg --profile andalusian_naskh --mode debug 2>&1 | python -c "
import sys, json
d = json.load(sys.stdin)
words = d.get('words', [])
zones = {}
for w in words:
    z = w.get('ralm_zone', 'none')
    zones[z] = zones.get(z, 0) + 1
print('RALM zone distribution:', zones)
"
```

Expected output:
- `النممولي` → `zone=abstain`, `fallback_used=True`
- `الماء` → `zone=accept` or `zone=review`
- Zone distribution shows mix of accept/review/abstain (not all one zone)

Report all results. Commit and push.

---

## Compatibility notes

| Component | Interaction | Risk |
|---|---|---|
| `ensemble.py` | RALM reads its output, doesn't modify it | None |
| `confidence_engine/decision.py` | Reads `ralm_zone` from TokenState | Low — additive field |
| `candidate_generator.py` | RALM runs AFTER it, uses same candidates | None — independent |
| `api/routes.py` correction endpoint | Extended with 2 new optional fields | None — backward compatible |
| `feedback_store.py` | Unchanged — RALM update is separate from feedback storage | None |

---

## Non-negotiable rules

1. RALM never modifies Kraken's original text in-place — it creates updated
   copies via `model_copy()`. Original text always preserved in `candidates`.
2. If RALM init fails for any reason, pipeline continues without RALM.
   Log WARNING once, do not raise.
3. W_base is built once and cached. Never rebuild unless `w_base.json` deleted.
4. M_domain is updated synchronously on every correction. Never async
   (race conditions on the matrix file).
5. `extract_root()` never raises. Always returns `str | None`.
6. All tests must pass with RALM both enabled and disabled.
7. `[UNVERIFIED]` label is config-driven — never hardcoded in logic.
