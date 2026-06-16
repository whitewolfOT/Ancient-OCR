"""
RALM Oracle: Re-ranks WordToken candidates using root-affinity scores.
Called after ensemble.py, before confidence_engine/decision.py.
Non-blocking: if RALM disabled or fails, original candidates returned unchanged.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ocr_engine.schema import WordToken

logger = logging.getLogger(__name__)


@dataclass
class RALMResult:
    token: WordToken          # updated with ralm_score, ralm_zone, ralm_root
    zone: str                 # "accept" | "review" | "abstain"
    score: float
    root: str | None
    fallback_used: bool       # True if all candidates abstained → UNVERIFIED label


class RALMOracle:
    """
    Singleton — initialized once, reused across all pipeline calls.
    If initialization fails for any reason, RALM is silently disabled for
    that run. Pipeline always continues.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = getattr(getattr(config, 'ralm', None), 'enabled', False)
        self._w_base: dict[str, float] = {}
        self._m_domain: dict = {}
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        if not self.enabled:
            return False
        try:
            from lexicon_engine.ralm_matrices import build_w_base, load_m_domain

            w_base_path = Path(self.config.ralm.paths.w_base)
            m_domain_path = Path(self.config.ralm.paths.m_domain)

            if not w_base_path.exists():
                logger.info("RALM: building W_base from Lane's Lexicon…")
                lanes_path = Path("data/lexicons/lanes")
                self._w_base = build_w_base(lanes_path, w_base_path)
            else:
                self._w_base = json.loads(w_base_path.read_text(encoding='utf-8'))

            self._m_domain = load_m_domain(m_domain_path)
            self._initialized = True
            logger.info(f"RALM: initialized — {len(self._w_base)} roots in W_base")
            return True
        except Exception as exc:
            logger.warning(f"RALM init failed: {exc}. Disabled for this run.")
            return False

    def score_token(self, token: WordToken,
                    context_words: list[WordToken]) -> RALMResult:
        """
        Score a single WordToken against the oracle.
        context_words: up to config.ralm.bayesian.context_window tokens
                       before and after in reading order.
        """
        if not self.enabled or not self._ensure_initialized():
            return RALMResult(token=token, zone="accept", score=1.0,
                              root=None, fallback_used=False)

        from lexicon_engine.root_extractor import extract_root
        from lexicon_engine.ralm_matrices import compute_affinity

        thresholds = self.config.ralm.thresholds
        accept_t = thresholds.accept
        review_t = thresholds.review
        fallback_label = self.config.ralm.fallback_label
        camel_preset = self.config.ralm.camel.preset

        # Context roots from surrounding tokens
        context_roots = []
        for w in context_words:
            r = extract_root(w.text, camel_preset)
            if r:
                context_roots.append(r)

        # Use WordToken.candidates (simple {text, confidence} dicts) if present
        candidates = token.candidates or [{"text": token.text,
                                           "confidence": token.confidence}]

        best_score = 0.0
        best_candidate = candidates[0]
        best_root: str | None = None

        for cand in candidates:
            cand_text = str(cand.get("text", ""))
            kraken_conf = float(cand.get("confidence", 0.5))
            root = extract_root(cand_text, camel_preset)
            affinity = compute_affinity(
                cand_text, context_roots, self._w_base, self._m_domain
            )
            combined = kraken_conf * affinity

            if combined > best_score:
                best_score = combined
                best_candidate = cand
                best_root = root

        # Determine zone
        fallback_used = False
        if best_score >= accept_t:
            zone = "accept"
        elif best_score >= review_t:
            zone = "review"
        else:
            zone = "abstain"
            # Fallback: highest Kraken confidence candidate, mark UNVERIFIED
            best_candidate = max(candidates,
                                 key=lambda c: float(c.get("confidence", 0)))
            fallback_used = True

        new_text = best_candidate.get("text", token.text)
        if fallback_used:
            new_text = f"{fallback_label} {new_text}"

        updated = token.model_copy(update={
            "text": new_text,
            "ralm_score": best_score,
            "ralm_zone": zone,
            "ralm_root": best_root,
        })

        return RALMResult(token=updated, zone=zone, score=best_score,
                          root=best_root, fallback_used=fallback_used)

    def score_page(self, tokens: list[WordToken]) -> list[WordToken]:
        """
        Score all tokens on a page with sliding context windows.
        Context: config.ralm.bayesian.context_window tokens before + after.
        Returns list of updated WordTokens (same length as input).
        """
        if not self.enabled:
            return tokens

        window = int(getattr(getattr(
            getattr(self.config, 'ralm', None), 'bayesian', None
        ), 'context_window', 2))

        results = []
        for i, token in enumerate(tokens):
            ctx_before = tokens[max(0, i - window):i]
            ctx_after = tokens[i + 1:i + 1 + window]
            context = ctx_before + ctx_after
            result = self.score_token(token, context)
            results.append(result.token)
        return results

    def update(self, corrected_word: str, context_words: list[str],
               user_trust_score: float) -> None:
        """
        Update domain matrix from a crowd correction.
        Called by the feedback endpoint in routes.py.
        Persists to disk immediately (synchronous — no race conditions).
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
