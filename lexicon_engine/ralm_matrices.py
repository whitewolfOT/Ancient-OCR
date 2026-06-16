"""
RALM matrix management: W_base (static root prior) and M_domain (Bayesian
co-occurrence matrix that grows with crowd corrections).

W_base is built once from Lane's Lexicon XML files and min-max normalized
so that common roots score near 1.0 and rare roots score lower.
(Note: values do NOT sum to 1.0 — they are rank-proportional in [0,1]
so that the affinity formula produces scores compatible with the
accept/review thresholds in config.)

M_domain starts empty (no domain adjustment) and is updated synchronously
on every crowd correction.  Absent rows in M_domain mean "no domain signal
yet" — treated as neutral (multiplier = 1.0) rather than penalizing.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# W_base — static root-frequency prior from Lane's Lexicon
# ---------------------------------------------------------------------------

_ARABIC_RE = re.compile(r'[؀-ۿ]')
# Alef variant normalization: fold أإآ → ا so w_base keys match root extractor output
_HAMZA_RE = re.compile(r'[أإآ]')


def _normalize_root(root: str) -> str:
    """Fold alef-hamza variants to bare alef for consistent w_base key lookups."""
    return _HAMZA_RE.sub('ا', root)


def _bw2ar(bw_root: str) -> str | None:
    """Convert a Buckwalter root string to Arabic script. Returns None on failure."""
    try:
        from camel_tools.utils.charmap import CharMapper
        mapper = CharMapper.builtin_mapper('bw2ar')
        result = mapper.map_string(bw_root)
        # Keep only Arabic letters
        arabic = re.sub(r'[^؀-ۿ]', '', result)
        return arabic if arabic else None
    except Exception:
        return None


def build_w_base(lexicon_path: Path, output_path: Path) -> dict[str, float]:
    """
    Build static root frequency map from Lane's Lexicon Perseus XML files.

    Iterates all .xml files under lexicon_path, counts entryFree elements
    per div2[@type='root'], converts Buckwalter roots to Arabic, and
    min-max normalizes so the most frequent root scores 1.0.

    Saves result to output_path as JSON. Returns the frequency dict.
    """
    counts: dict[str, int] = {}
    xml_files = list(lexicon_path.glob('*.xml'))
    if not xml_files:
        logger.warning(f'build_w_base: no XML files found in {lexicon_path}')

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root_el = tree.getroot()
            for div in root_el.findall('.//div2'):
                if div.get('type') != 'root':
                    continue
                bw_root = div.get('n', '')
                if not bw_root:
                    continue
                ar_root = _bw2ar(bw_root)
                if not ar_root:
                    continue
                ar_root = _normalize_root(ar_root)
                entry_count = len(div.findall('.//entryFree'))
                if entry_count == 0:
                    entry_count = 1  # count the div itself
                counts[ar_root] = counts.get(ar_root, 0) + entry_count
        except Exception as exc:
            logger.warning(f'build_w_base: failed to parse {xml_file.name}: {exc}')

    if not counts:
        logger.warning('build_w_base: no roots extracted — returning empty dict')
        return {}

    max_count = max(counts.values())
    w_base = {root: count / max_count for root, count in counts.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(w_base, ensure_ascii=False, indent=None),
        encoding='utf-8',
    )
    logger.info(f'build_w_base: {len(w_base)} roots → {output_path}')
    return w_base


# ---------------------------------------------------------------------------
# M_domain — sparse Bayesian co-occurrence matrix
# ---------------------------------------------------------------------------

def load_m_domain(path: Path) -> dict:
    """Load domain matrix from disk, or return empty dict if absent."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            logger.warning(f'load_m_domain: failed to load {path}: {exc}')
    return {}


def save_m_domain(matrix: dict, path: Path) -> None:
    """Persist domain matrix to disk after every update."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(matrix, ensure_ascii=False, indent=None),
        encoding='utf-8',
    )


def update_m_domain(
    corrected_word: str,
    context_words: list[str],
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
    - Extract root of corrected_word and each context_word via root_extractor
    - Skip any word that returns root=None
    - Normalize M_domain[root_corrected] row after update (sum to 1.0)
    - Returns updated matrix (caller must call save_m_domain to persist)
    """
    if user_trust_score < min_trust:
        return matrix

    from lexicon_engine.root_extractor import extract_root

    corrected_root = extract_root(corrected_word)
    if corrected_root is None:
        return matrix
    corrected_root = _normalize_root(corrected_root)

    context_roots = [_normalize_root(r) for w in context_words if (r := extract_root(w)) is not None]
    if not context_roots:
        return matrix

    weight = learning_rate * user_trust_score
    row = matrix.setdefault(corrected_root, {})
    for cr in context_roots:
        row[cr] = row.get(cr, 0.0) + weight

    # Normalize row so values sum to 1.0
    total = sum(row.values())
    if total > 0:
        for k in row:
            row[k] /= total

    return matrix


# ---------------------------------------------------------------------------
# Affinity scoring
# ---------------------------------------------------------------------------

def compute_affinity(
    candidate_text: str,
    context_roots: list[str],
    w_base: dict[str, float],
    m_domain: dict,
    epsilon: float = 0.01,
) -> float:
    """
    Score = W_base(root) * M_domain_score(root, context_roots)

    W_base(root): min-max normalized frequency from Lane's (0..1).
                  Common roots score near 1.0; missing roots use epsilon.
    M_domain_score: sum of co-occurrence weights with context_roots, clamped
                    to [0,1].  Defaults to 1.0 (neutral) when M_domain has
                    no entry for this root or context is empty.

    If root is None: return epsilon (not 0 — allows Kraken-only fallback).
    Final score is clamped to [epsilon, 1.0].
    """
    from lexicon_engine.root_extractor import extract_root

    root = extract_root(candidate_text)
    if root is None:
        return epsilon

    # Normalize hamza variants and look up; for defective roots ending in ء
    # (e.g. ماء) also try the classical ه-final form (ماه = م-و-ه root family)
    norm = _normalize_root(root)
    w = w_base.get(norm, None)
    if w is None and norm.endswith('ء'):
        w = w_base.get(norm[:-1] + 'ه', None)
    if w is not None:
        # Floor known roots at 0.6: any recognized Arabic root gets base credit;
        # Lane's raw frequency score (0..1) scales only the top 40% of the range.
        # Without this, the skewed entry-count distribution makes kraken_conf*w
        # unable to reach the 0.50 review threshold for most real Arabic words.
        w = 0.6 + 0.4 * w
    else:
        w = epsilon

    # M_domain: neutral (1.0) when no domain signal yet
    if not context_roots or root not in m_domain:
        m_score = 1.0
    else:
        row = m_domain[root]
        m_score = sum(row.get(cr, 0.0) for cr in context_roots)
        m_score = min(1.0, m_score)
        if m_score == 0.0:
            m_score = epsilon  # floor: don't zero-out unknown context patterns

    return float(min(1.0, max(epsilon, w * m_score)))
