"""Per-source parser adapters → canonical LexiconEntry."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from confidence_engine.state import LexiconEntry
from lexicon_ingestion.sources import SourceConfig
from utils.logging import get_logger

log = get_logger(__name__)

# Arabic Unicode block
_ARABIC_RE = re.compile(r"[؀-ۿ]")
# Root header pattern used in Shamela lexicon texts: spaced letters like ( ك ت ب ) or ك ت ب
_ROOT_HEADER_RE = re.compile(
    r"^[\s()*]*"
    r"([؀-ۿ][\sـ]*"
    r"[؀-ۿ][\sـ]*"
    r"[؀-ۿ](?:[\sـ]*[؀-ۿ])?)"
    r"[\s()*]*$"
)


# ---------------------------------------------------------------------------
# ArabTeX → Arabic Unicode
# ---------------------------------------------------------------------------

def _arabtex_to_arabic(s: str) -> str:
    """Convert ArabTeX transliteration string to Arabic Unicode consonants."""
    _TWO = {
        "A^": "أ",  # أ
        "A=": "إ",  # إ
        "A_": "آ",  # آ
        "w^": "ؤ",  # ؤ
        "y^": "ئ",  # ئ
    }
    _ONE = {
        "b": "ب", "t": "ت", "v": "ث", "j": "ج",
        "H": "ح", "x": "خ", "d": "د", "*": "ذ",
        "r": "ر", "z": "ز", "s": "س", "^": "ش",
        "S": "ص", "D": "ض", "T": "ط", "Z": "ظ",
        "E": "ع", "g": "غ", "f": "ف", "q": "ق",
        "k": "ك", "l": "ل", "m": "م", "n": "ن",
        "h": "ه", "w": "و", "y": "ي", "'": "ء",
        "A": "ا", "Y": "ى",
    }
    _VOWELS = frozenset("aiuoNKF~")
    out: list[str] = []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in _TWO:
            out.append(_TWO[two])
            i += 2
        elif s[i] in _ONE:
            out.append(_ONE[s[i]])
            i += 1
        else:
            i += 1  # vowel diacritic or unknown — strip
    return "".join(out)


class DisabledSourceError(Exception):
    pass


def parse_source(source: SourceConfig) -> list[LexiconEntry]:
    """Dispatch to the correct per-source adapter."""
    adapter = source.parser_adapter
    if adapter == "fixture":
        return _parse_fixture(source)
    if adapter == "almaany_disabled":
        return _parse_almaany_disabled(source)
    if adapter == "lanes_xml":
        return _parse_lanes_xml(source)
    if adapter == "qamus_lmf":
        return _parse_qamus_lmf(source)
    if adapter == "shamela_sqlite":
        return _parse_shamela_sqlite(source)
    if adapter == "openiti":
        return _parse_openiti_markdown(source)
    if adapter in ("wordnet",):
        log.warning(f"parser adapter '{adapter}' not yet implemented; returning empty")
        return []
    log.warning(f"unknown parser adapter '{adapter}'; skipping source '{source.name}'")
    return []


# ---------------------------------------------------------------------------
# Fixture adapter
# ---------------------------------------------------------------------------

def _parse_fixture(source: SourceConfig) -> list[LexiconEntry]:
    path = Path(source.path)
    if not path.exists():
        log.warning(f"fixture path not found: {path}")
        return []
    entries: list[LexiconEntry] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
                entries.append(LexiconEntry(**raw))
            except Exception as exc:
                log.warning(f"fixture parse error line={lineno}: {exc}")
    log.debug(f"fixture loaded entries={len(entries)}")
    return entries


# ---------------------------------------------------------------------------
# Lane's Lexicon — TEI.2 XML (laneslexicon/lexicon_xml)
# ---------------------------------------------------------------------------

_LANES_SKIP_TAGS = frozenset({"form", "foreign", "pb"})


def _lanes_english_gloss(entry_el) -> str:
    """Collect English gloss text from entryFree, skipping form/foreign/pb content."""
    parts: list[str] = []
    if entry_el.text and entry_el.text.strip():
        parts.append(entry_el.text.strip())

    def _collect(el) -> None:
        for child in el:
            if child.tag not in _LANES_SKIP_TAGS:
                if child.text and child.text.strip():
                    parts.append(child.text.strip())
                _collect(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())

    _collect(entry_el)
    return " ".join(parts)


def _parse_lanes_xml(source: SourceConfig) -> list[LexiconEntry]:
    """Parse Lane's Arabic-English Lexicon from TEI.2 XML files (laneslexicon/lexicon_xml)."""
    base = Path(source.path)
    if not base.exists():
        log.warning(
            f"lanes: XML directory not found at {base}. "
            "Run: python scripts/download_lexicons.py --lanes"
        )
        return []

    xml_files = sorted(base.glob("*.xml"))
    if not xml_files:
        log.warning(f"lanes: no XML files found in {base}")
        return []

    entries: list[LexiconEntry] = []
    for xml_path in xml_files:
        try:
            entries.extend(_parse_lanes_xml_file(xml_path, source))
        except Exception as exc:
            log.warning(f"lanes: error parsing {xml_path.name}: {exc}")

    log.info(f"lanes: parsed entries={len(entries)} from {len(xml_files)} file(s)")
    return entries


def _parse_lanes_xml_file(xml_path: Path, source: SourceConfig) -> list[LexiconEntry]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root_el = tree.getroot()

    entries: list[LexiconEntry] = []

    for div2 in root_el.iter("div2"):
        raw_root = div2.get("n", "")
        root_ar = _arabtex_to_arabic(raw_root)
        if len(_ARABIC_RE.findall(root_ar)) < 2:
            continue  # skip roots with fewer than 2 Arabic consonants

        for entry_el in div2.findall("entryFree"):
            # Lemma: first <orth orig="" lang="ar"> in <form> (contains ArabTeX)
            lemma_ar = ""
            form_el = entry_el.find("form")
            if form_el is not None:
                for orth in form_el.findall("orth"):
                    if orth.get("lang") == "ar" and orth.get("orig") == "":
                        raw = (orth.text or "").strip()
                        if raw and raw != "*":
                            lemma_ar = _arabtex_to_arabic(raw)
                            break

            if not lemma_ar:
                lemma_ar = _arabtex_to_arabic(entry_el.get("key", ""))

            if not lemma_ar or not _ARABIC_RE.search(lemma_ar):
                continue

            gloss = _lanes_english_gloss(entry_el).strip()[:600]
            if len(gloss) < 15:
                continue  # cross-reference only — no substantive definition

            entries.append(LexiconEntry(
                lemma=lemma_ar,
                root=root_ar,
                pattern=None,
                gloss=gloss,
                source=source.name,
                era=source.era,
                domain=None,
                examples=[],
                priority=source.priority,
            ))

    return entries


# ---------------------------------------------------------------------------
# al-Qāmūs al-Muḥīṭ — LMF XML (ILC4CLARIN, CC BY-SA 4.0)
# ---------------------------------------------------------------------------

def _parse_qamus_lmf(source: SourceConfig) -> list[LexiconEntry]:
    """Parse al-Qāmūs al-Muḥīṭ from LMF (Lexical Markup Framework) XML files."""
    base = Path(source.path)
    if not base.exists():
        log.warning(
            f"qamus: directory not found at {base}. "
            "Run: python scripts/download_lexicons.py --qamus"
        )
        return []

    xml_files = sorted(dict.fromkeys(base.glob("**/*.xml")))
    if not xml_files:
        log.warning(f"qamus: no XML files found in {base}")
        return []

    entries: list[LexiconEntry] = []
    for xml_path in xml_files:
        try:
            entries.extend(_parse_lmf_file(xml_path, source))
        except Exception as exc:
            log.warning(f"qamus: error parsing {xml_path.name}: {exc}")

    log.info(f"qamus: parsed entries={len(entries)} from {len(xml_files)} file(s)")
    return entries


def _parse_lmf_file(xml_path: Path, source: SourceConfig) -> list[LexiconEntry]:
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(xml_path))
    root_el = tree.getroot()

    # Strip namespace if present
    def _strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    def _find_all(el, tag: str):
        # Try with and without namespace
        children = []
        for child in el:
            if _strip_ns(child.tag) == tag:
                children.append(child)
        return children

    def _find(el, tag: str):
        for child in el:
            if _strip_ns(child.tag) == tag:
                return child
        return None

    def _feat(el, att: str) -> str | None:
        for child in el:
            if _strip_ns(child.tag) == "feat":
                if child.get("att") == att:
                    return child.get("val")
        return None

    entries: list[LexiconEntry] = []

    # Walk to Lexicon element
    lexicon_el = _find(root_el, "Lexicon")
    if lexicon_el is None:
        # Root itself might be Lexicon
        lexicon_el = root_el

    for le in _find_all(lexicon_el, "LexicalEntry"):
        lemma_el = _find(le, "Lemma")
        if lemma_el is None:
            continue

        written_form = _feat(lemma_el, "writtenForm") or lemma_el.get("writtenForm", "")
        if not written_form or not _ARABIC_RE.search(written_form):
            continue

        # Root from feat or derivable
        root_val = _feat(le, "root") or _feat(le, "triliteral") or None

        # Collect senses
        senses = _find_all(le, "Sense")
        if senses:
            for sense in senses:
                defn_el = _find(sense, "Definition")
                if defn_el is not None:
                    gloss = _feat(defn_el, "writtenForm") or ""
                else:
                    gloss = _feat(sense, "gloss") or ""
                if not gloss:
                    continue
                entries.append(LexiconEntry(
                    lemma=written_form,
                    root=root_val,
                    pattern=None,
                    gloss=gloss,
                    source=source.name,
                    era=source.era,
                    domain=_feat(le, "domain"),
                    examples=[],
                    priority=source.priority,
                ))
        else:
            # No sense elements — use any available gloss feat
            gloss = _feat(le, "gloss") or _feat(le, "definition") or written_form
            entries.append(LexiconEntry(
                lemma=written_form,
                root=root_val,
                pattern=None,
                gloss=gloss,
                source=source.name,
                era=source.era,
                domain=None,
                examples=[],
                priority=source.priority,
            ))

    return entries


# ---------------------------------------------------------------------------
# Shamela4 SQLite — Lisān al-ʿArab + Tāj al-ʿArūs
# ---------------------------------------------------------------------------

def _parse_shamela_sqlite(source: SourceConfig) -> list[LexiconEntry]:
    """
    Parse a classical lexicon from a Shamela4 SQLite export.

    Shamela4 layout (post-March 2020):
      - Metadata DB:  <path>/index.db  (tables: b[id,name,auth,cat,...])
      - Book DBs:     <path>/<book_id>.db  (table: t[id,nass,page,part])

    Each lexicon entry starts with a spaced-letter root header, e.g.
      ( ك ت ب )  or  ك ت ب :
    followed by paragraphs that form the definition.
    """
    base = Path(source.path)
    if not base.exists():
        log.warning(
            f"{source.name}: Shamela directory not found at {base}. "
            "Download Shamela4 from https://huggingface.co/datasets/AuthenticIlm/Shamela4_Full_DB "
            "and place the extracted folder at data/lexicons/shamela/"
        )
        return []

    book_name = source.book_name or ""
    book_id = _shamela_find_book_id(base, book_name)
    if book_id is None:
        log.warning(f"{source.name}: book '{book_name}' not found in Shamela index")
        return []

    book_db = base / f"{book_id}.db"
    if not book_db.exists():
        log.warning(f"{source.name}: book DB not found at {book_db}")
        return []

    entries = _shamela_extract_entries(book_db, source)
    log.info(f"{source.name}: parsed entries={len(entries)} from book_id={book_id}")
    return entries


def _shamela_find_book_id(base: Path, book_name: str) -> int | None:
    """Search Shamela metadata DBs for a book by Arabic name."""
    # Shamela4 may use 'index.db', 'main.db', or a per-letter directory
    candidate_dbs = list(base.glob("*.db"))
    # Prioritise files named index/main/catalog
    candidate_dbs.sort(key=lambda p: (
        0 if p.stem in ("index", "main", "catalog", "shamela") else 1,
        p.name,
    ))

    for meta_db in candidate_dbs:
        try:
            conn = sqlite3.connect(str(meta_db))
            cur = conn.cursor()
            tables = {r[0] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            # Shamela book table is 'b'
            if "b" in tables:
                cols = {r[1] for r in cur.execute("PRAGMA table_info(b)").fetchall()}
                name_col = next((c for c in ("name", "title", "bookname") if c in cols), None)
                id_col = next((c for c in ("id", "bookid", "book_id") if c in cols), None)
                if name_col and id_col:
                    row = cur.execute(
                        f"SELECT {id_col} FROM b WHERE {name_col} LIKE ?",
                        (f"%{book_name}%",),
                    ).fetchone()
                    if row:
                        conn.close()
                        return row[0]
            conn.close()
        except Exception as exc:
            log.debug(f"shamela meta probe {meta_db.name}: {exc}")

    return None


def _shamela_extract_entries(
    book_db: Path, source: SourceConfig
) -> list[LexiconEntry]:
    """Extract LexiconEntry objects from a Shamela book SQLite file."""
    entries: list[LexiconEntry] = []
    try:
        conn = sqlite3.connect(str(book_db))
        cur = conn.cursor()

        tables = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        text_table = next((t for t in ("t", "text", "content") if t in tables), None)
        if text_table is None:
            log.warning(f"shamela: no text table in {book_db.name} (tables={tables})")
            conn.close()
            return []

        cols = {r[1] for r in cur.execute(f"PRAGMA table_info({text_table})").fetchall()}
        nass_col = next((c for c in ("nass", "text", "content", "body") if c in cols), None)
        if not nass_col:
            conn.close()
            return []

        rows = cur.execute(
            f"SELECT {nass_col} FROM {text_table} ORDER BY id"
        ).fetchall()
        conn.close()

        # Parse running text: detect root-header lines, accumulate definition text
        paragraphs = [r[0] for r in rows if r[0] and r[0].strip()]
        entries = _parse_shamela_text(paragraphs, source)

    except Exception as exc:
        log.warning(f"shamela extract error {book_db.name}: {exc}")

    return entries


def _parse_shamela_text(
    paragraphs: list[str], source: SourceConfig
) -> list[LexiconEntry]:
    """
    Walk paragraphs from a Shamela lexicon text and extract (root, definition) pairs.

    A new entry begins when a short paragraph matches the root-header pattern.
    Everything until the next header is accumulated as the definition.
    """
    entries: list[LexiconEntry] = []
    current_root: str | None = None
    current_gloss_parts: list[str] = []
    current_lemma: str | None = None

    def _flush():
        if current_root and current_gloss_parts:
            gloss = " ".join(current_gloss_parts[:3])[:500]  # cap length
            entries.append(LexiconEntry(
                lemma=current_lemma or current_root,
                root=current_root,
                pattern=None,
                gloss=gloss,
                source=source.name,
                era=source.era,
                domain=None,
                examples=[],
                priority=source.priority,
            ))

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Remove Shamela markup: page markers, HTML tags, footnote refs
        para = re.sub(r"<[^>]+>", " ", para)
        para = re.sub(r"\[[\d\s]+\]", "", para)
        para = re.sub(r"PageV\d+P\d+", "", para)
        para = para.strip()
        if not para:
            continue

        m = _ROOT_HEADER_RE.match(para)
        if m and len(para) <= 30:
            # New root section
            _flush()
            raw_root = m.group(1)
            # Collapse spaced letters: "ك ت ب" → "كتب"
            current_root = re.sub(r"\s+", "", raw_root)
            current_gloss_parts = []
            current_lemma = None
        else:
            if current_root is not None:
                if current_lemma is None and _ARABIC_RE.search(para):
                    # First substantive paragraph: extract the defined lemma
                    # (typically the first word before colon or comma)
                    first_word = re.split(r"[\s:،,؛;]", para)[0]
                    if _ARABIC_RE.search(first_word) and len(first_word) > 1:
                        current_lemma = first_word
                current_gloss_parts.append(para)

    _flush()
    return entries


# ---------------------------------------------------------------------------
# OpenITI mARkdown — generic fallback
# ---------------------------------------------------------------------------

def _parse_openiti_markdown(source: SourceConfig) -> list[LexiconEntry]:
    """
    Parse OpenITI mARkdown text files (.txt).

    mARkdown conventions used here:
      - Section headers: ### | <heading text>
      - Paragraph separator: ~~
      - Page markers: PageVxxPyyy  (ignored)
    """
    base = Path(source.path) if source.path else None
    if base is None or not base.exists():
        log.info(f"openiti source '{source.name}' path not found; skipping")
        return []

    txt_files = sorted(base.glob("*.txt")) + sorted(base.glob("**/*.txt"))
    if not txt_files:
        log.warning(f"openiti: no .txt files in {base}")
        return []

    entries: list[LexiconEntry] = []
    for txt_path in txt_files:
        try:
            entries.extend(_parse_openiti_file(txt_path, source))
        except Exception as exc:
            log.warning(f"openiti: error parsing {txt_path.name}: {exc}")

    log.info(f"openiti '{source.name}': parsed entries={len(entries)}")
    return entries


def _parse_openiti_file(txt_path: Path, source: SourceConfig) -> list[LexiconEntry]:
    entries: list[LexiconEntry] = []
    current_heading: str | None = None
    current_body: list[str] = []

    def _flush_section():
        if not current_heading or not current_body:
            return
        # Heading is the root / lemma
        lemma = re.sub(r"\s+", "", current_heading.strip())
        if not _ARABIC_RE.search(lemma):
            return
        root_candidate = lemma if len(lemma) <= 4 else None
        body_text = " ".join(current_body)
        # First sentence as gloss
        gloss = re.split(r"[.،؛\n]", body_text)[0].strip()[:400]
        if not gloss:
            return
        entries.append(LexiconEntry(
            lemma=lemma,
            root=root_candidate,
            pattern=None,
            gloss=gloss,
            source=source.name,
            era=source.era,
            domain=None,
            examples=[],
            priority=source.priority,
        ))

    with open(txt_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            # Skip metadata header lines
            if line.startswith("######") or line.startswith("#META"):
                continue
            # Section header
            if line.startswith("### |") or line.startswith("###|"):
                _flush_section()
                heading_text = line.lstrip("#").lstrip("|").strip()
                current_heading = heading_text
                current_body = []
                continue
            # Paragraph separator
            if line.strip() == "~~":
                continue
            # Page marker
            if re.match(r"^PageV\d+P\d+", line.strip()):
                continue
            if line.strip() and current_heading is not None:
                current_body.append(line.strip())

    _flush_section()
    return entries


# ---------------------------------------------------------------------------
# Disabled stub
# ---------------------------------------------------------------------------

def _parse_almaany_disabled(_source: SourceConfig) -> list[LexiconEntry]:
    raise DisabledSourceError(
        "Almaany scraping is prohibited by ToS. "
        "This source must never be enabled."
    )
