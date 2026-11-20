"""Official Bandai FAQ & errata PDF/text parser and idempotent SQLite ingestor."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal

import pymupdf

from optcg_knowledge.db.models import RulingRecord, SourceType, authority_for_source
from optcg_knowledge.db.schema import set_meta, utc_now_iso
from optcg_knowledge.ingest.chunking import build_ruling_chunk, upsert_chunks

logger = logging.getLogger(__name__)

FaqSourceMode = Literal["auto", "official_faq", "errata"]

SUPPORTED_FAQ_SUFFIXES: frozenset[str] = frozenset({".pdf", ".txt", ".html", ".htm"})
_HTML_VOID_TAGS: frozenset[str] = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)

# Canonical OPTCG card IDs: OP05-119, ST10-001, EB01-001, PRB01-001, P-001
_CARD_ID_RE = re.compile(
    r"\b(?:(?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})\b",
    re.IGNORECASE,
)

# Date patterns to mask before scanning for rule numbers.
_DATE_TOKEN_RE = re.compile(r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2}-\d{4})\b")

# Explicit rule context: "Rule 7-1-4-1", "Rules 7-1-4 and 7-1-5", "Section 10-1"
_RULE_CONTEXT_RE = re.compile(
    r"\b(?:Comprehensive\s+Rules?|Rules?|Sections?|Sec\.)\s+"
    r"(\d+(?:-\d+){1,4}(?:\s*(?:,|/|&|\band\b|\bor\b)\s*\d+(?:-\d+){1,4})*)",
    re.IGNORECASE,
)
_RULE_ID_RE = re.compile(r"\b(\d+(?:-\d+){1,4})\b")
_DEEP_RULE_ID_RE = re.compile(r"\b(\d{1,2}(?:-\d{1,2}){2,4})\b")

# Header metadata detection.
_RULES_VERSION_RE = re.compile(
    r"\b(?:Comprehensive\s+Rules?|Rules?)\s+(?:Ver\.?|Version)\s*[:\-]?\s*(\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)
_DATE_HEADER_RE = re.compile(
    r"(?:Effective(?:\s+Date)?|Last\s+Updated|Updated|Revised|Revision\s+Date|Date|As\s+of)"
    r"\s*[:\-]?\s*"
    r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})",
    re.IGNORECASE,
)
_FILENAME_DATE_RE = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")

# Page noise patterns (headers, footers, page counters, copyright notices).
_NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^-\s*\d{1,4}\s*-$"),
    re.compile(r"^Page\s+\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE),
    re.compile(r"^\d{1,3}\s*/\s*\d{1,3}$"),
    re.compile(r"^©.*(?:Eiichiro\s+Oda|Shueisha|Bandai).*$", re.IGNORECASE),
    re.compile(
        r"^ONE\s+PIECE\s+CARD\s+GAME(?:\s+Official)?\s+"
        r"(?:Card\s+)?(?:Q\s*&\s*A|FAQ|Errata|Card\s+Errata|Rules\s+Q\s*&\s*A).*$",
        re.IGNORECASE,
    ),
)

# Section resets for general (non-card-specific) Q&A sections.
_GENERAL_SECTION_RE = re.compile(
    r"^(?:General(?:\s+Rules)?|Rules|Keyword|Game\s+Rules|Miscellaneous)\s+(?:Q\s*&\s*A|FAQ)s?$",
    re.IGNORECASE,
)

# Card header patterns in official Q&A and errata sheets.
_EXPLICIT_CARD_NO_RE = re.compile(
    r"^(?:Card\s*(?:No\.?|ID|Number)\s*[:\-]?\s*|\[)"
    r"((?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})"
    r"\]?(?:\s*[\|:\-—]?\s*(.*))?$",
    re.IGNORECASE,
)
_EXPLICIT_CARD_NAME_RE = re.compile(r"^Card\s*Name\s*[:\-]\s*(.+)$", re.IGNORECASE)
_INLINE_CARD_HEADER_RE = re.compile(
    r"^((?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})(?:\s*[\|:\-—]\s*|\s+)(.*)$",
    re.IGNORECASE,
)
_CARD_ONLY_LINE_RE = re.compile(
    r"^((?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})$",
    re.IGNORECASE,
)

# Question / Answer markers (supports inline and standalone 2-column PDF extractions).
_Q_EXPLICIT_RE = re.compile(r"^(?:Q(?:uestion)?\s*\d*\s*[:.)]|Q\d+\b)\s*(.*)$", re.IGNORECASE)
_Q_STANDALONE_RE = re.compile(r"^Q(?:uestion)?\s*\d*$", re.IGNORECASE)
_Q_BARE_RE = re.compile(r"^Q(?:\t|\s+)(.+)$")

_A_EXPLICIT_RE = re.compile(r"^(?:A(?:nswer)?\s*\d*\s*[:.)]|A\d+\b)\s*(.*)$", re.IGNORECASE)
_A_STANDALONE_RE = re.compile(r"^A(?:nswer)?\s*\d*$", re.IGNORECASE)
_A_WIDE_GAP_RE = re.compile(r"^A(?:\t|\s{2,})(.+)$")
_A_BARE_RE = re.compile(r"^A\s+([A-Z\"\[0-9].*)$")

# Errata block markers.
_ERRATA_BEFORE_RE = re.compile(
    r"^(?:Before(?:\s+Correction)?|Pre[\-\s]?Errata|Incorrect(?:\s+Text)?|"
    r"Original(?:\s+Text)?|Old\s+Text|Error)\s*(?:[:\-]\s*(.*))?$",
    re.IGNORECASE,
)
_ERRATA_AFTER_RE = re.compile(
    r"^(?:After(?:\s+Correction)?|Post[\-\s]?Errata|Correct(?:ed)?(?:\s+Text)?|"
    r"New\s+Text|Correction)\s*(?:[:\-]\s*(.*))?$",
    re.IGNORECASE,
)
_ERRATA_NOTE_RE = re.compile(
    r"^(?:Note|Notes|Reason|Details?|Summary|Remarks?)\s*[:\-]\s*(.*)$",
    re.IGNORECASE,
)
_ERRATA_INLINE_DATE_RE = re.compile(
    r"^(?:Effective(?:\s+Date)?|Updated|Revised|Date)\s*[:\-]\s*"
    r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})$",
    re.IGNORECASE,
)

_PRESERVE_BREAK_PREFIXES = ("Example:", "Note:", "※", "•", "-", "*")
_HTML_BR_RE = re.compile(r"\s*<br\s*/?>\s*", re.IGNORECASE)
_PDF_META_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")
_PROSE_WORDS_RE = re.compile(
    r"\b(?:is|are|was|were|can|cannot|will|would|does|do|when|if|while|during|your|opponent)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedFaqEntry:
    """Intermediate parsed ruling keeping contextual vs standalone rule IDs."""

    ruling: RulingRecord
    explicit_rule_ids: tuple[str, ...] = ()
    standalone_rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ParsedFaqDocument:
    """Output of parsing a single official FAQ or errata PDF/text document."""

    source_name: str
    source: SourceType
    entries: tuple[ParsedFaqEntry, ...]
    date: str | None = None
    rules_version: str | None = None
    url: str | None = None

    @property
    def rulings(self) -> tuple[RulingRecord, ...]:
        """Return the parsed RulingRecord items for convenience."""
        return tuple(entry.ruling for entry in self.entries)


@dataclass(frozen=True, slots=True)
class FaqIngestSummary:
    """Counters returned after ingesting one or more official FAQ/errata documents."""

    files_processed: int
    rulings_upserted: int
    official_faq_count: int
    errata_count: int
    card_links_upserted: int
    rule_links_upserted: int
    chunks_upserted: int


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _clean_html_breaks(text: str) -> str:
    """Replace literal ``<br>`` tags from Bandai CMS exports with clean line breaks."""
    if "<br" not in text.lower():
        return text.strip()
    parts = [part.strip() for part in _HTML_BR_RE.split(text)]
    return "\n".join(part for part in parts if part)


def _normalize_for_hash(text: str) -> str:
    return _normalize_whitespace(text).lower()


def _rule_sort_key(rule_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in rule_id.split("-"))


def _normalize_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    cleaned = " ".join(raw_date.strip().split())
    if not cleaned:
        return None

    # Normalize slash or dot separated YYYY/M/D or YYYY.M.D
    ymd_match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", cleaned)
    if ymd_match:
        year, month, day = (int(part) for part in ymd_match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return cleaned

    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return cleaned


def _extract_date_from_filename(source_name: str) -> str | None:
    match = _FILENAME_DATE_RE.search(source_name)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _is_noise_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in _NOISE_LINE_PATTERNS)


def _append_continuation(existing: str, line: str) -> str:
    if not existing:
        return line
    if line.startswith(_PRESERVE_BREAK_PREFIXES):
        return f"{existing}\n{line}"
    if existing.endswith("-") and not existing.endswith(" -"):
        return f"{existing[:-1]}{line}"
    return f"{existing} {line}"


def _extract_card_ids(text: str, *, primary_card_id: str | None = None) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    if primary_card_id:
        norm_primary = primary_card_id.upper()
        found.append(norm_primary)
        seen.add(norm_primary)
    for match in _CARD_ID_RE.finditer(text):
        cid = match.group(0).upper()
        if cid not in seen:
            seen.add(cid)
            found.append(cid)
    return tuple(found)


def _is_plausible_rule_id(rule_id: str) -> bool:
    parts = rule_id.split("-")
    if len(parts) < 2 or len(parts) > 5:
        return False
    nums = [int(part) for part in parts]
    # Comprehensive Rules top-level sections are small integers (currently 1..15);
    # reject 4-digit years or oversized values.
    if nums[0] < 1 or nums[0] > 30:
        return False
    return all(0 <= num <= 99 for num in nums[1:])


def _extract_rule_ids(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(explicit_context_rule_ids, standalone_deep_rule_ids)``."""
    masked = _DATE_TOKEN_RE.sub(lambda m: " " * len(m.group(0)), text)
    masked = _CARD_ID_RE.sub(lambda m: " " * len(m.group(0)), masked)

    explicit: set[str] = set()
    for ctx_match in _RULE_CONTEXT_RE.finditer(masked):
        group_text = ctx_match.group(1)
        for rid_match in _RULE_ID_RE.finditer(group_text):
            rid = rid_match.group(1)
            if _is_plausible_rule_id(rid):
                explicit.add(rid)

    standalone: set[str] = set()
    for deep_match in _DEEP_RULE_ID_RE.finditer(masked):
        rid = deep_match.group(1)
        if rid not in explicit and _is_plausible_rule_id(rid):
            standalone.add(rid)

    return (
        tuple(sorted(explicit, key=_rule_sort_key)),
        tuple(sorted(standalone, key=_rule_sort_key)),
    )


def _merge_rule_ids(
    explicit_ids: Iterable[str],
    standalone_ids: Iterable[str],
    *,
    known_rule_ids: set[str] | None = None,
) -> tuple[str, ...]:
    merged: set[str] = set(explicit_ids)
    for rid in standalone_ids:
        if known_rule_ids is None or not known_rule_ids or rid in known_rule_ids:
            merged.add(rid)
    return tuple(sorted(merged, key=_rule_sort_key))


def _make_ruling_id(
    *,
    source: SourceType,
    primary_card: str | None,
    hash_parts: Sequence[str],
) -> str:
    prefix = "errata" if source == "errata" else "faq"
    scope = primary_card.upper() if primary_card else "general"
    payload = "|".join([source, scope, *(_normalize_for_hash(part) for part in hash_parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{scope}:{digest}"


def _parse_card_header(line: str) -> tuple[str, str | None, bool] | None:
    """Detect whether a line is a card section header.

    Returns ``(card_id, card_name, is_explicit)`` where ``is_explicit`` is True for
    unambiguous delimiters (``Card No.:``, ``[OP05-119]``, ``OP05-119 | ...``, or ID-only).
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None

    explicit = _EXPLICIT_CARD_NO_RE.match(stripped)
    if explicit:
        cid = explicit.group(1).upper()
        name = (explicit.group(2) or "").strip() or None
        return cid, name, True

    only_id = _CARD_ONLY_LINE_RE.match(stripped)
    if only_id:
        return only_id.group(1).upper(), None, True

    inline = _INLINE_CARD_HEADER_RE.match(stripped)
    if not inline:
        return None

    cid = inline.group(1).upper()
    rest = inline.group(2).strip()
    if not rest:
        return cid, None, True

    has_explicit_sep = bool(
        re.match(
            r"^(?:(?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})\s*[\|:\-—]\s*",
            stripped,
            re.IGNORECASE,
        )
    )

    # Guard against prose sentences that merely happen to start with a card ID.
    if len(rest) > 80:
        return None
    if any(ch in rest for ch in ("?", ";", ",")):
        return None
    if rest.endswith(".") and not re.search(r"(?:K\.O\.|Jr\.|Sr\.|\b[A-Z]\.)$", rest):
        return None
    if not has_explicit_sep and _PROSE_WORDS_RE.search(rest):
        return None

    return cid, rest, has_explicit_sep


def _extract_document_metadata(
    lines: Sequence[tuple[int | None, str]],
    *,
    source_name: str,
    date_override: str | None,
    rules_version_override: str | None,
) -> tuple[str | None, str | None]:
    header_sample = "\n".join(text for _, text in lines[:60] if text)

    detected_version = rules_version_override
    if detected_version is None:
        ver_match = _RULES_VERSION_RE.search(header_sample)
        if ver_match:
            detected_version = ver_match.group(1)

    detected_date = _normalize_date(date_override) if date_override else None
    if detected_date is None:
        date_match = _DATE_HEADER_RE.search(header_sample)
        if date_match:
            detected_date = _normalize_date(date_match.group(1))
    if detected_date is None:
        detected_date = _extract_date_from_filename(source_name)

    return detected_date, detected_version


def _detect_source_type(
    lines: Sequence[tuple[int | None, str]],
    *,
    source_name: str,
    source_mode: FaqSourceMode,
) -> SourceType:
    if source_mode in ("official_faq", "errata"):
        return source_mode

    lower_name = source_name.lower()
    if "errata" in lower_name or "correction" in lower_name:
        return "errata"

    before_hits = 0
    after_hits = 0
    qa_hits = 0
    for _, line in lines:
        if _ERRATA_BEFORE_RE.match(line):
            before_hits += 1
        elif _ERRATA_AFTER_RE.match(line):
            after_hits += 1
        elif _Q_EXPLICIT_RE.match(line) or _Q_STANDALONE_RE.match(line):
            qa_hits += 1

    if before_hits > 0 and after_hits > 0 and qa_hits == 0:
        return "errata"
    return "official_faq"


def _match_question_start(line: str, *, state: str) -> tuple[bool, str]:
    explicit = _Q_EXPLICIT_RE.match(line)
    if explicit:
        return True, explicit.group(1).strip()

    if _Q_STANDALONE_RE.match(line):
        return True, ""

    if state != "Q":
        bare = _Q_BARE_RE.match(line)
        if bare:
            return True, bare.group(1).strip()

    return False, ""


def _match_answer_start(line: str, *, state: str, current_question: str) -> tuple[bool, str]:
    explicit = _A_EXPLICIT_RE.match(line)
    if explicit:
        return True, explicit.group(1).strip()

    if state == "Q":
        if _A_STANDALONE_RE.match(line):
            return True, ""
        wide = _A_WIDE_GAP_RE.match(line)
        if wide:
            return True, wide.group(1).strip()
        bare = _A_BARE_RE.match(line)
        if bare and current_question.rstrip().endswith(("?", '?"', "?)")):
            return True, bare.group(1).strip()

    return False, ""


def _format_raw_ref(source_name: str, page_num: int | None) -> str:
    if page_num is not None:
        return f"{source_name}#p{page_num}"
    return source_name


def _build_entry(
    *,
    source: SourceType,
    question: str,
    answer: str,
    section_card_id: str | None,
    date: str | None,
    rules_version: str | None,
    url: str | None,
    raw_ref: str,
    hash_parts: Sequence[str],
    seen_ids: set[str],
) -> ParsedFaqEntry:
    combined_text = f"{question}\n{answer}"
    card_ids = _extract_card_ids(combined_text, primary_card_id=section_card_id)
    primary_card = section_card_id or (card_ids[0] if card_ids else None)
    explicit_rules, standalone_rules = _extract_rule_ids(combined_text)
    all_rules = _merge_rule_ids(explicit_rules, standalone_rules)

    ruling_id = _make_ruling_id(
        source=source,
        primary_card=primary_card,
        hash_parts=hash_parts,
    )
    if ruling_id in seen_ids:
        ruling_id = _make_ruling_id(
            source=source,
            primary_card=primary_card,
            hash_parts=[*hash_parts, answer],
        )
    seen_ids.add(ruling_id)

    ruling = RulingRecord(
        ruling_id=ruling_id,
        question=_clean_html_breaks(question),
        answer=_clean_html_breaks(answer),
        source=source,
        authority=authority_for_source(source),
        date=date,
        rules_version=rules_version,
        url=url,
        superseded_by=None,
        reviewed=True,
        raw_ref=raw_ref,
        card_ids=card_ids,
        rule_ids=all_rules,
    )
    return ParsedFaqEntry(
        ruling=ruling,
        explicit_rule_ids=explicit_rules,
        standalone_rule_ids=standalone_rules,
    )


def _parse_qa_lines(
    lines: Sequence[tuple[int | None, str]],
    *,
    source_name: str,
    date: str | None,
    rules_version: str | None,
    url: str | None,
) -> list[ParsedFaqEntry]:
    entries: list[ParsedFaqEntry] = []
    seen_ids: set[str] = set()

    current_card_id: str | None = None
    state: Literal["IDLE", "Q", "A"] = "IDLE"
    q_text = ""
    a_text = ""
    entry_page: int | None = None

    def flush() -> None:
        nonlocal q_text, a_text, entry_page, state
        cleaned_q = q_text.strip()
        cleaned_a = a_text.strip()
        if cleaned_q and cleaned_a:
            entries.append(
                _build_entry(
                    source="official_faq",
                    question=cleaned_q,
                    answer=cleaned_a,
                    section_card_id=current_card_id,
                    date=date,
                    rules_version=rules_version,
                    url=url,
                    raw_ref=_format_raw_ref(source_name, entry_page),
                    hash_parts=(cleaned_q,),
                    seen_ids=seen_ids,
                )
            )
        q_text = ""
        a_text = ""
        entry_page = None
        state = "IDLE"

    for page_num, raw_line in lines:
        line = raw_line.strip()
        if not line or _is_noise_line(line):
            continue

        if _GENERAL_SECTION_RE.match(line):
            flush()
            current_card_id = None
            continue

        if _EXPLICIT_CARD_NAME_RE.match(line) and state != "Q":
            continue

        if state != "Q":
            card_header = _parse_card_header(line)
            if card_header is not None:
                cid, _cname, is_explicit = card_header
                can_break_answer = (
                    state != "A"
                    or is_explicit
                    or not a_text.strip()
                    or a_text.rstrip().endswith((".", "!", "?", '."', ".)", "”"))
                )
                if can_break_answer:
                    flush()
                    current_card_id = cid
                    continue

        is_q, q_body = _match_question_start(line, state=state)
        if is_q:
            flush()
            state = "Q"
            q_text = q_body
            entry_page = page_num
            continue

        is_a, a_body = _match_answer_start(line, state=state, current_question=q_text)
        if is_a and state == "Q":
            state = "A"
            a_text = a_body
            continue

        if state == "Q":
            q_text = _append_continuation(q_text, line)
        elif state == "A":
            a_text = _append_continuation(a_text, line)

    flush()
    return entries


def _parse_errata_lines(
    lines: Sequence[tuple[int | None, str]],
    *,
    source_name: str,
    date: str | None,
    rules_version: str | None,
    url: str | None,
) -> list[ParsedFaqEntry]:
    entries: list[ParsedFaqEntry] = []
    seen_ids: set[str] = set()

    current_card_id: str | None = None
    current_card_name: str | None = None
    entry_date: str | None = date
    entry_page: int | None = None

    state: Literal["IDLE", "BEFORE", "AFTER", "NOTE"] = "IDLE"
    before_text = ""
    after_text = ""
    note_text = ""

    def flush() -> None:
        nonlocal before_text, after_text, note_text, state, entry_page, entry_date
        cleaned_before = before_text.strip()
        cleaned_after = after_text.strip()
        cleaned_note = note_text.strip()
        if cleaned_before and cleaned_after:
            if current_card_id and current_card_name:
                question = (
                    f"Errata for {current_card_id} ({current_card_name}): "
                    "what is the corrected card text?"
                )
            elif current_card_id:
                question = f"Errata for {current_card_id}: what is the corrected card text?"
            else:
                question = f"Official errata ({source_name}): what is the corrected text?"

            answer_lines = [
                f"Before: {cleaned_before}",
                f"After: {cleaned_after}",
            ]
            if cleaned_note:
                answer_lines.append(f"Note: {cleaned_note}")
            answer = "\n".join(answer_lines)

            entries.append(
                _build_entry(
                    source="errata",
                    question=question,
                    answer=answer,
                    section_card_id=current_card_id,
                    date=entry_date or date,
                    rules_version=rules_version,
                    url=url,
                    raw_ref=_format_raw_ref(source_name, entry_page),
                    hash_parts=(cleaned_before, cleaned_after),
                    seen_ids=seen_ids,
                )
            )
            entry_date = date
        before_text = ""
        after_text = ""
        note_text = ""
        entry_page = None
        state = "IDLE"

    for page_num, raw_line in lines:
        line = raw_line.strip()
        if not line or _is_noise_line(line):
            continue

        # Ignore document-level date/version banners before any entry starts.
        inline_date = _ERRATA_INLINE_DATE_RE.match(line)
        if inline_date:
            parsed_inline_date = _normalize_date(inline_date.group(1))
            if state == "IDLE":
                entry_date = parsed_inline_date or entry_date
                continue
            if state in ("AFTER", "NOTE") and before_text and after_text:
                entry_date = parsed_inline_date or entry_date
                continue

        card_name_match = _EXPLICIT_CARD_NAME_RE.match(line)
        if card_name_match and state == "IDLE":
            current_card_name = card_name_match.group(1).strip() or current_card_name
            continue

        card_header = _parse_card_header(line)
        if card_header is not None and state != "BEFORE":
            cid, cname, is_explicit = card_header
            can_break = (
                state == "IDLE"
                or is_explicit
                or (bool(before_text) and bool(after_text))
            )
            if can_break:
                flush()
                current_card_id, current_card_name = cid, cname
                entry_page = page_num
                continue

        before_match = _ERRATA_BEFORE_RE.match(line)
        if before_match:
            if before_text and after_text:
                flush()
            state = "BEFORE"
            if entry_page is None:
                entry_page = page_num
            body = (before_match.group(1) or "").strip()
            before_text = body
            continue

        after_match = _ERRATA_AFTER_RE.match(line)
        if after_match and (state in ("BEFORE", "IDLE") or before_text):
            state = "AFTER"
            body = (after_match.group(1) or "").strip()
            after_text = body
            continue

        note_match = _ERRATA_NOTE_RE.match(line)
        if note_match and state in ("AFTER", "NOTE"):
            state = "NOTE"
            body = (note_match.group(1) or "").strip()
            note_text = body
            continue

        if state == "BEFORE":
            before_text = _append_continuation(before_text, line)
        elif state == "AFTER":
            after_text = _append_continuation(after_text, line)
        elif state == "NOTE":
            note_text = _append_continuation(note_text, line)

    flush()
    return entries


def _extract_pdf_metadata_date(doc: pymupdf.Document) -> str | None:
    """Extract an ISO YYYY-MM-DD date from PDF ``modDate`` or ``creationDate`` metadata."""
    meta = doc.metadata or {}
    for key in ("modDate", "creationDate"):
        raw = meta.get(key)
        if not raw:
            continue
        match = _PDF_META_DATE_RE.search(str(raw))
        if not match:
            continue
        year, month, day = (int(part) for part in match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _cluster_coords(values: Sequence[float], *, tolerance: float = 3.0) -> list[float]:
    """Cluster nearby 1D coordinates into sorted representative line positions."""
    if not values:
        return []
    ordered = sorted(values)
    clusters: list[list[float]] = [[ordered[0]]]
    for val in ordered[1:]:
        if val - clusters[-1][-1] <= tolerance:
            clusters[-1].append(val)
        else:
            clusters.append([val])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _extract_page_grid_rows(page: pymupdf.Page) -> list[list[str]]:
    """Extract table cells using PDF vector grid lines and clipped page text.

    Using ``page.get_text("text", clip=cell_rect)`` preserves exact word spacing and
    punctuation on Bandai's Excel-exported FAQ PDFs while avoiding row-splitting quirks
    on short final pages.
    """
    h_ys: list[float] = []
    v_xs: list[float] = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        width = float(rect.width)
        height = float(rect.height)
        if width >= 200.0 and height <= 4.0:
            h_ys.append((float(rect.y0) + float(rect.y1)) / 2.0)
        elif height >= 10.0 and width <= 4.0:
            v_xs.append((float(rect.x0) + float(rect.x1)) / 2.0)
        elif width >= 30.0 and height >= 10.0 and drawing.get("color") is not None:
            h_ys.extend([float(rect.y0), float(rect.y1)])
            v_xs.extend([float(rect.x0), float(rect.x1)])

    ys = _cluster_coords(h_ys, tolerance=3.0)
    xs = _cluster_coords(v_xs, tolerance=3.0)
    if len(ys) < 2 or len(xs) < 3:
        return []

    rows: list[list[str]] = []
    for row_idx in range(len(ys) - 1):
        if ys[row_idx + 1] - ys[row_idx] < 6.0:
            continue
        row: list[str] = []
        for col_idx in range(len(xs) - 1):
            if xs[col_idx + 1] - xs[col_idx] < 10.0:
                continue
            cell_rect = pymupdf.Rect(xs[col_idx], ys[row_idx], xs[col_idx + 1], ys[row_idx + 1])
            raw_cell = page.get_text("text", clip=cell_rect).replace("\u00a0", " ")
            cell_text = _normalize_whitespace(raw_cell)
            row.append(cell_text)
        if any(row):
            rows.append(row)
    return rows


def _classify_table_header(
    row: Sequence[str],
) -> tuple[Literal["card_qa", "rules_qa", "errata"], dict[str, int]] | None:
    """Identify a Bandai official FAQ or errata table header row and column indices."""
    normalized = [re.sub(r"[\s.:_\-]+", "", cell.lower()) for cell in row]
    col_map: dict[str, int] = {}

    for idx, token in enumerate(normalized):
        if token in ("cardno", "cardid", "cardnumber"):
            col_map["card_no"] = idx
        elif token == "cardname":
            col_map["card_name"] = idx
        elif token == "category":
            col_map["category"] = idx
        elif token in ("question", "q"):
            col_map["question"] = idx
        elif token in ("answer", "a"):
            col_map["answer"] = idx
        elif token in ("before", "beforecorrection", "preerrata", "incorrect", "original"):
            col_map["before"] = idx
        elif token in ("after", "aftercorrection", "posterrata", "correct", "corrected"):
            col_map["after"] = idx
        elif token in ("note", "notes", "reason", "details"):
            col_map["note"] = idx

    if "before" in col_map and "after" in col_map:
        return ("errata", col_map)
    if "question" in col_map and "answer" in col_map:
        if "card_no" in col_map:
            return ("card_qa", col_map)
        if "category" in col_map:
            return ("rules_qa", col_map)
    return None


def _parse_pdf_table_document(
    doc: pymupdf.Document,
    *,
    source_name: str,
    source_mode: FaqSourceMode,
    date: str | None,
    rules_version: str | None,
    url: str | None,
) -> ParsedFaqDocument | None:
    """Parse Bandai's official tabular PDFs (``Card No. | Card Name | Question | Answer``)."""
    table_kind: Literal["card_qa", "rules_qa", "errata"] | None = None
    col_map: dict[str, int] = {}
    paged_rows: list[tuple[int, list[str]]] = []

    for page_num, page in enumerate(doc, start=1):
        rows = _extract_page_grid_rows(page)
        for row in rows:
            header = _classify_table_header(row)
            if header is not None:
                if table_kind is None:
                    table_kind, col_map = header
                continue
            paged_rows.append((page_num, row))

    if table_kind is None or not paged_rows:
        return None

    first_page_lines = [
        (1, line) for line in doc[0].get_text("text").replace("\r\n", "\n").splitlines()
    ]
    detected_date, detected_version = _extract_document_metadata(
        first_page_lines,
        source_name=source_name,
        date_override=date,
        rules_version_override=rules_version,
    )
    if detected_date is None:
        detected_date = _extract_pdf_metadata_date(doc)

    if source_mode in ("official_faq", "errata"):
        resolved_source: SourceType = source_mode
    else:
        resolved_source = "errata" if table_kind == "errata" else "official_faq"

    entries: list[ParsedFaqEntry] = []
    seen_ids: set[str] = set()

    for page_num, row in paged_rows:
        if table_kind == "errata":
            before_idx = col_map["before"]
            after_idx = col_map["after"]
            if max(before_idx, after_idx) >= len(row):
                continue
            before_text = _clean_html_breaks(row[before_idx])
            after_text = _clean_html_breaks(row[after_idx])
            if not before_text or not after_text:
                continue

            card_id: str | None = None
            card_name: str | None = None
            if "card_no" in col_map and col_map["card_no"] < len(row):
                m = _CARD_ID_RE.search(row[col_map["card_no"]])
                if m:
                    card_id = m.group(0).upper()
            if "card_name" in col_map and col_map["card_name"] < len(row):
                card_name = row[col_map["card_name"]].strip() or None

            if card_id and card_name:
                question = f"Errata for {card_id} ({card_name}): what is the corrected card text?"
            elif card_id:
                question = f"Errata for {card_id}: what is the corrected card text?"
            else:
                question = f"Official errata ({source_name}): what is the corrected text?"

            answer_lines = [f"Before: {before_text}", f"After: {after_text}"]
            if "note" in col_map and col_map["note"] < len(row):
                note_text = _clean_html_breaks(row[col_map["note"]])
                if note_text:
                    answer_lines.append(f"Note: {note_text}")

            entries.append(
                _build_entry(
                    source=resolved_source,
                    question=question,
                    answer="\n".join(answer_lines),
                    section_card_id=card_id,
                    date=detected_date,
                    rules_version=detected_version,
                    url=url,
                    raw_ref=_format_raw_ref(source_name, page_num),
                    hash_parts=(before_text, after_text),
                    seen_ids=seen_ids,
                )
            )
            continue

        q_idx = col_map["question"]
        a_idx = col_map["answer"]
        if max(q_idx, a_idx) >= len(row):
            continue
        question_text = _clean_html_breaks(row[q_idx])
        answer_text = _clean_html_breaks(row[a_idx])
        if not question_text or not answer_text:
            continue

        section_card_id: str | None = None
        if table_kind == "card_qa" and "card_no" in col_map and col_map["card_no"] < len(row):
            card_match = _CARD_ID_RE.search(row[col_map["card_no"]])
            if card_match:
                section_card_id = card_match.group(0).upper()

        entries.append(
            _build_entry(
                source=resolved_source,
                question=question_text,
                answer=answer_text,
                section_card_id=section_card_id,
                date=detected_date,
                rules_version=detected_version,
                url=url,
                raw_ref=_format_raw_ref(source_name, page_num),
                hash_parts=(question_text,),
                seen_ids=seen_ids,
            )
        )

    if not entries:
        return None

    return ParsedFaqDocument(
        source_name=source_name,
        source=resolved_source,
        entries=tuple(entries),
        date=detected_date,
        rules_version=detected_version,
        url=url,
    )


def extract_faq_pages(pdf_path: Path | str) -> list[tuple[int, str]]:
    """Extract plain text per 1-indexed page from an official FAQ/errata PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"FAQ PDF not found: {path}")

    pages: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            pages.append((index, page.get_text("text")))
    return pages


def _pages_to_lines(pages: Sequence[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
    result: list[tuple[int | None, str]] = []
    for page_num, page_text in pages:
        normalized = page_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        for raw_line in normalized.split("\n"):
            result.append((page_num, raw_line))
    return result


def parse_faq_pages(
    pages: Sequence[tuple[int | None, str]],
    *,
    source_name: str,
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
    fallback_date: str | None = None,
) -> ParsedFaqDocument:
    """Parse extracted page-text pairs into structured official FAQ or errata entries."""
    lines = _pages_to_lines(pages)
    detected_date, detected_version = _extract_document_metadata(
        lines,
        source_name=source_name,
        date_override=date,
        rules_version_override=rules_version,
    )
    if detected_date is None:
        detected_date = fallback_date
    resolved_source = _detect_source_type(
        lines,
        source_name=source_name,
        source_mode=source_mode,
    )

    if resolved_source == "errata":
        entries = _parse_errata_lines(
            lines,
            source_name=source_name,
            date=detected_date,
            rules_version=detected_version,
            url=url,
        )
    else:
        entries = _parse_qa_lines(
            lines,
            source_name=source_name,
            date=detected_date,
            rules_version=detected_version,
            url=url,
        )

    return ParsedFaqDocument(
        source_name=source_name,
        source=resolved_source,
        entries=tuple(entries),
        date=detected_date,
        rules_version=detected_version,
        url=url,
    )


def parse_faq_text(
    raw_text: str,
    *,
    source_name: str = "faq.txt",
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
) -> ParsedFaqDocument:
    """Parse raw official FAQ or errata plain text into structured RulingRecord entries."""
    return parse_faq_pages(
        [(None, raw_text)],
        source_name=source_name,
        source_mode=source_mode,
        date=date,
        rules_version=rules_version,
        url=url,
    )


def _parse_pdf_with_name(
    pdf_path: Path,
    *,
    source_name: str,
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
) -> ParsedFaqDocument:
    if not pdf_path.is_file():
        raise FileNotFoundError(f"FAQ PDF not found: {pdf_path}")

    with pymupdf.open(pdf_path) as doc:
        table_doc = _parse_pdf_table_document(
            doc,
            source_name=source_name,
            source_mode=source_mode,
            date=date,
            rules_version=rules_version,
            url=url,
        )
        if table_doc is not None:
            return table_doc

        meta_date = _extract_pdf_metadata_date(doc)
        pages: list[tuple[int | None, str]] = [
            (index, page.get_text("text")) for index, page in enumerate(doc, start=1)
        ]

    return parse_faq_pages(
        pages,
        source_name=source_name,
        source_mode=source_mode,
        date=date,
        rules_version=rules_version,
        url=url,
        fallback_date=meta_date,
    )


def parse_faq_pdf(
    pdf_path: Path | str,
    *,
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
) -> ParsedFaqDocument:
    """Extract and parse an official Bandai FAQ/errata PDF."""
    path = Path(pdf_path)
    return _parse_pdf_with_name(
        path,
        source_name=path.name,
        source_mode=source_mode,
        date=date,
        rules_version=rules_version,
        url=url,
    )


def _extract_date_from_text(raw_text: str | None) -> str | None:
    if not raw_text:
        return None
    cleaned = _normalize_whitespace(raw_text)
    direct = _normalize_date(cleaned)
    if direct and re.fullmatch(r"\d{4}-\d{2}-\d{2}", direct):
        return direct
    match = re.search(
        r"([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})",
        cleaned,
    )
    if match:
        return _normalize_date(match.group(1))
    return None


class _BandaiErrataHTMLParser(HTMLParser):
    """Extract structured errata blocks from Bandai's official errata_card HTML page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._section_date: str | None = None
        self._in_section_date = False
        self._section_date_tag: str | None = None
        self._buf: list[str] = []

        self._entry_depth: int | None = None
        self._entry: dict[str, object] | None = None
        self._field: str | None = None
        self._current_dt: str | None = None
        self.blocks: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        if lower_tag not in _HTML_VOID_TAGS:
            self._depth += 1

        attr_map = {k.lower(): (v or "") for k, v in attrs}
        cls_tokens = set(attr_map.get("class", "").split())

        if lower_tag == "br":
            if self._field or self._in_section_date:
                self._buf.append("\n")
            return

        if lower_tag == "h4" and "mediumTit" in cls_tokens:
            self._in_section_date = True
            self._section_date_tag = "h4"
            self._buf = []
            return
        if lower_tag == "div" and "dateText" in cls_tokens:
            self._in_section_date = True
            self._section_date_tag = "div"
            self._buf = []
            return

        if (
            self._entry_depth is None
            and lower_tag in ("div", "section")
            and ("errataModal" in cls_tokens or "detailCol" in cls_tokens)
        ):
            self._entry_depth = self._depth
            self._entry = {
                "id": attr_map.get("id") or None,
                "section_date": self._section_date,
                "h5": "",
                "card_num": "",
                "card_name": "",
                "pairs": [],
            }
            return

        if self._entry is not None:
            if lower_tag == "h5":
                self._field = "h5"
                self._buf = []
            elif lower_tag == "p" and "cardNum" in cls_tokens:
                self._field = "card_num"
                self._buf = []
            elif lower_tag == "h6" and "cardName" in cls_tokens:
                self._field = "card_name"
                self._buf = []
            elif lower_tag == "dt":
                self._field = "dt"
                self._buf = []
            elif lower_tag == "dd":
                self._field = "dd"
                self._buf = []

    def handle_data(self, data: str) -> None:
        if self._in_section_date or self._field is not None:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        lower_tag = tag.lower()
        if lower_tag in _HTML_VOID_TAGS:
            return

        if self._in_section_date and lower_tag == self._section_date_tag:
            self._section_date = _normalize_whitespace("".join(self._buf))
            self._in_section_date = False
            self._section_date_tag = None
            self._buf = []
        elif self._entry is not None:
            if self._field == "h5" and lower_tag == "h5":
                self._entry["h5"] = "".join(self._buf).strip()
                self._field = None
                self._buf = []
            elif self._field == "card_num" and lower_tag == "p":
                self._entry["card_num"] = _normalize_whitespace("".join(self._buf))
                self._field = None
                self._buf = []
            elif self._field == "card_name" and lower_tag == "h6":
                self._entry["card_name"] = _normalize_whitespace("".join(self._buf))
                self._field = None
                self._buf = []
            elif self._field == "dt" and lower_tag == "dt":
                self._current_dt = _normalize_whitespace("".join(self._buf))
                self._field = None
                self._buf = []
            elif self._field == "dd" and lower_tag == "dd":
                lines = [
                    _normalize_whitespace(line)
                    for line in "".join(self._buf).splitlines()
                    if line.strip()
                ]
                dd_text = "\n".join(lines)
                if self._current_dt and dd_text:
                    pairs = self._entry["pairs"]
                    if isinstance(pairs, list):
                        pairs.append((self._current_dt, dd_text))
                self._current_dt = None
                self._field = None
                self._buf = []

            if self._depth == self._entry_depth:
                self.blocks.append(self._entry)
                self._entry = None
                self._entry_depth = None

        self._depth -= 1


def parse_faq_html(
    raw_html: str,
    *,
    source_name: str = "errata_card.html",
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
) -> ParsedFaqDocument:
    """Parse Bandai's official errata HTML page (``errata_card``) into RulingRecord entries."""
    parser = _BandaiErrataHTMLParser()
    parser.feed(raw_html)

    resolved_source: SourceType = (
        source_mode if source_mode in ("official_faq", "errata") else "errata"
    )
    override_date = _normalize_date(date) if date else None

    entries: list[ParsedFaqEntry] = []
    seen_ids: set[str] = set()
    latest_date: str | None = override_date

    for block in parser.blocks:
        pairs = block.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            continue

        h5_raw = str(block.get("h5") or "")
        h5_lines = [_normalize_whitespace(line) for line in h5_raw.splitlines() if line.strip()]

        card_id: str | None = None
        card_name: str | None = None
        entry_date: str | None = override_date

        card_num_str = str(block.get("card_num") or "").strip()
        card_name_str = str(block.get("card_name") or "").strip()
        if card_num_str:
            m = _CARD_ID_RE.search(card_num_str)
            if m:
                card_id = m.group(0).upper()
        if card_name_str:
            card_name = card_name_str

        for hline in h5_lines:
            if card_id is None:
                m = _CARD_ID_RE.search(hline)
                if m:
                    card_id = m.group(0).upper()
                    remainder = hline[m.end() :].strip(" |:-—")
                    if remainder and not card_name:
                        card_name = remainder
                    continue
            if entry_date is None:
                maybe_date = _extract_date_from_text(hline)
                if maybe_date:
                    entry_date = maybe_date

        if entry_date is None:
            entry_date = _extract_date_from_text(str(block.get("section_date") or ""))

        if entry_date and (latest_date is None or entry_date > latest_date):
            latest_date = entry_date

        before_parts: list[str] = []
        after_parts: list[str] = []
        note_parts: list[str] = []

        for dt_label, dd_text in pairs:
            low_dt = dt_label.lower().strip()
            if low_dt.startswith(("before", "pre-errata", "pre errata", "incorrect", "original")):
                before_parts.append(dd_text)
            elif low_dt.startswith(("after", "post-errata", "post errata", "correct")):
                if "rule revision" in low_dt:
                    note_parts.append(dt_label)
                after_parts.append(dd_text)
            elif low_dt.startswith(("note", "reason", "detail", "remark")):
                note_parts.append(dd_text)
            else:
                note_parts.append(f"{dt_label} {dd_text}")

        before_text = "\n".join(before_parts).strip()
        after_text = "\n".join(after_parts).strip()
        note_text = "\n".join(note_parts).strip()

        if not before_text and not after_text:
            continue

        if card_id and card_name:
            question = f"Errata for {card_id} ({card_name}): what is the corrected card text?"
        elif card_id:
            question = f"Errata for {card_id}: what is the corrected card text?"
        else:
            question = f"Official errata ({source_name}): what is the corrected text?"

        answer_lines: list[str] = []
        if before_text:
            answer_lines.append(f"Before: {before_text}")
        if after_text:
            answer_lines.append(f"After: {after_text}")
        if note_text:
            answer_lines.append(f"Note: {note_text}")

        block_id = block.get("id")
        raw_ref = f"{source_name}#{block_id}" if block_id else source_name

        entries.append(
            _build_entry(
                source=resolved_source,
                question=question,
                answer="\n".join(answer_lines),
                section_card_id=card_id,
                date=entry_date,
                rules_version=rules_version,
                url=url,
                raw_ref=raw_ref,
                hash_parts=(before_text, after_text),
                seen_ids=seen_ids,
            )
        )

    return ParsedFaqDocument(
        source_name=source_name,
        source=resolved_source,
        entries=tuple(entries),
        date=latest_date,
        rules_version=rules_version,
        url=url,
    )


def parse_faq_file(
    file_path: Path | str,
    *,
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
) -> ParsedFaqDocument:
    """Parse a single `.pdf`, `.html`, `.htm`, or `.txt` official FAQ/errata file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"FAQ file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_faq_pdf(
            path,
            source_mode=source_mode,
            date=date,
            rules_version=rules_version,
            url=url,
        )
    if suffix in (".html", ".htm"):
        return parse_faq_html(
            path.read_text(encoding="utf-8", errors="replace"),
            source_name=path.name,
            source_mode=source_mode,
            date=date,
            rules_version=rules_version,
            url=url,
        )
    if suffix == ".txt":
        return parse_faq_text(
            path.read_text(encoding="utf-8"),
            source_name=path.name,
            source_mode=source_mode,
            date=date,
            rules_version=rules_version,
            url=url,
        )

    supported = ", ".join(sorted(SUPPORTED_FAQ_SUFFIXES))
    raise ValueError(f"Unsupported FAQ file type {path.suffix!r}; expected one of: {supported}")


def parse_faq_path(
    path: Path | str,
    *,
    source_mode: FaqSourceMode = "auto",
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
) -> list[ParsedFaqDocument]:
    """Parse a single FAQ/errata file or all supported files in a directory."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"FAQ path not found: {target}")

    if target.is_file():
        return [
            parse_faq_file(
                target,
                source_mode=source_mode,
                date=date,
                rules_version=rules_version,
                url=url,
            )
        ]

    files = sorted(
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_FAQ_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(
            f"No official FAQ/errata files ({', '.join(sorted(SUPPORTED_FAQ_SUFFIXES))}) "
            f"found in {target}"
        )

    docs: list[ParsedFaqDocument] = []
    for file_path in files:
        rel_name = file_path.relative_to(target).as_posix()
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            docs.append(
                _parse_pdf_with_name(
                    file_path,
                    source_name=rel_name,
                    source_mode=source_mode,
                    date=date,
                    rules_version=rules_version,
                    url=url,
                )
            )
        elif suffix in (".html", ".htm"):
            docs.append(
                parse_faq_html(
                    file_path.read_text(encoding="utf-8", errors="replace"),
                    source_name=rel_name,
                    source_mode=source_mode,
                    date=date,
                    rules_version=rules_version,
                    url=url,
                )
            )
        else:
            docs.append(
                parse_faq_text(
                    file_path.read_text(encoding="utf-8"),
                    source_name=rel_name,
                    source_mode=source_mode,
                    date=date,
                    rules_version=rules_version,
                    url=url,
                )
            )
    return docs


def ingest_faqs(
    conn: sqlite3.Connection,
    documents: ParsedFaqDocument | Sequence[ParsedFaqDocument],
) -> FaqIngestSummary:
    """Idempotently ingest official FAQ and errata rulings into SQLite.

    Preserves existing ``superseded_by`` links and ``reviewed`` flags on re-ingestion,
    never deletes outdated rulings, and updates linked ``ruling_cards``, ``ruling_rules``,
    and ``chunks`` transactionally.
    """
    doc_list = [documents] if isinstance(documents, ParsedFaqDocument) else list(documents)
    if not doc_list:
        raise ValueError("Cannot ingest empty FAQ document list.")

    known_rule_ids = {
        str(row[0]) for row in conn.execute("SELECT rule_id FROM rules").fetchall()
    }

    rulings_by_id: dict[str, RulingRecord] = {}

    for doc in doc_list:
        for entry in doc.entries:
            base = entry.ruling
            # When the rules table is populated, filter standalone (non-contextual) rule IDs
            # against known Comprehensive Rules to prevent false positives.
            filtered_rule_ids = _merge_rule_ids(
                entry.explicit_rule_ids,
                entry.standalone_rule_ids,
                known_rule_ids=known_rule_ids if known_rule_ids else None,
            )
            ruling = (
                base
                if filtered_rule_ids == base.rule_ids
                else RulingRecord(
                    ruling_id=base.ruling_id,
                    question=base.question,
                    answer=base.answer,
                    source=base.source,
                    authority=base.authority,
                    date=base.date,
                    rules_version=base.rules_version,
                    url=base.url,
                    superseded_by=base.superseded_by,
                    reviewed=base.reviewed,
                    raw_ref=base.raw_ref,
                    card_ids=base.card_ids,
                    rule_ids=filtered_rule_ids,
                )
            )
            rulings_by_id[ruling.ruling_id] = ruling

    if not rulings_by_id:
        raise ValueError("No official FAQ or errata entries were parsed from the provided input.")

    ruling_rows: list[tuple[object, ...]] = []
    card_links_by_ruling: dict[str, tuple[str, ...]] = {}
    rule_links_by_ruling: dict[str, tuple[str, ...]] = {}
    chunks = []
    official_faq_count = 0
    errata_count = 0

    for ruling in rulings_by_id.values():
        if ruling.source == "errata":
            errata_count += 1
        else:
            official_faq_count += 1

        ruling_rows.append(
            (
                ruling.ruling_id,
                ruling.question,
                ruling.answer,
                ruling.source,
                ruling.authority,
                ruling.date,
                ruling.rules_version,
                ruling.url,
                ruling.superseded_by,
                1 if ruling.reviewed else 0,
                ruling.raw_ref,
            )
        )
        card_links_by_ruling[ruling.ruling_id] = ruling.card_ids
        rule_links_by_ruling[ruling.ruling_id] = ruling.rule_ids
        chunks.append(build_ruling_chunk(ruling))

    card_links_upserted = 0
    rule_links_upserted = 0

    with conn:
        conn.executemany(
            """
            INSERT INTO rulings (
                ruling_id, question, answer, source, authority,
                date, rules_version, url, superseded_by, reviewed, raw_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ruling_id) DO UPDATE SET
                question = excluded.question,
                answer = excluded.answer,
                source = excluded.source,
                authority = excluded.authority,
                date = COALESCE(excluded.date, rulings.date),
                rules_version = COALESCE(excluded.rules_version, rulings.rules_version),
                url = COALESCE(excluded.url, rulings.url),
                superseded_by = COALESCE(rulings.superseded_by, excluded.superseded_by),
                reviewed = MAX(rulings.reviewed, excluded.reviewed),
                raw_ref = COALESCE(excluded.raw_ref, rulings.raw_ref)
            """,
            ruling_rows,
        )

        for ruling_id, card_ids in card_links_by_ruling.items():
            conn.execute("DELETE FROM ruling_cards WHERE ruling_id = ?", (ruling_id,))
            if card_ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO ruling_cards (ruling_id, card_id) VALUES (?, ?)",
                    [(ruling_id, cid) for cid in card_ids],
                )
                card_links_upserted += len(card_ids)

        for ruling_id, rule_ids in rule_links_by_ruling.items():
            conn.execute("DELETE FROM ruling_rules WHERE ruling_id = ?", (ruling_id,))
            if rule_ids:
                conn.executemany(
                    "INSERT OR IGNORE INTO ruling_rules (ruling_id, rule_id) VALUES (?, ?)",
                    [(ruling_id, rid) for rid in rule_ids],
                )
                rule_links_upserted += len(rule_ids)

        chunks_upserted = upsert_chunks(conn, chunks)
        now = utc_now_iso()
        set_meta(conn, "last_faq_ingested_at", now)
        set_meta(conn, "last_ingested_at", now)

    logger.info(
        "Ingested %d official rulings (%d faq, %d errata) from %d file(s)",
        len(ruling_rows),
        official_faq_count,
        errata_count,
        len(doc_list),
    )

    return FaqIngestSummary(
        files_processed=len(doc_list),
        rulings_upserted=len(ruling_rows),
        official_faq_count=official_faq_count,
        errata_count=errata_count,
        card_links_upserted=card_links_upserted,
        rule_links_upserted=rule_links_upserted,
        chunks_upserted=chunks_upserted,
    )
