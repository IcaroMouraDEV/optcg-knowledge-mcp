"""Official Bandai Tournament Rules Manual (TRM) PDF parser and idempotent SQLite ingestor."""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from optcg_knowledge.db.models import TrmSectionRecord
from optcg_knowledge.db.schema import set_meta, utc_now_iso
from optcg_knowledge.ingest.chunking import build_trm_chunk, upsert_chunks
from optcg_knowledge.ingest.dates import normalize_official_date

logger = logging.getLogger(__name__)

TRM_ID_PREFIX = "TRM-"

# Matches numbered TRM section headers like "1.0 Event Participants" or "7.4.1 Game Play Error".
_TRM_HEADER_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+([A-Z][^\n]{1,120})$")
# Matches a bare section number on its own line inside the PDF Table of Contents (e.g., "1.1").
_BARE_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)+)$")
# Matches TOC lines ending with dotted leaders or wide spacing + page number.
_TOC_LINE_RE = re.compile(r"^.*(?:\.{3,}\s*|\s{3,})\d{1,3}\s*$")
# Matches numbered infraction entries inside sections like 7.4.1 / 7.4.2 / 7.4.3.
_INFRACTION_ITEM_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]{2,110})$")
_INFRACTION_SIGNAL_RE = re.compile(r"\b(?:Penalty|Resolution)\s*:", re.IGNORECASE)

_VERSION_RE = re.compile(r"\b(?:Ver\.?|Version)\s*[:\-]?\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_EFFECTIVE_DATE_RE = re.compile(
    r"(?:Last\s+Updated|Effective(?:\s+Date)?|Updated)\s*[:\-]?\s*"
    r"([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)

_NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^-\s*\d{1,3}\s*-$"),
    re.compile(r"^Page\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^BANDAI\s+ORGANIZED\s+PLAY$", re.IGNORECASE),
    re.compile(r"^ONE\s+PIECE\s+CARD\s+GAME\s+Tournament\s+Rules\s+Manual.*$", re.IGNORECASE),
    re.compile(r"^©.*(?:Eiichiro\s+Oda|Shueisha|Bandai).*$", re.IGNORECASE),
)
_PRESERVE_BREAK_PREFIXES = (
    "Penalty:",
    "Resolution:",
    "Examples:",
    "Example:",
    "Note:",
    "**",
    "•",
    "·",
    "o ",
    "- ",
    "* ",
)
_BARE_BULLET_MARKERS = frozenset({"•", "·", "o", "-", "*"})


@dataclass(frozen=True, slots=True)
class ParsedTrmDocument:
    """Output of parsing an official Bandai Tournament Rules Manual PDF or text."""

    sections: tuple[TrmSectionRecord, ...]
    trm_version: str | None
    effective_date: str | None


@dataclass(frozen=True, slots=True)
class TrmIngestSummary:
    """Counters returned after ingesting Tournament Rules Manual sections."""

    sections_upserted: int
    sections_removed: int
    chunks_upserted: int
    trm_version: str | None
    effective_date: str | None


def extract_trm_pdf_text(pdf_path: Path | str) -> str:
    """Extract plain text from every page of a Tournament Rules Manual PDF via PyMuPDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tournament Rules Manual PDF not found: {path}")

    pages: list[str] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def normalize_trm_section_id(raw_id: str) -> str:
    """Normalize a TRM section ID (e.g. ``'7.4.1'`` or ``'trm-7.4.1'``) to ``'TRM-7.4.1'``."""
    cleaned = raw_id.strip().rstrip(".")
    if not cleaned:
        return ""
    if cleaned.upper().startswith(TRM_ID_PREFIX):
        suffix = cleaned[len(TRM_ID_PREFIX) :].strip().rstrip(".")
        return f"{TRM_ID_PREFIX}{suffix}" if suffix else ""
    return f"{TRM_ID_PREFIX}{cleaned}"


def _trm_sort_key(section_id: str) -> tuple[tuple[int, int | str], ...]:
    """Natural numeric sort key so ``TRM-7.4.2`` sorts before ``TRM-7.4.10``."""
    raw = (
        section_id[len(TRM_ID_PREFIX) :]
        if section_id.upper().startswith(TRM_ID_PREFIX)
        else section_id
    )
    parts: list[tuple[int, int | str]] = []
    for segment in raw.split("."):
        if segment.isdigit():
            parts.append((0, int(segment)))
        else:
            parts.append((1, segment.lower()))
    return tuple(parts)


def _section_depth(section_id: str) -> int:
    """Return hierarchy depth (0 for ``TRM-1.0``, 1 for ``TRM-1.1``, 2 for ``TRM-7.4.1``, etc.)."""
    raw = (
        section_id[len(TRM_ID_PREFIX) :]
        if section_id.upper().startswith(TRM_ID_PREFIX)
        else section_id
    )
    parts = raw.split(".")
    if len(parts) == 2 and parts[1] == "0":
        return 0
    return max(len(parts) - 1, 1)


def _resolve_trm_parent_id(section_id: str, known_ids: set[str]) -> str | None:
    """Resolve the nearest existing parent TRM section ID."""
    raw = (
        section_id[len(TRM_ID_PREFIX) :]
        if section_id.upper().startswith(TRM_ID_PREFIX)
        else section_id
    )
    parts = raw.split(".")
    if len(parts) == 2 and parts[1] == "0":
        return None

    while len(parts) > 1:
        parts.pop()
        if len(parts) == 1:
            chapter_candidate = f"{TRM_ID_PREFIX}{parts[0]}.0"
            if chapter_candidate in known_ids and chapter_candidate != section_id:
                return chapter_candidate
            break
        candidate = f"{TRM_ID_PREFIX}{'.'.join(parts)}"
        if candidate in known_ids and candidate != section_id:
            return candidate
    return None


def _is_noise_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in _NOISE_LINE_PATTERNS)


def _extract_metadata(
    lines: list[str],
    *,
    trm_version: str | None,
    effective_date: str | None,
) -> tuple[str | None, str | None]:
    header_sample = "\n".join(lines[:60])

    detected_version = trm_version.strip() if trm_version and trm_version.strip() else None
    if detected_version is None:
        match = _VERSION_RE.search(header_sample)
        if match:
            detected_version = match.group(1)

    detected_date = normalize_official_date(effective_date)
    if detected_date is None:
        match = _EFFECTIVE_DATE_RE.search(header_sample)
        if match:
            detected_date = normalize_official_date(match.group(1))

    return (detected_version, detected_date)


def _append_body_line(existing: str, line: str) -> str:
    if not existing:
        return line
    if line.startswith(_PRESERVE_BREAK_PREFIXES):
        return f"{existing}\n{line}"
    if _INFRACTION_ITEM_RE.match(line) and not line.endswith("."):
        return f"{existing}\n{line}"
    if existing.endswith("-") and not existing.endswith(" -"):
        return f"{existing[:-1]}{line}"
    # Preserve line break after Penalty/Resolution/Examples or numbered infraction headers.
    last_line = existing.rsplit("\n", 1)[-1]
    if last_line in {"Penalty:", "Resolution:", "Examples:"}:
        return f"{existing}\n{line}"
    if _INFRACTION_ITEM_RE.match(last_line) and not last_line.endswith("."):
        return f"{existing}\n{line}"
    return f"{existing} {line}"


def _split_infraction_subsections(
    entries: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Split numbered infraction items (e.g., inside 7.4.1/7.4.2/7.4.3) into child sections.

    Only splits when a section contains >= 2 sequentially numbered headings starting at 1
    and at least half of those blocks include ``Penalty:`` or ``Resolution:``, preventing
    false splits on ordinary numbered lists elsewhere in the manual.
    """
    expanded: list[tuple[str, str, str]] = []
    for num_id, title, body in entries:
        lines = body.split("\n")
        candidate_indices: list[tuple[int, int, str]] = []
        expected_next = 1

        for idx, line in enumerate(lines):
            match = _INFRACTION_ITEM_RE.match(line.strip())
            if not match:
                continue
            item_num = int(match.group(1))
            item_title = match.group(2).strip()
            if item_title.endswith(".") or ":" in item_title:
                continue
            if item_num == expected_next:
                candidate_indices.append((idx, item_num, item_title))
                expected_next += 1

        if len(candidate_indices) < 2:
            expanded.append((num_id, title, body))
            continue

        blocks: list[tuple[int, str, str]] = []
        for pos, (start_idx, item_num, item_title) in enumerate(candidate_indices):
            end_idx = (
                candidate_indices[pos + 1][0]
                if pos + 1 < len(candidate_indices)
                else len(lines)
            )
            sub_body = "\n".join(lines[start_idx + 1 : end_idx]).strip()
            blocks.append((item_num, item_title, sub_body))

        infraction_hits = sum(
            1 for _, _, sub_body in blocks if _INFRACTION_SIGNAL_RE.search(sub_body)
        )
        if infraction_hits * 2 < len(blocks):
            expanded.append((num_id, title, body))
            continue

        intro = "\n".join(lines[: candidate_indices[0][0]]).strip()
        expanded.append((num_id, title, intro))
        for item_num, item_title, sub_body in blocks:
            expanded.append((f"{num_id}.{item_num}", item_title, sub_body))

    return expanded


def parse_trm_text(
    raw_text: str,
    *,
    trm_version: str | None = None,
    effective_date: str | None = None,
) -> ParsedTrmDocument:
    """Parse extracted Tournament Rules Manual text into hierarchical TrmSectionRecord entries."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    raw_lines = [line.strip() for line in normalized.split("\n")]

    detected_version, detected_date = _extract_metadata(
        raw_lines,
        trm_version=trm_version,
        effective_date=effective_date,
    )

    # First pass: skip cover/TOC/page numbers and group section headings with their body text.
    entries: list[tuple[str, str, str]] = []
    seen_num_ids: set[str] = set()
    toc_seen_ids: set[str] = set()
    in_toc = False
    pending_bullet: str | None = None

    for line in raw_lines:
        if not line:
            continue
        if _is_noise_line(line):
            continue

        if line.lower() in {"contents", "table of contents"}:
            in_toc = True
            pending_bullet = None
            continue

        if _TOC_LINE_RE.match(line):
            in_toc = True
            pending_bullet = None
            toc_header = _TRM_HEADER_RE.match(line)
            if toc_header:
                toc_seen_ids.add(toc_header.group(1))
            continue

        if in_toc:
            bare_match = _BARE_SECTION_NUM_RE.match(line)
            if bare_match:
                toc_seen_ids.add(bare_match.group(1))
                continue

        header_match = _TRM_HEADER_RE.match(line)
        if header_match:
            num_id, title = header_match.group(1), header_match.group(2).strip()
            top_num = int(num_id.split(".", 1)[0])
            plausible_header = (
                1 <= top_num <= 30
                and not title.endswith(".")
                and not re.match(r"^(?:minutes?|hours?|days?|points?|cards?)\b", title, re.I)
            )

            if plausible_header:
                if in_toc:
                    if num_id in toc_seen_ids or num_id == "1.0":
                        in_toc = False
                    else:
                        toc_seen_ids.add(num_id)
                        continue

                pending_bullet = None
                if num_id in seen_num_ids:
                    # Handle duplicate section numbers in official PDFs (e.g. "8.4 Setup Examples").
                    logger.warning("Duplicate TRM section %s (%s); merging text", num_id, title)
                    for idx, (existing_id, existing_title, existing_body) in enumerate(entries):
                        if existing_id == num_id:
                            merged_body = (
                                f"{existing_body}\n{title}" if existing_body else title
                            )
                            entries[idx] = (existing_id, existing_title, merged_body)
                            break
                    continue

                seen_num_ids.add(num_id)
                entries.append((num_id, title, ""))
                continue

        if in_toc or not entries:
            continue

        # Merge PDF-isolated bullet glyphs ("•", "·", or "o") with the following text line.
        if line in _BARE_BULLET_MARKERS:
            pending_bullet = "•" if line == "·" else line
            continue

        effective_line = f"{pending_bullet} {line}" if pending_bullet else line
        pending_bullet = None

        prev_id, prev_title, prev_body = entries[-1]
        entries[-1] = (prev_id, prev_title, _append_body_line(prev_body, effective_line))

    expanded_entries = _split_infraction_subsections(entries)

    # Build top-level chapter titles (e.g., "7.0 Tournament Penalties").
    chapter_titles: dict[str, str] = {}
    for num_id, title, _ in expanded_entries:
        parts = num_id.split(".")
        if len(parts) == 2 and parts[1] == "0":
            chapter_titles[parts[0]] = f"{num_id} {title}"

    prefixed_entries = [
        (f"{TRM_ID_PREFIX}{num_id}", num_id, title, body.strip())
        for num_id, title, body in expanded_entries
    ]
    known_ids = {section_id for section_id, _, _, _ in prefixed_entries}
    ordered_entries = sorted(prefixed_entries, key=lambda item: _trm_sort_key(item[0]))

    records: list[TrmSectionRecord] = []
    for section_id, num_id, title, body in ordered_entries:
        chapter_key = num_id.split(".", 1)[0]
        chapter_label = chapter_titles.get(chapter_key, f"Section {chapter_key}.0")
        parent_id = _resolve_trm_parent_id(section_id, known_ids)
        records.append(
            TrmSectionRecord(
                section_id=section_id,
                parent_id=parent_id,
                section=chapter_label,
                title=title,
                text=body,
                trm_version=detected_version,
                effective_date=detected_date,
            )
        )

    return ParsedTrmDocument(
        sections=tuple(records),
        trm_version=detected_version,
        effective_date=detected_date,
    )


def parse_trm_pdf(
    pdf_path: Path | str,
    *,
    trm_version: str | None = None,
    effective_date: str | None = None,
) -> ParsedTrmDocument:
    """Extract and parse an official Tournament Rules Manual PDF."""
    raw_text = extract_trm_pdf_text(pdf_path)
    return parse_trm_text(
        raw_text,
        trm_version=trm_version,
        effective_date=effective_date,
    )


def ingest_trm(
    conn: sqlite3.Connection,
    document: ParsedTrmDocument,
) -> TrmIngestSummary:
    """Idempotently ingest a parsed Tournament Rules Manual document into SQLite."""
    if not document.sections:
        raise ValueError("No TRM sections found in document; refusing to overwrite existing TRM")

    sections_sorted = sorted(
        document.sections,
        key=lambda s: (_section_depth(s.section_id), _trm_sort_key(s.section_id)),
    )
    sections_by_id = {s.section_id: s for s in sections_sorted}
    chunks = [build_trm_chunk(section, sections_by_id) for section in sections_sorted]
    new_section_ids = set(sections_by_id)

    with conn:
        existing_ids = {
            str(row[0])
            for row in conn.execute("SELECT section_id FROM trm_sections").fetchall()
        }
        removed_ids = sorted(
            existing_ids - new_section_ids,
            key=lambda sid: (-_section_depth(sid), _trm_sort_key(sid)),
        )

        conn.executemany(
            """
            INSERT INTO trm_sections (
                section_id, parent_id, section, title, text, trm_version, effective_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(section_id) DO UPDATE SET
                parent_id = excluded.parent_id,
                section = excluded.section,
                title = excluded.title,
                text = excluded.text,
                trm_version = excluded.trm_version,
                effective_date = excluded.effective_date
            """,
            [
                (
                    s.section_id,
                    s.parent_id,
                    s.section,
                    s.title,
                    s.text,
                    s.trm_version,
                    s.effective_date,
                )
                for s in sections_sorted
            ],
        )

        chunks_upserted = upsert_chunks(conn, chunks)

        if removed_ids:
            for stale_id in removed_ids:
                conn.execute(
                    "DELETE FROM chunks_vec WHERE rowid IN ("
                    "SELECT rowid FROM chunks "
                    "WHERE doc_type = 'tournament_rule' AND doc_id = ?"
                    ")",
                    (stale_id,),
                )
                conn.execute(
                    "DELETE FROM chunks WHERE doc_type = 'tournament_rule' AND doc_id = ?",
                    (stale_id,),
                )
                conn.execute("DELETE FROM trm_sections WHERE section_id = ?", (stale_id,))

        if document.trm_version:
            set_meta(conn, "trm_version", document.trm_version)
        if document.effective_date:
            set_meta(conn, "trm_effective_date", document.effective_date)
        now = utc_now_iso()
        set_meta(conn, "trm_last_ingested_at", now)
        set_meta(conn, "last_ingested_at", now)

    return TrmIngestSummary(
        sections_upserted=len(sections_sorted),
        sections_removed=len(removed_ids),
        chunks_upserted=chunks_upserted,
        trm_version=document.trm_version,
        effective_date=document.effective_date,
    )
