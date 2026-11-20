"""Comprehensive Rules PDF parser and idempotent SQLite ingestor."""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from optcg_knowledge.db.models import RuleRecord
from optcg_knowledge.db.schema import set_meta, utc_now_iso
from optcg_knowledge.ingest.chunking import build_rule_chunk, upsert_chunks
from optcg_knowledge.ingest.dates import normalize_official_date

logger = logging.getLogger(__name__)

_RULE_LINE_RE = re.compile(r"^(\d+(?:-\d+)*)\.\s+(.*)$")
_TOC_LINE_RE = re.compile(r"^\d+(?:-\d+)*\.\s+.*(?:\.{2,}|\s{3,})\d+\s*$")
_VERSION_RE = re.compile(r"\b(?:Ver\.?|Version)\s*[:\-]?\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_EFFECTIVE_DATE_RE = re.compile(
    r"(?:Effective(?:\s+Date)?|Last\s+Updated|Updated)\s*[:\-]?\s*"
    r"([A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)
_NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^-\s*\d{1,3}\s*-$"),
    re.compile(r"^Page\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^ONE\s+PIECE\s+CARD\s+GAME\s+Comprehensive\s+Rules.*$", re.IGNORECASE),
    re.compile(
        r"^Comprehensive\s+Rules(?:\s+(?:Ver\.?|Version)\s*[:\-]?\s*\d+(?:\.\d+)*)?$",
        re.IGNORECASE,
    ),
    re.compile(r"^©.*(?:Eiichiro\s+Oda|Shueisha|Bandai).*$", re.IGNORECASE),
)
_PRESERVE_BREAK_PREFIXES = ("Example:", "Note:", "※", "•", "-", "*")


@dataclass(frozen=True, slots=True)
class ParsedRulesDocument:
    """Output of parsing a Comprehensive Rules PDF or extracted text."""

    rules: tuple[RuleRecord, ...]
    rules_version: str | None
    effective_date: str | None


@dataclass(frozen=True, slots=True)
class RulesIngestSummary:
    """Counters returned after ingesting Comprehensive Rules."""

    rules_upserted: int
    rules_removed: int
    chunks_upserted: int
    rules_version: str | None
    effective_date: str | None


def extract_pdf_text(pdf_path: Path | str) -> str:
    """Extract plain text from every page of a PDF using PyMuPDF."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Rules PDF not found: {path}")

    pages: list[str] = []
    with pymupdf.open(path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def _is_noise_line(line: str) -> bool:
    return any(pattern.match(line) for pattern in _NOISE_LINE_PATTERNS)


def _extract_metadata(
    lines: list[str],
    *,
    rules_version: str | None,
    effective_date: str | None,
) -> tuple[str | None, str | None]:
    header_sample = "\n".join(lines[:80])
    detected_version = rules_version.strip() if rules_version and rules_version.strip() else None
    if detected_version is None:
        match = _VERSION_RE.search(header_sample)
        if match:
            detected_version = match.group(1)

    detected_date = normalize_official_date(effective_date)
    if detected_date is None:
        match = _EFFECTIVE_DATE_RE.search(header_sample)
        if match:
            detected_date = normalize_official_date(match.group(1))

    return detected_version, detected_date


def _append_continuation(existing: str, line: str) -> str:
    if not existing:
        return line
    if line.startswith(_PRESERVE_BREAK_PREFIXES):
        return f"{existing}\n{line}"
    if existing.endswith("-") and not existing.endswith(" -"):
        return f"{existing[:-1]}{line}"
    return f"{existing} {line}"


def _rule_sort_key(rule_id: str) -> tuple[int, ...]:
    return tuple(int(part) for part in rule_id.split("-"))


def _resolve_parent_id(rule_id: str, known_ids: set[str]) -> str | None:
    parts = rule_id.split("-")
    while len(parts) > 1:
        parts.pop()
        candidate = "-".join(parts)
        if candidate in known_ids:
            return candidate
    return None


def parse_rules_text(
    raw_text: str,
    *,
    rules_version: str | None = None,
    effective_date: str | None = None,
) -> ParsedRulesDocument:
    """Parse extracted Comprehensive Rules text into hierarchical RuleRecord entries."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    raw_lines = [line.strip() for line in normalized.split("\n")]

    detected_version, detected_date = _extract_metadata(
        raw_lines,
        rules_version=rules_version,
        effective_date=effective_date,
    )

    # First pass: strip empty/noise/TOC lines and parse rule blocks.
    entries: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    toc_seen_ids: set[str] = set()
    in_toc = False

    for line in raw_lines:
        if not line:
            continue
        if _is_noise_line(line):
            continue
        if line.lower() in {"contents", "table of contents"}:
            in_toc = True
            continue
        if _TOC_LINE_RE.match(line):
            in_toc = True
            toc_match = _RULE_LINE_RE.match(line)
            if toc_match:
                toc_seen_ids.add(toc_match.group(1))
            continue

        match = _RULE_LINE_RE.match(line)
        if match:
            rule_id, body = match.group(1), match.group(2).strip()
            is_top_level = "-" not in rule_id
            top_num = int(rule_id.split("-", 1)[0])

            # Exit TOC when we hit a rule ID already listed in the TOC or a deep sub-rule.
            if in_toc:
                if rule_id in toc_seen_ids or rule_id.count("-") >= 2:
                    in_toc = False
                else:
                    toc_seen_ids.add(rule_id)
                    continue

            # Ignore implausible top-level numbers (e.g., years) or repeated inline numbered lists.
            if is_top_level and (top_num == 0 or top_num > 50 or rule_id in seen_ids):
                if entries:
                    prev_id, prev_text = entries[-1]
                    entries[-1] = (prev_id, _append_continuation(prev_text, line))
                continue

            if rule_id in seen_ids:
                # Duplicate rule IDs shouldn't overwrite earlier body entries; append as note.
                logger.warning("Duplicate rule_id %s encountered; merging text", rule_id)
                for idx, (existing_id, existing_text) in enumerate(entries):
                    if existing_id == rule_id:
                        entries[idx] = (existing_id, _append_continuation(existing_text, body))
                        break
                continue

            seen_ids.add(rule_id)
            entries.append((rule_id, body))
        else:
            if in_toc or not entries:
                continue
            prev_id, prev_text = entries[-1]
            entries[-1] = (prev_id, _append_continuation(prev_text, line))

    # Build top-level section titles and hierarchical RuleRecords.
    section_titles: dict[str, str] = {}
    for rule_id, text in entries:
        if "-" not in rule_id:
            first_line = text.split("\n", 1)[0].strip()
            section_titles[rule_id] = f"{rule_id}. {first_line}"

    known_ids = {rule_id for rule_id, _ in entries}
    ordered_entries = sorted(entries, key=lambda item: _rule_sort_key(item[0]))

    records: list[RuleRecord] = []
    for rule_id, text in ordered_entries:
        top_section_key = rule_id.split("-", 1)[0]
        section = section_titles.get(top_section_key, f"Section {top_section_key}")
        parent_id = _resolve_parent_id(rule_id, known_ids)
        records.append(
            RuleRecord(
                rule_id=rule_id,
                parent_id=parent_id,
                section=section,
                text=text.strip(),
                rules_version=detected_version,
                effective_date=detected_date,
            )
        )

    return ParsedRulesDocument(
        rules=tuple(records),
        rules_version=detected_version,
        effective_date=detected_date,
    )


def parse_rules_pdf(
    pdf_path: Path | str,
    *,
    rules_version: str | None = None,
    effective_date: str | None = None,
) -> ParsedRulesDocument:
    """Extract and parse an official Comprehensive Rules PDF."""
    raw_text = extract_pdf_text(pdf_path)
    return parse_rules_text(
        raw_text,
        rules_version=rules_version,
        effective_date=effective_date,
    )


def ingest_rules(
    conn: sqlite3.Connection,
    document: ParsedRulesDocument,
) -> RulesIngestSummary:
    """Idempotently ingest a parsed Comprehensive Rules document into SQLite."""
    if not document.rules:
        raise ValueError("No rules found in document; refusing to overwrite existing rules")

    # Sort by hierarchy depth first so parent_id foreign keys always exist on insert.
    rules_sorted = sorted(
        document.rules,
        key=lambda r: (r.rule_id.count("-"), _rule_sort_key(r.rule_id)),
    )
    rules_by_id = {rule.rule_id: rule for rule in rules_sorted}
    chunks = [build_rule_chunk(rule, rules_by_id) for rule in rules_sorted]

    new_rule_ids = set(rules_by_id)

    with conn:
        existing_ids = {
            row[0] for row in conn.execute("SELECT rule_id FROM rules").fetchall()
        }
        removed_ids = sorted(existing_ids - new_rule_ids, key=lambda r: -r.count("-"))

        conn.executemany(
            """
            INSERT INTO rules (
                rule_id, parent_id, section, text, rules_version, effective_date
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(rule_id) DO UPDATE SET
                parent_id = excluded.parent_id,
                section = excluded.section,
                text = excluded.text,
                rules_version = excluded.rules_version,
                effective_date = excluded.effective_date
            """,
            [
                (
                    r.rule_id,
                    r.parent_id,
                    r.section,
                    r.text,
                    r.rules_version,
                    r.effective_date,
                )
                for r in rules_sorted
            ],
        )

        chunks_upserted = upsert_chunks(conn, chunks)

        if removed_ids:
            for stale_id in removed_ids:
                conn.execute(
                    "DELETE FROM chunks_vec WHERE rowid IN ("
                    "SELECT rowid FROM chunks WHERE doc_type = 'rule' AND doc_id = ?"
                    ")",
                    (stale_id,),
                )
                conn.execute(
                    "DELETE FROM chunks WHERE doc_type = 'rule' AND doc_id = ?",
                    (stale_id,),
                )
                conn.execute("DELETE FROM rules WHERE rule_id = ?", (stale_id,))

        if document.rules_version:
            set_meta(conn, "rules_version", document.rules_version)
        if document.effective_date:
            set_meta(conn, "rules_effective_date", document.effective_date)
        set_meta(conn, "rules_last_ingested_at", utc_now_iso())
        set_meta(conn, "last_ingested_at", utc_now_iso())

    return RulesIngestSummary(
        rules_upserted=len(rules_sorted),
        rules_removed=len(removed_ids),
        chunks_upserted=chunks_upserted,
        rules_version=document.rules_version,
        effective_date=document.effective_date,
    )
