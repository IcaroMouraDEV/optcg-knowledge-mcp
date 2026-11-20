"""Ruling curation and interoperable JSON export services for optcg-knowledge."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from optcg_knowledge.db.models import (
    JudgeTier,
    RulingRecord,
    SourceType,
    authority_for_source,
    normalize_judge_tier,
)
from optcg_knowledge.db.schema import get_meta, utc_now_iso
from optcg_knowledge.ingest.chunking import build_ruling_chunk, upsert_chunks

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = "1.0"
MAX_LIST_LIMIT = 100

CURATABLE_SOURCES: frozenset[str] = frozenset({"official_faq", "errata", "judge", "community"})
_SOURCE_PREFIX: dict[str, str] = {
    "official_faq": "faq",
    "errata": "errata",
    "judge": "judge",
    "community": "community",
}

_CARD_ID_RE = re.compile(r"^(?:(?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})$", re.IGNORECASE)
_RULE_ID_RE = re.compile(r"^\d+(?:-\d+){1,4}$")

# Privacy guards: block raw Discord user mentions, @handles, and legacy name#1234 tags.
_DISCORD_MENTION_RE = re.compile(r"<@!?\d{15,22}>")
_AT_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.-]{2,32}\b")
_DISCORD_DISCRIMINATOR_RE = re.compile(r"\b[A-Za-z0-9_.-]{2,32}#\d{4}\b")

_JUDGE_TIER_ORDER_SQL = """
    CASE r.judge_tier
        WHEN 'tier_0' THEN 1
        WHEN 'tier_2' THEN 2
        WHEN 'tier_1' THEN 3
        ELSE 4
    END
"""


def _rule_sort_key(rule_id: str) -> tuple[int, ...]:
    parts = rule_id.split("-")
    if all(part.isdigit() for part in parts):
        return tuple(int(part) for part in parts)
    return (999999,)


def _normalize_date(raw_date: str | None, *, field_name: str = "date") -> str | None:
    if raw_date is None:
        return None
    cleaned = " ".join(raw_date.strip().split())
    if not cleaned:
        return None

    ymd_match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", cleaned)
    if ymd_match:
        year, month, day = (int(part) for part in ymd_match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name} {raw_date!r}: {exc}") from exc

    for fmt in ("%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    raise ValueError(
        f"Invalid {field_name} {raw_date!r}; expected ISO 'YYYY-MM-DD' (e.g. '2026-11-18')"
    )


def _check_no_pii(text: str | None, *, field_name: str, check_discriminator: bool = True) -> None:
    if not text:
        return
    if _DISCORD_MENTION_RE.search(text):
        raise ValueError(
            f"{field_name} contains a raw Discord user mention (<@...>); "
            "strip all personal identifiers before calling add_ruling."
        )
    if _AT_HANDLE_RE.search(text):
        raise ValueError(
            f"{field_name} contains an @handle; "
            "strip all personal names and handles before calling add_ruling."
        )
    if check_discriminator and _DISCORD_DISCRIMINATOR_RE.search(text):
        raise ValueError(
            f"{field_name} contains a Discord username#0000 tag; "
            "strip all personal identifiers before calling add_ruling."
        )


def _normalize_card_ids(card_ids: Sequence[str] | None) -> tuple[str, ...]:
    if not card_ids:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for raw in card_ids:
        cleaned = raw.strip().upper()
        if not cleaned:
            continue
        if not _CARD_ID_RE.match(cleaned):
            raise ValueError(
                f"Invalid canonical card_id {raw!r}; expected format like 'OP05-119' or 'P-001'"
            )
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return tuple(sorted(result))


def _normalize_rule_refs(rule_refs: Sequence[str] | None) -> tuple[str, ...]:
    if not rule_refs:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for raw in rule_refs:
        cleaned = raw.strip()
        if not cleaned:
            continue
        if not _RULE_ID_RE.match(cleaned):
            raise ValueError(
                f"Invalid canonical rule_id {raw!r}; expected format like '7-1-4' or '7-1-4-1'"
            )
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    result.sort(key=_rule_sort_key)
    return tuple(result)


def make_curated_ruling_id(
    *,
    source: SourceType,
    card_ids: Sequence[str],
    question: str,
    date: str | None,
) -> str:
    """Compute a deterministic ruling_id for curated rulings."""
    prefix = _SOURCE_PREFIX.get(source, source)
    scope = card_ids[0].upper() if card_ids else "general"
    norm_question = " ".join(question.lower().split())
    payload = f"{source}|{scope}|{norm_question}|{date or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{scope}:{digest}"


def _hydrate_rulings(
    conn: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
    *,
    current_rules_version: str | None,
) -> list[dict[str, Any]]:
    ruling_ids = [str(row["ruling_id"]) for row in rows]
    cards_by_ruling: dict[str, list[str]] = {rid: [] for rid in ruling_ids}
    rules_by_ruling: dict[str, list[str]] = {rid: [] for rid in ruling_ids}

    if ruling_ids:
        placeholders = ", ".join("?" for _ in ruling_ids)
        for link in conn.execute(
            f"""
            SELECT ruling_id, card_id
            FROM ruling_cards
            WHERE ruling_id IN ({placeholders})
            ORDER BY card_id ASC
            """,
            ruling_ids,
        ).fetchall():
            cards_by_ruling[str(link["ruling_id"])].append(str(link["card_id"]))

        for link in conn.execute(
            f"""
            SELECT ruling_id, rule_id
            FROM ruling_rules
            WHERE ruling_id IN ({placeholders})
            """,
            ruling_ids,
        ).fetchall():
            rules_by_ruling[str(link["ruling_id"])].append(str(link["rule_id"]))

        for rid in rules_by_ruling:
            rules_by_ruling[rid].sort(key=_rule_sort_key)

    hydrated: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row["ruling_id"])
        ruling_version = str(row["rules_version"]) if row["rules_version"] is not None else None
        potentially_outdated = bool(
            ruling_version and current_rules_version and ruling_version != current_rules_version
        )
        hydrated.append(
            {
                "ruling_id": rid,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "source": str(row["source"]),
                "authority": int(row["authority"]),
                "judge_tier": (
                    str(row["judge_tier"]) if row["judge_tier"] is not None else None
                ),
                "date": str(row["date"]) if row["date"] is not None else None,
                "rules_version": ruling_version,
                "url": str(row["url"]) if row["url"] is not None else None,
                "raw_ref": str(row["raw_ref"]) if row["raw_ref"] is not None else None,
                "superseded_by": (
                    str(row["superseded_by"]) if row["superseded_by"] is not None else None
                ),
                "reviewed": bool(row["reviewed"]),
                "potentially_outdated": potentially_outdated,
                "card_ids": cards_by_ruling[rid],
                "rule_ids": rules_by_ruling[rid],
            }
        )
    return hydrated


def _fetch_ruling_by_id(conn: sqlite3.Connection, ruling_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            ruling_id, question, answer, source, authority, judge_tier,
            date, rules_version, url, raw_ref, superseded_by, reviewed
        FROM rulings
        WHERE ruling_id = ?
        """,
        (ruling_id,),
    ).fetchone()
    if row is None:
        return None
    current_rules_version = get_meta(conn, "rules_version")
    return _hydrate_rulings(conn, [row], current_rules_version=current_rules_version)[0]


def add_ruling(
    conn: sqlite3.Connection,
    *,
    question: str,
    answer: str,
    source: str,
    date: str | None = None,
    card_ids: Sequence[str] | None = None,
    rule_refs: Sequence[str] | None = None,
    url: str | None = None,
    raw_ref: str | None = None,
    rules_version: str | None = None,
    judge_tier: str | None = None,
) -> dict[str, Any]:
    """Insert or update a curated ruling with ``reviewed = False`` and index its chunk.

    Enforces authority mapping, privacy guards (no Discord mentions/handles/tags),
    optional ``judge_tier`` validation (``tier_0 | tier_2 | tier_1`` for ``source='judge'``),
    and immediate FTS5 indexing via ``build_ruling_chunk`` + ``upsert_chunks``.
    """
    cleaned_question = question.strip()
    cleaned_answer = answer.strip()
    if not cleaned_question:
        raise ValueError("question must be a non-empty string")
    if not cleaned_answer:
        raise ValueError("answer must be a non-empty string")

    cleaned_source = source.strip().lower()
    if cleaned_source not in CURATABLE_SOURCES:
        valid = ", ".join(sorted(CURATABLE_SOURCES))
        raise ValueError(
            f"Unsupported curation source {source!r}; expected one of: {valid}"
        )
    typed_source: SourceType = cleaned_source  # type: ignore[assignment]
    authority = authority_for_source(typed_source)

    normalized_tier: JudgeTier | None = normalize_judge_tier(judge_tier)
    if normalized_tier is not None and typed_source != "judge":
        raise ValueError("judge_tier can only be set when source == 'judge'")

    _check_no_pii(cleaned_question, field_name="question")
    _check_no_pii(cleaned_answer, field_name="answer")
    cleaned_raw_ref = raw_ref.strip() if raw_ref and raw_ref.strip() else None
    _check_no_pii(cleaned_raw_ref, field_name="raw_ref", check_discriminator=False)
    cleaned_url = url.strip() if url and url.strip() else None

    normalized_date = _normalize_date(date, field_name="date")
    normalized_cards = _normalize_card_ids(card_ids)
    normalized_rules = _normalize_rule_refs(rule_refs)

    resolved_rules_version = (
        rules_version.strip()
        if rules_version and rules_version.strip()
        else get_meta(conn, "rules_version")
    )

    warnings: list[str] = []
    if normalized_cards:
        has_cards = conn.execute("SELECT 1 FROM cards LIMIT 1").fetchone() is not None
        if has_cards:
            placeholders = ", ".join("?" for _ in normalized_cards)
            existing_cards = {
                str(row[0]).upper()
                for row in conn.execute(
                    f"SELECT card_id FROM cards WHERE UPPER(card_id) IN ({placeholders})",
                    normalized_cards,
                ).fetchall()
            }
            missing_cards = [cid for cid in normalized_cards if cid not in existing_cards]
            if missing_cards:
                warnings.append(
                    f"Card ID(s) not currently in cards table: {', '.join(missing_cards)}"
                )

    if normalized_rules:
        has_rules = conn.execute("SELECT 1 FROM rules LIMIT 1").fetchone() is not None
        if has_rules:
            placeholders = ", ".join("?" for _ in normalized_rules)
            existing_rules = {
                str(row[0])
                for row in conn.execute(
                    f"SELECT rule_id FROM rules WHERE rule_id IN ({placeholders})",
                    normalized_rules,
                ).fetchall()
            }
            missing_rules = [rid for rid in normalized_rules if rid not in existing_rules]
            if missing_rules:
                warnings.append(
                    f"Rule ID(s) not currently in rules table: {', '.join(missing_rules)}"
                )

    ruling_id = make_curated_ruling_id(
        source=typed_source,
        card_ids=normalized_cards,
        question=cleaned_question,
        date=normalized_date,
    )

    ruling_record = RulingRecord(
        ruling_id=ruling_id,
        question=cleaned_question,
        answer=cleaned_answer,
        source=typed_source,
        authority=authority,
        date=normalized_date,
        rules_version=resolved_rules_version,
        url=cleaned_url,
        superseded_by=None,
        reviewed=False,
        raw_ref=cleaned_raw_ref,
        judge_tier=normalized_tier,
        card_ids=normalized_cards,
        rule_ids=normalized_rules,
    )

    with conn:
        existing = conn.execute(
            "SELECT 1 FROM rulings WHERE ruling_id = ?",
            (ruling_id,),
        ).fetchone()
        created = existing is None

        conn.execute(
            """
            INSERT INTO rulings (
                ruling_id, question, answer, source, authority, judge_tier,
                date, rules_version, url, superseded_by, reviewed, raw_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, ?)
            ON CONFLICT(ruling_id) DO UPDATE SET
                question = excluded.question,
                answer = excluded.answer,
                source = excluded.source,
                authority = excluded.authority,
                judge_tier = COALESCE(excluded.judge_tier, rulings.judge_tier),
                date = COALESCE(excluded.date, rulings.date),
                rules_version = COALESCE(excluded.rules_version, rulings.rules_version),
                url = COALESCE(excluded.url, rulings.url),
                superseded_by = rulings.superseded_by,
                reviewed = 0,
                raw_ref = COALESCE(excluded.raw_ref, rulings.raw_ref)
            """,
            (
                ruling_record.ruling_id,
                ruling_record.question,
                ruling_record.answer,
                ruling_record.source,
                ruling_record.authority,
                ruling_record.judge_tier,
                ruling_record.date,
                ruling_record.rules_version,
                ruling_record.url,
                ruling_record.raw_ref,
            ),
        )

        conn.execute("DELETE FROM ruling_cards WHERE ruling_id = ?", (ruling_id,))
        if normalized_cards:
            conn.executemany(
                "INSERT OR IGNORE INTO ruling_cards (ruling_id, card_id) VALUES (?, ?)",
                [(ruling_id, cid) for cid in normalized_cards],
            )

        conn.execute("DELETE FROM ruling_rules WHERE ruling_id = ?", (ruling_id,))
        if normalized_rules:
            conn.executemany(
                "INSERT OR IGNORE INTO ruling_rules (ruling_id, rule_id) VALUES (?, ?)",
                [(ruling_id, rid) for rid in normalized_rules],
            )

        upsert_chunks(conn, [build_ruling_chunk(ruling_record)])

    stored = _fetch_ruling_by_id(conn, ruling_id)
    assert stored is not None
    logger.info(
        "Curated ruling %s (created=%s, source=%s, authority=%d, judge_tier=%s)",
        ruling_id,
        created,
        typed_source,
        authority,
        normalized_tier,
    )
    return {
        "ruling_id": ruling_id,
        "created": created,
        "ruling": stored,
        "warnings": warnings,
    }


def _has_superseding_cycle(
    conn: sqlite3.Connection,
    *,
    ruling_id: str,
    target_id: str,
) -> bool:
    """Return True if setting ``ruling_id.superseded_by = target_id`` creates a cycle."""
    cursor: str | None = target_id
    visited: set[str] = {ruling_id}
    while cursor is not None:
        if cursor in visited:
            return True
        visited.add(cursor)
        row = conn.execute(
            "SELECT superseded_by FROM rulings WHERE ruling_id = ?",
            (cursor,),
        ).fetchone()
        if row is None:
            break
        cursor = str(row["superseded_by"]) if row["superseded_by"] is not None else None
    return False


def mark_ruling(
    conn: sqlite3.Connection,
    ruling_id: str,
    *,
    reviewed: bool | None = None,
    superseded_by: str | None = None,
    judge_tier: str | None = None,
) -> dict[str, Any]:
    """Update review state, ``superseded_by`` link, and/or ``judge_tier`` for a ruling.

    Never deletes outdated rulings; rejects self-superseding, missing IDs, and cycles.
    """
    cleaned_id = ruling_id.strip()
    if not cleaned_id:
        raise ValueError("ruling_id must be a non-empty string")
    if reviewed is None and superseded_by is None and judge_tier is None:
        raise ValueError(
            "At least one of reviewed, superseded_by, or judge_tier must be provided"
        )

    existing = _fetch_ruling_by_id(conn, cleaned_id)
    if existing is None:
        raise ValueError(f"Ruling not found: {cleaned_id!r}")

    cleaned_superseded: str | None = None
    if superseded_by is not None:
        cleaned_superseded = superseded_by.strip()
        if not cleaned_superseded:
            raise ValueError("superseded_by must be a non-empty ruling_id when provided")
        if cleaned_superseded == cleaned_id:
            raise ValueError(f"Ruling {cleaned_id!r} cannot supersede itself")
        target = conn.execute(
            "SELECT 1 FROM rulings WHERE ruling_id = ?",
            (cleaned_superseded,),
        ).fetchone()
        if target is None:
            raise ValueError(f"Target superseded_by ruling not found: {cleaned_superseded!r}")
        if _has_superseding_cycle(conn, ruling_id=cleaned_id, target_id=cleaned_superseded):
            raise ValueError(
                f"Setting superseded_by={cleaned_superseded!r} on {cleaned_id!r} "
                "would create a circular superseding chain"
            )

    normalized_tier: JudgeTier | None = None
    if judge_tier is not None:
        normalized_tier = normalize_judge_tier(judge_tier)
        if normalized_tier is not None and existing["source"] != "judge":
            raise ValueError(
                f"Cannot set judge_tier on ruling {cleaned_id!r} with source={existing['source']!r}"
            )

    updates: list[str] = []
    params: list[Any] = []
    if reviewed is not None:
        updates.append("reviewed = ?")
        params.append(1 if reviewed else 0)
    if cleaned_superseded is not None:
        updates.append("superseded_by = ?")
        params.append(cleaned_superseded)
    if judge_tier is not None:
        updates.append("judge_tier = ?")
        params.append(normalized_tier)
    params.append(cleaned_id)

    with conn:
        conn.execute(
            f"UPDATE rulings SET {', '.join(updates)} WHERE ruling_id = ?",
            params,
        )
        if judge_tier is not None:
            refreshed = _fetch_ruling_by_id(conn, cleaned_id)
            assert refreshed is not None
            updated_record = RulingRecord(
                ruling_id=refreshed["ruling_id"],
                question=refreshed["question"],
                answer=refreshed["answer"],
                source=refreshed["source"],
                authority=refreshed["authority"],
                date=refreshed["date"],
                rules_version=refreshed["rules_version"],
                url=refreshed["url"],
                superseded_by=refreshed["superseded_by"],
                reviewed=refreshed["reviewed"],
                raw_ref=refreshed["raw_ref"],
                judge_tier=refreshed["judge_tier"],
                card_ids=tuple(refreshed["card_ids"]),
                rule_ids=tuple(refreshed["rule_ids"]),
            )
            upsert_chunks(conn, [build_ruling_chunk(updated_record)])

    final_ruling = _fetch_ruling_by_id(conn, cleaned_id)
    assert final_ruling is not None
    logger.info(
        "Marked ruling %s (reviewed=%s, superseded_by=%s, judge_tier=%s)",
        cleaned_id,
        final_ruling["reviewed"],
        final_ruling["superseded_by"],
        final_ruling["judge_tier"],
    )
    return {
        "ruling_id": cleaned_id,
        "updated": True,
        "ruling": final_ruling,
    }


def list_unreviewed_rulings(
    conn: sqlite3.Connection,
    *,
    limit: int = 20,
) -> dict[str, Any]:
    """List pending unreviewed rulings ordered by authority, judge tier, and date."""
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    effective_limit = min(limit, MAX_LIST_LIMIT)
    current_rules_version = get_meta(conn, "rules_version")

    total_row = conn.execute("SELECT COUNT(*) FROM rulings WHERE reviewed = 0").fetchone()
    total_unreviewed = int(total_row[0]) if total_row is not None else 0

    rows = conn.execute(
        f"""
        SELECT
            r.ruling_id,
            r.question,
            r.answer,
            r.source,
            r.authority,
            r.judge_tier,
            r.date,
            r.rules_version,
            r.url,
            r.raw_ref,
            r.superseded_by,
            r.reviewed
        FROM rulings r
        WHERE r.reviewed = 0
        ORDER BY
            r.authority ASC,
            {_JUDGE_TIER_ORDER_SQL} ASC,
            COALESCE(r.date, '') DESC,
            r.ruling_id ASC
        LIMIT ?
        """,
        (effective_limit,),
    ).fetchall()

    rulings = _hydrate_rulings(conn, rows, current_rules_version=current_rules_version)
    return {
        "total_unreviewed": total_unreviewed,
        "returned": len(rulings),
        "current_rules_version": current_rules_version,
        "rulings": rulings,
    }


def export_rulings(
    conn: sqlite3.Connection,
    *,
    card_id: str | None = None,
    since: str | None = None,
    min_authority: int | None = None,
    include_superseded: bool = False,
    include_unreviewed: bool = False,
) -> dict[str, Any]:
    """Export rulings in a stable, versioned JSON envelope shared by MCP and CLI.

    By default exports only reviewed, non-superseded rulings so downstream consumers
    (such as an external ``optcg-engine`` test generator) receive vetted entries.
    """
    normalized_card_id: str | None = None
    if card_id is not None and card_id.strip():
        normalized_card_id = card_id.strip().upper()

    normalized_since = _normalize_date(since, field_name="since")

    if min_authority is not None and (
        not isinstance(min_authority, int) or min_authority not in (1, 2, 3, 4)
    ):
        raise ValueError("min_authority must be an integer between 1 (highest) and 4 (lowest)")

    clauses: list[str] = []
    params: list[Any] = []
    join_sql = ""

    if normalized_card_id is not None:
        join_sql = "JOIN ruling_cards rc ON rc.ruling_id = r.ruling_id"
        clauses.append("UPPER(rc.card_id) = ?")
        params.append(normalized_card_id)

    if normalized_since is not None:
        clauses.append("r.date IS NOT NULL AND r.date >= ?")
        params.append(normalized_since)

    if min_authority is not None:
        clauses.append("r.authority <= ?")
        params.append(min_authority)

    if not include_superseded:
        clauses.append("r.superseded_by IS NULL")

    if not include_unreviewed:
        clauses.append("r.reviewed = 1")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    current_rules_version = get_meta(conn, "rules_version")

    rows = conn.execute(
        f"""
        SELECT DISTINCT
            r.ruling_id,
            r.question,
            r.answer,
            r.source,
            r.authority,
            r.judge_tier,
            r.date,
            r.rules_version,
            r.url,
            r.raw_ref,
            r.superseded_by,
            r.reviewed
        FROM rulings r
        {join_sql}
        {where_sql}
        ORDER BY
            r.authority ASC,
            {_JUDGE_TIER_ORDER_SQL} ASC,
            COALESCE(r.date, '') DESC,
            r.ruling_id ASC
        """,
        params,
    ).fetchall()

    rulings = _hydrate_rulings(conn, rows, current_rules_version=current_rules_version)
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": utc_now_iso(),
        "rules_version": current_rules_version,
        "filters": {
            "card_id": normalized_card_id,
            "since": normalized_since,
            "min_authority": min_authority,
            "include_superseded": include_superseded,
            "include_unreviewed": include_unreviewed,
        },
        "total": len(rulings),
        "rulings": rulings,
    }
