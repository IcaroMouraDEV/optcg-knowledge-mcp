"""Deterministic chunk builders and upsert helpers for FTS5 + sqlite-vec indexing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping

from optcg_knowledge.db.models import CardRecord, ChunkRecord, RuleRecord

MAX_ANCESTOR_CHARS = 240


def compute_content_hash(text: str) -> str:
    """Compute a stable SHA-256 hex digest for chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int = MAX_ANCESTOR_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 1].rstrip()}…"


def build_rule_chunk(
    rule: RuleRecord,
    rules_by_id: Mapping[str, RuleRecord],
) -> ChunkRecord:
    """Build one chunk per rule, prepending the full ancestor path for context."""
    ancestors: list[RuleRecord] = []
    seen: set[str] = {rule.rule_id}
    cursor = rule.parent_id
    while cursor is not None and cursor not in seen:
        parent = rules_by_id.get(cursor)
        if parent is None:
            break
        ancestors.append(parent)
        seen.add(cursor)
        cursor = parent.parent_id

    ancestors.reverse()

    lines: list[str] = []
    if ancestors:
        path = " > ".join(f"{anc.rule_id}. {_truncate(anc.text)}" for anc in ancestors)
        lines.append(path)
    lines.append(f"{rule.rule_id}. {rule.text}")
    text = "\n".join(lines)

    return ChunkRecord(
        chunk_id=f"rule:{rule.rule_id}",
        doc_type="rule",
        doc_id=rule.rule_id,
        text=text,
        authority=1,
        metadata={
            "rule_id": rule.rule_id,
            "parent_id": rule.parent_id,
            "section": rule.section,
            "rules_version": rule.rules_version,
            "effective_date": rule.effective_date,
        },
        content_hash=compute_content_hash(text),
    )


def build_card_chunk(card: CardRecord) -> ChunkRecord:
    """Build one chunk per card combining identity, stats, traits, effect, and trigger."""
    lines = [f"{card.card_id} — {card.name}"]

    header_parts = [f"Category: {card.category}", f"Set: {card.set_code}"]
    if card.colors:
        header_parts.append(f"Colors: {'/'.join(card.colors)}")
    if card.rarity:
        header_parts.append(f"Rarity: {card.rarity}")
    lines.append(" | ".join(header_parts))

    stat_parts: list[str] = []
    if card.life is not None:
        stat_parts.append(f"Life: {card.life}")
    if card.cost is not None:
        stat_parts.append(f"Cost: {card.cost}")
    if card.power is not None:
        stat_parts.append(f"Power: {card.power}")
    if card.counter is not None:
        stat_parts.append(f"Counter: +{card.counter}")
    if card.attributes:
        stat_parts.append(f"Attributes: {'/'.join(card.attributes)}")
    if stat_parts:
        lines.append(" | ".join(stat_parts))

    if card.traits:
        lines.append(f"Traits: {' / '.join(card.traits)}")
    if card.effect_text:
        lines.append(f"Effect: {card.effect_text}")
    if card.trigger_text:
        lines.append(f"Trigger: {card.trigger_text}")

    text = "\n".join(lines)
    return ChunkRecord(
        chunk_id=f"card:{card.card_id}",
        doc_type="card",
        doc_id=card.card_id,
        text=text,
        authority=1,
        metadata={
            "card_id": card.card_id,
            "name": card.name,
            "set_code": card.set_code,
            "category": card.category,
            "colors": list(card.colors),
            "traits": list(card.traits),
            "rarity": card.rarity,
        },
        content_hash=compute_content_hash(text),
    )


def upsert_chunks(conn: sqlite3.Connection, chunks: Iterable[ChunkRecord]) -> int:
    """Idempotently upsert chunks while preserving existing rowids and valid embeddings."""
    rows = [
        (
            chunk.chunk_id,
            chunk.doc_type,
            chunk.doc_id,
            chunk.text,
            chunk.authority,
            json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
            chunk.content_hash,
        )
        for chunk in chunks
    ]
    if not rows:
        return 0

    conn.executemany(
        """
        INSERT INTO chunks (
            chunk_id, doc_type, doc_id, text, authority, metadata, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            doc_type = excluded.doc_type,
            doc_id = excluded.doc_id,
            text = excluded.text,
            authority = excluded.authority,
            metadata = excluded.metadata,
            content_hash = excluded.content_hash,
            embedded_hash = CASE
                WHEN chunks.content_hash = excluded.content_hash THEN chunks.embedded_hash
                ELSE NULL
            END,
            embedded_model = CASE
                WHEN chunks.content_hash = excluded.content_hash THEN chunks.embedded_model
                ELSE NULL
            END
        """,
        rows,
    )
    return len(rows)
