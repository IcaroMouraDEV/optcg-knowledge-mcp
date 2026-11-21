"""Deterministic chunk builders and upsert helpers for FTS5 + sqlite-vec indexing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping

from optcg_knowledge.db.models import (
    CardRecord,
    ChunkRecord,
    RuleRecord,
    RulingRecord,
    TrmSectionRecord,
)

MAX_ANCESTOR_CHARS = 240

_RULING_SOURCE_LABELS: dict[str, str] = {
    "rules": "RULES",
    "official_faq": "OFFICIAL FAQ",
    "errata": "ERRATA",
    "judge": "JUDGE RULING",
    "community": "COMMUNITY RULING",
}


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


def build_trm_chunk(
    section: TrmSectionRecord,
    sections_by_id: Mapping[str, TrmSectionRecord],
) -> ChunkRecord:
    """Build one chunk per TRM section, prepending the ancestor breadcrumb for context."""
    ancestors: list[TrmSectionRecord] = []
    seen: set[str] = {section.section_id}
    cursor = section.parent_id
    while cursor is not None and cursor not in seen:
        parent = sections_by_id.get(cursor)
        if parent is None:
            break
        ancestors.append(parent)
        seen.add(cursor)
        cursor = parent.parent_id

    ancestors.reverse()

    lines: list[str] = ["[TOURNAMENT RULES MANUAL]"]
    if ancestors:
        path = " > ".join(
            f"{anc.section_id} {_truncate(anc.title)}" for anc in ancestors
        )
        lines.append(path)
    lines.append(f"{section.section_id} {section.title}")
    if section.text:
        lines.append(section.text)
    text = "\n".join(lines)

    return ChunkRecord(
        chunk_id=f"trm:{section.section_id}",
        doc_type="tournament_rule",
        doc_id=section.section_id,
        text=text,
        authority=1,
        metadata={
            "section_id": section.section_id,
            "parent_id": section.parent_id,
            "section": section.section,
            "title": section.title,
            "source": "tournament_rules",
            "trm_version": section.trm_version,
            "effective_date": section.effective_date,
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
            "block_icon": card.block_icon,
        },
        content_hash=compute_content_hash(text),
    )


_JUDGE_TIER_LABELS: dict[str, str] = {
    "tier_0": "TIER 0",
    "tier_2": "TIER 2",
    "tier_1": "TIER 1",
}


def build_ruling_chunk(ruling: RulingRecord) -> ChunkRecord:
    """Build one chunk per ruling combining question, answer, and linked IDs."""
    source_label = _RULING_SOURCE_LABELS.get(ruling.source, ruling.source.upper())
    tier_label = _JUDGE_TIER_LABELS.get(ruling.judge_tier or "") if ruling.judge_tier else None
    tag = f"{source_label} | {tier_label}" if tier_label else source_label

    context_parts: list[str] = []
    if ruling.card_ids:
        context_parts.append(f"Cards: {', '.join(ruling.card_ids)}")
    if ruling.rule_ids:
        context_parts.append(f"Rules: {', '.join(ruling.rule_ids)}")

    header = (
        f"[{tag}] {' | '.join(context_parts)}"
        if context_parts
        else f"[{tag}]"
    )
    text = f"{header}\nQ: {ruling.question}\nA: {ruling.answer}"

    return ChunkRecord(
        chunk_id=f"ruling:{ruling.ruling_id}",
        doc_type="ruling",
        doc_id=ruling.ruling_id,
        text=text,
        authority=ruling.authority,
        metadata={
            "ruling_id": ruling.ruling_id,
            "source": ruling.source,
            "judge_tier": ruling.judge_tier,
            "date": ruling.date,
            "rules_version": ruling.rules_version,
            "url": ruling.url,
            "raw_ref": ruling.raw_ref,
            "card_ids": list(ruling.card_ids),
            "rule_ids": list(ruling.rule_ids),
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
