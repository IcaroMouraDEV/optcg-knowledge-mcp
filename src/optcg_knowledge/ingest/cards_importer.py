"""Card JSON normalization, parallel deduplication, and idempotent SQLite ingestion."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from optcg_knowledge.db.models import CardRecord
from optcg_knowledge.db.schema import set_meta, utc_now_iso
from optcg_knowledge.ingest.chunking import build_card_chunk, upsert_chunks
from optcg_knowledge.ingest.sources.base import CardSource

logger = logging.getLogger(__name__)

_CARD_ID_RE = re.compile(
    r"^(?P<base>(?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})(?P<suffix>_[A-Za-z0-9]+)?$",
    re.IGNORECASE,
)
_SET_PREFIX_RE = re.compile(r"^([A-Z]+\d{2}|P)-\d{3}$")

_CATEGORY_CANONICAL: dict[str, str] = {
    "leader": "Leader",
    "character": "Character",
    "event": "Event",
    "stage": "Stage",
    "don": "DON!!",
    "don!!": "DON!!",
}


@dataclass(frozen=True, slots=True)
class CardsIngestSummary:
    """Counters returned after ingesting card records."""

    raw_records_seen: int
    cards_upserted: int
    variants_merged: int
    chunks_upserted: int
    source_name: str


def parse_card_id(raw_id: str) -> tuple[str, bool]:
    """Return (canonical_card_id, is_variant) for an OPTCG card identifier."""
    cleaned = raw_id.strip()
    match = _CARD_ID_RE.match(cleaned)
    if match:
        base = match.group("base").upper()
        is_variant = match.group("suffix") is not None
        return base, is_variant
    return cleaned.upper(), "_" in cleaned


def _derive_set_code(card_id: str, raw: dict[str, Any]) -> str:
    explicit = raw.get("set_code") or raw.get("set_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().upper().replace("-", "")

    match = _SET_PREFIX_RE.match(card_id)
    if match:
        return match.group(1)

    pack_id = raw.get("pack_id")
    if isinstance(pack_id, str) and pack_id.strip():
        return pack_id.strip().upper()
    return "UNKNOWN"


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "Unknown"
    cleaned = value.strip()
    return _CATEGORY_CANONICAL.get(cleaned.lower(), cleaned)


def _normalize_str_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[/,]", value)]
    elif isinstance(value, list):
        parts = [str(p).strip() for p in value]
    else:
        return ()
    cleaned = [p for p in parts if p and p != "-"]
    return tuple(dict.fromkeys(cleaned))


def _normalize_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lstrip("+")
        if not cleaned or cleaned == "-":
            return None
        try:
            return int(cleaned)
        except ValueError:
            return None
    return None


def _normalize_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned == "-":
        return None
    return cleaned


def _normalize_image_url(raw: dict[str, Any]) -> str | None:
    for key in ("img_full_url", "image_url", "img_url"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip().startswith(("http://", "https://")):
            return val.strip()
    return None


def _variant_metadata(raw_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw_id,
        "pack_id": raw.get("pack_id") or raw.get("set_code"),
        "rarity": _normalize_optional_text(raw.get("rarity")),
        "image_url": _normalize_image_url(raw),
    }


def _build_card_record(
    canonical_id: str,
    primary_raw: dict[str, Any],
    variants: list[dict[str, Any]],
) -> CardRecord:
    name = _normalize_optional_text(primary_raw.get("name")) or canonical_id
    category = _normalize_category(primary_raw.get("category") or primary_raw.get("card_type"))
    colors = _normalize_str_list(primary_raw.get("colors") or primary_raw.get("color"))
    attributes = _normalize_str_list(
        primary_raw.get("attributes") or primary_raw.get("attribute")
    )
    traits = _normalize_str_list(
        primary_raw.get("traits") or primary_raw.get("types") or primary_raw.get("sub_types")
    )

    cost = _normalize_optional_int(primary_raw.get("cost"))
    life = _normalize_optional_int(primary_raw.get("life"))
    # Vegapull/Bandai datasets often place a Leader's Life value in the cost field.
    if category == "Leader" and life is None and cost is not None:
        life = cost
        cost = None

    power = _normalize_optional_int(primary_raw.get("power"))
    counter = _normalize_optional_int(primary_raw.get("counter"))
    effect_text = _normalize_optional_text(
        primary_raw.get("effect_text")
        if "effect_text" in primary_raw
        else primary_raw.get("effect")
    )
    trigger_text = _normalize_optional_text(
        primary_raw.get("trigger_text")
        if "trigger_text" in primary_raw
        else primary_raw.get("trigger")
    )
    rarity = _normalize_optional_text(primary_raw.get("rarity"))
    image_url = _normalize_image_url(primary_raw)
    set_code = _derive_set_code(canonical_id, primary_raw)

    raw_payload = dict(primary_raw)
    if variants:
        raw_payload["variants"] = sorted(
            variants,
            key=lambda item: (str(item.get("id", "")), str(item.get("pack_id", ""))),
        )

    return CardRecord(
        card_id=canonical_id,
        name=name,
        set_code=set_code,
        category=category,
        colors=colors,
        cost=cost,
        life=life,
        power=power,
        counter=counter,
        attributes=attributes,
        traits=traits,
        effect_text=effect_text,
        trigger_text=trigger_text,
        rarity=rarity,
        image_url=image_url,
        raw_json=raw_payload,
    )


def normalize_cards(raw_records: Iterable[dict[str, Any]]) -> tuple[list[CardRecord], int]:
    """Normalize raw card dicts into deduplicated canonical CardRecords.

    Returns (cards, variants_merged_count).
    """
    grouped: dict[str, list[tuple[str, bool, dict[str, Any]]]] = {}
    for raw in raw_records:
        raw_id_val = raw.get("card_id") or raw.get("id") or raw.get("card_set_id")
        if not isinstance(raw_id_val, str) or not raw_id_val.strip():
            logger.warning("Skipping card record missing id/card_id: %r", raw.get("name"))
            continue
        canonical_id, is_variant = parse_card_id(raw_id_val)
        grouped.setdefault(canonical_id, []).append((raw_id_val.strip(), is_variant, raw))

    cards: list[CardRecord] = []
    variants_merged = 0

    for canonical_id in sorted(grouped):
        entries = grouped[canonical_id]
        # Prefer non-variant base printings first, then deterministic sort by raw_id.
        entries_sorted = sorted(entries, key=lambda item: (item[1], item[0]))
        _, _, primary_raw = entries_sorted[0]
        variants = [
            _variant_metadata(raw_id, raw_item)
            for raw_id, _, raw_item in entries_sorted[1:]
        ]
        variants_merged += len(variants)
        cards.append(_build_card_record(canonical_id, primary_raw, variants))

    return cards, variants_merged


def ingest_cards(conn: sqlite3.Connection, source: CardSource) -> CardsIngestSummary:
    """Fetch, normalize, and idempotently upsert cards and card chunks into SQLite."""
    raw_records = source.fetch_raw_cards()
    cards, variants_merged = normalize_cards(raw_records)
    if not cards:
        raise ValueError(f"No valid card records loaded from source {source.source_name}")

    chunks = [build_card_chunk(card) for card in cards]

    with conn:
        conn.executemany(
            """
            INSERT INTO cards (
                card_id, name, set_code, category, colors, cost, life, power, counter,
                attributes, traits, effect_text, trigger_text, rarity, image_url, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(card_id) DO UPDATE SET
                name = excluded.name,
                set_code = excluded.set_code,
                category = excluded.category,
                colors = excluded.colors,
                cost = excluded.cost,
                life = excluded.life,
                power = excluded.power,
                counter = excluded.counter,
                attributes = excluded.attributes,
                traits = excluded.traits,
                effect_text = excluded.effect_text,
                trigger_text = excluded.trigger_text,
                rarity = excluded.rarity,
                image_url = excluded.image_url,
                raw_json = excluded.raw_json
            """,
            [
                (
                    c.card_id,
                    c.name,
                    c.set_code,
                    c.category,
                    json.dumps(list(c.colors), ensure_ascii=False),
                    c.cost,
                    c.life,
                    c.power,
                    c.counter,
                    json.dumps(list(c.attributes), ensure_ascii=False),
                    json.dumps(list(c.traits), ensure_ascii=False),
                    c.effect_text,
                    c.trigger_text,
                    c.rarity,
                    c.image_url,
                    json.dumps(c.raw_json, ensure_ascii=False, sort_keys=True),
                )
                for c in cards
            ],
        )

        chunks_upserted = upsert_chunks(conn, chunks)
        now = utc_now_iso()
        set_meta(conn, "cards_source", source.source_name)
        set_meta(conn, "cards_last_ingested_at", now)
        set_meta(conn, "last_ingested_at", now)

    return CardsIngestSummary(
        raw_records_seen=len(raw_records),
        cards_upserted=len(cards),
        variants_merged=variants_merged,
        chunks_upserted=chunks_upserted,
        source_name=source.source_name,
    )
