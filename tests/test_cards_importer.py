"""Tests for card normalization, sources, chunk generation, and idempotency."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx

from optcg_knowledge.db import get_meta
from optcg_knowledge.ingest.cards_importer import ingest_cards, normalize_cards, parse_card_id
from optcg_knowledge.ingest.sources import LocalCardSource, PunkRecordsSource


def test_parse_card_id_handles_base_variants_and_promos() -> None:
    assert parse_card_id("OP05-119") == ("OP05-119", False)
    assert parse_card_id("op05-119_p1") == ("OP05-119", True)
    assert parse_card_id("ST10-001_r1") == ("ST10-001", True)
    assert parse_card_id("P-001") == ("P-001", False)


def test_normalize_cards_deduplicates_parallels_and_maps_leader_life(
    fixtures_dir: Path,
) -> None:
    raw = json.loads((fixtures_dir / "sample_cards_punk_records.json").read_text(encoding="utf-8"))
    cards, variants_merged = normalize_cards(raw)

    assert len(cards) == 5
    assert variants_merged == 1

    by_id = {card.card_id: card for card in cards}

    # OP05-119 prefers base printing over _p1 even though _p1 appeared first in JSON,
    # while promoting block_icon to the highest printing block across variants (3 > 1).
    luffy = by_id["OP05-119"]
    assert luffy.set_code == "OP05"
    assert luffy.cost == 10
    assert luffy.life is None
    assert luffy.power == 12000
    assert luffy.block_icon == 3
    assert luffy.image_url == "https://en.onepiece-cardgame.com/images/cardlist/card/OP05-119.png"
    assert "variants" in luffy.raw_json
    assert len(luffy.raw_json["variants"]) == 1
    assert luffy.raw_json["variants"][0]["id"] == "OP05-119_p1"
    assert luffy.raw_json["variants"][0]["block_icon"] == 3

    # Leaders convert vegapull's `cost` field into `life` and leave `cost` None.
    zoro = by_id["OP01-001"]
    assert zoro.category == "Leader"
    assert zoro.life == 5
    assert zoro.cost is None
    assert zoro.colors == ("Red",)
    assert zoro.block_icon == 1

    law = by_id["ST10-001"]
    assert law.set_code == "ST10"
    assert law.life == 4
    assert law.cost is None
    assert law.colors == ("Red", "Purple")
    assert law.block_icon == 2

    # Events keep both effect_text and trigger_text.
    guard = by_id["ST01-014"]
    assert guard.category == "Event"
    assert guard.cost == 1
    assert guard.block_icon == 1
    assert guard.effect_text is not None and "[Counter]" in guard.effect_text
    assert guard.trigger_text is not None and "[Trigger]" in guard.trigger_text

    # Promos derive set_code = "P" and allow null block_icon.
    promo = by_id["P-001"]
    assert promo.set_code == "P"
    assert promo.rarity == "Promo"
    assert promo.block_icon is None


def test_local_source_directory_ignores_metadata_and_index(
    tmp_path: Path,
    fixtures_dir: Path,
) -> None:
    cards_dir = tmp_path / "cards"
    (cards_dir / "cards").mkdir(parents=True)
    (cards_dir / "index").mkdir(parents=True)

    sample_json = (fixtures_dir / "sample_cards_punk_records.json").read_text(encoding="utf-8")
    (cards_dir / "cards" / "OP05.json").write_text(sample_json, encoding="utf-8")
    (cards_dir / "packs.json").write_text('[{"id": "OP05"}]', encoding="utf-8")
    (cards_dir / "manifest.json").write_text('{"version": 1}', encoding="utf-8")
    (cards_dir / "index" / "cards_by_id.json").write_text("{}", encoding="utf-8")

    source = LocalCardSource(cards_dir)
    raw_cards = source.fetch_raw_cards()
    assert len(raw_cards) == 6


def test_punk_records_source_offline_with_mock_transport(fixtures_dir: Path) -> None:
    sample_cards = json.loads(
        (fixtures_dir / "sample_cards_punk_records.json").read_text(encoding="utf-8")
    )

    # Verify v2.0 layout: dict packs.json + data/{pack_id}.json.
    def handler_v2(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/packs.json"):
            return httpx.Response(
                200,
                json={
                    "569105": {"id": "569105", "raw_title": "BOOSTER PACK [OP-05]"},
                    "999999": {"id": "999999", "raw_title": "MISSING"},
                },
            )
        if request.url.path.endswith("/data/569105.json"):
            return httpx.Response(200, json=sample_cards)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler_v2)) as client:
        source = PunkRecordsSource(
            base_url="https://example.invalid/punk-records/english",
            client=client,
        )
        assert len(source.fetch_raw_cards()) == 6

    # Verify legacy layout fallback: list packs.json + cards/{pack_id}.json.
    def handler_v1(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/packs.json"):
            return httpx.Response(200, json=[{"id": "OP05"}])
        if request.url.path.endswith("/cards/OP05.json"):
            return httpx.Response(200, json=sample_cards)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler_v1)) as client:
        source_v1 = PunkRecordsSource(
            base_url="https://example.invalid/punk-records/english",
            client=client,
        )
        assert len(source_v1.fetch_raw_cards()) == 6


def test_ingest_cards_chunks_and_idempotency(
    db_conn: sqlite3.Connection,
    fixtures_dir: Path,
) -> None:
    source = LocalCardSource(fixtures_dir / "sample_cards_punk_records.json")

    first = ingest_cards(db_conn, source)
    second = ingest_cards(db_conn, source)

    assert first.cards_upserted == 5
    assert first.variants_merged == 1
    assert second.cards_upserted == 5
    assert second.variants_merged == 1

    card_count = db_conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    chunk_count = db_conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_type = 'card'"
    ).fetchone()[0]
    fts_count = db_conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    assert card_count == 5
    assert chunk_count == 5
    assert fts_count == 5

    chunk = db_conn.execute(
        "SELECT text, metadata FROM chunks WHERE chunk_id = 'card:ST01-014'"
    ).fetchone()
    assert chunk is not None
    assert "ST01-014 — Guard Point" in chunk["text"]
    assert "[Counter]" in chunk["text"]
    assert "[Trigger]" in chunk["text"]
    assert "Block:" not in chunk["text"]
    assert json.loads(chunk["metadata"])["block_icon"] == 1

    persisted_blocks = {
        str(r["card_id"]): r["block_icon"]
        for r in db_conn.execute("SELECT card_id, block_icon FROM cards").fetchall()
    }
    assert persisted_blocks == {
        "OP01-001": 1,
        "OP05-119": 3,
        "P-001": None,
        "ST01-014": 1,
        "ST10-001": 2,
    }

    assert get_meta(db_conn, "cards_source") is not None
    db_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")


def test_normalize_cards_handles_local_block_icon_aliases_and_invalid_values() -> None:
    cards, _ = normalize_cards(
        [
            {"card_id": "OP09-001", "name": "Shanks", "category": "Leader", "block_icon": "3"},
            {"card_id": "OP10-001", "name": "Smoker", "category": "Leader", "block": 4},
            {"card_id": "OP11-001", "name": "Koby", "category": "Leader", "block_number": "-"},
            {"card_id": "OP12-001", "name": "Zero", "category": "Leader", "block_number": 0},
            {"card_id": "OP13-001", "name": "Bool", "category": "Leader", "block_icon": True},
        ]
    )
    by_id = {card.card_id: card for card in cards}
    assert by_id["OP09-001"].block_icon == 3
    assert by_id["OP10-001"].block_icon == 4
    assert by_id["OP11-001"].block_icon is None
    assert by_id["OP12-001"].block_icon is None
    assert by_id["OP13-001"].block_icon is None
