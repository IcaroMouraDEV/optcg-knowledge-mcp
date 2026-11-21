"""Shared pytest fixtures for offline optcg-knowledge tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest

from optcg_knowledge.db import ChunkRecord, connect
from optcg_knowledge.ingest import (
    HashEmbedder,
    compute_content_hash,
    embed_chunks,
    ingest_cards,
    ingest_rules,
    parse_rules_text,
    upsert_chunks,
)
from optcg_knowledge.ingest.sources import LocalCardSource

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def sample_rules_text() -> str:
    return (FIXTURES_DIR / "sample_rules_text.txt").read_text(encoding="utf-8")


@pytest.fixture()
def sample_rules_pdf(tmp_path: Path, sample_rules_text: str) -> Path:
    """Create a real multi-page PDF on disk from the sample rules text."""
    pdf_path = tmp_path / "rule_comprehensive.pdf"
    doc = pymupdf.open()
    lines = sample_rules_text.splitlines()
    lines_per_page = 14
    for start in range(0, len(lines), lines_per_page):
        page = doc.new_page(width=595, height=842)
        chunk = "\n".join(lines[start : start + lines_per_page])
        page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), chunk, fontsize=10)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture()
def db_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Provide an initialized temporary SQLite database with FTS5 + sqlite-vec."""
    db_path = tmp_path / "test_optcg.db"
    conn = connect(db_path, init=True, embedding_model="test/hash-embedder", embedding_dim=64)
    try:
        yield conn
    finally:
        conn.close()


def insert_test_ruling(
    conn: sqlite3.Connection,
    *,
    ruling_id: str,
    question: str,
    answer: str,
    source: str,
    authority: int,
    date: str | None = None,
    rules_version: str | None = None,
    url: str | None = None,
    superseded_by: str | None = None,
    reviewed: bool = True,
    card_ids: list[str] | None = None,
    rule_ids: list[str] | None = None,
) -> None:
    """Insert a ruling, its card/rule links, and its searchable chunk for tests."""
    linked_cards = card_ids or []
    linked_rules = rule_ids or []
    with conn:
        conn.execute(
            """
            INSERT INTO rulings (
                ruling_id, question, answer, source, authority,
                date, rules_version, url, superseded_by, reviewed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ruling_id,
                question,
                answer,
                source,
                authority,
                date,
                rules_version,
                url,
                superseded_by,
                1 if reviewed else 0,
            ),
        )
        for cid in linked_cards:
            conn.execute(
                "INSERT INTO ruling_cards (ruling_id, card_id) VALUES (?, ?)",
                (ruling_id, cid),
            )
        for rid in linked_rules:
            conn.execute(
                "INSERT INTO ruling_rules (ruling_id, rule_id) VALUES (?, ?)",
                (ruling_id, rid),
            )

        chunk_text = f"Q: {question}\nA: {answer}"
        chunk = ChunkRecord(
            chunk_id=f"ruling:{ruling_id}",
            doc_type="ruling",
            doc_id=ruling_id,
            text=chunk_text,
            authority=authority,  # type: ignore[arg-type]
            metadata={
                "ruling_id": ruling_id,
                "source": source,
                "date": date,
                "rules_version": rules_version,
                "card_ids": linked_cards,
                "rule_ids": linked_rules,
            },
            content_hash=compute_content_hash(chunk_text),
        )
        upsert_chunks(conn, [chunk])


@pytest.fixture()
def populated_db(
    tmp_path: Path,
    sample_rules_text: str,
    fixtures_dir: Path,
) -> Iterator[tuple[Path, sqlite3.Connection, HashEmbedder]]:
    """Populate a temporary DB with rules, cards, rulings, and HashEmbedder vectors."""
    db_path = tmp_path / "populated_optcg.db"
    embedder = HashEmbedder(dimension=64)
    conn = connect(
        db_path,
        init=True,
        embedding_model=embedder.model_name,
        embedding_dim=embedder.dimension,
    )
    try:
        ingest_rules(conn, parse_rules_text(sample_rules_text))
        ingest_cards(conn, LocalCardSource(fixtures_dir / "sample_cards_punk_records.json"))

        # Insert an extra card with [On K.O.] and [Blocker] to cover bracketed keyword searches
        extra_card_path = tmp_path / "extra_cards.json"
        extra_card_path.write_text(
            json.dumps(
                [
                    {
                        "id": "OP04-083",
                        "name": "Sabo",
                        "rarity": "SuperRare",
                        "block_number": 1,
                        "category": "Character",
                        "colors": ["Black"],
                        "cost": 5,
                        "power": 6000,
                        "counter": 1000,
                        "attributes": ["Special"],
                        "types": ["Revolutionary Army"],
                        "effect": (
                            "[Blocker] [On Play] None of your Characters can be K.O.'d by "
                            "effects until the start of your next turn."
                        ),
                        "trigger": None,
                    },
                    {
                        "id": "OP02-106",
                        "name": "Tsuru",
                        "rarity": "Uncommon",
                        "block_number": 1,
                        "category": "Character",
                        "colors": ["Black"],
                        "cost": 1,
                        "power": 0,
                        "counter": 2000,
                        "attributes": ["Wisdom"],
                        "types": ["Navy"],
                        "effect": (
                            "[On K.O.] Give up to 1 of your opponent's Characters -2 cost "
                            "during this turn."
                        ),
                        "trigger": None,
                    },
                ]
            ),
            encoding="utf-8",
        )
        ingest_cards(conn, LocalCardSource(extra_card_path))

        # Active official FAQ for OP05-119 (older rules_version -> potentially_outdated=True)
        insert_test_ruling(
            conn,
            ruling_id="faq:op05:119-1",
            question=(
                "If I play OP05-119 Monkey.D.Luffy with 0 other Characters, "
                "do I still take an extra turn?"
            ),
            answer=(
                "Yes. Placing 0 other Characters at the bottom of your deck "
                "still allows taking the extra turn."
            ),
            source="official_faq",
            authority=2,
            date="2024-01-15",
            rules_version="1.20",
            url="https://en.onepiece-cardgame.com/rules/qa/",
            reviewed=True,
            card_ids=["OP05-119"],
            rule_ids=["1-3-2"],
        )
        # Active judge ruling for OP05-119 (current rules_version -> not outdated)
        insert_test_ruling(
            conn,
            ruling_id="judge:2025-09-10:001",
            question="During the extra turn granted by OP05-119, do I draw a card and add 2 DON!!?",
            answer="Yes, the extra turn follows the normal Refresh, Draw, and DON!! phases.",
            source="judge",
            authority=3,
            date="2025-09-10",
            rules_version="1.24",
            reviewed=True,
            card_ids=["OP05-119"],
            rule_ids=["1-3-2"],
        )
        # Superseded judge ruling for OP05-119
        insert_test_ruling(
            conn,
            ruling_id="judge:2023-12-01:old",
            question="Old superseded ruling about OP05-119 extra turn requiring another Character.",
            answer="Outdated answer claiming at least 1 other Character was required.",
            source="judge",
            authority=3,
            date="2023-12-01",
            rules_version="1.15",
            superseded_by="faq:op05:119-1",
            reviewed=True,
            card_ids=["OP05-119"],
        )
        # Identical-wording rulings at different authorities for tie-break authority ordering tests
        insert_test_ruling(
            conn,
            ruling_id="community:on-ko:tie",
            question="How does [On K.O.] timing resolve when a Character is trashed by battle?",
            answer=(
                "[On K.O.] activates when the Character is K.O.'d in battle "
                "during damage resolution."
            ),
            source="community",
            authority=4,
            date="2025-09-12",
            rules_version="1.24",
            reviewed=True,
            card_ids=["OP02-106"],
        )
        insert_test_ruling(
            conn,
            ruling_id="faq:on-ko:tie",
            question="How does [On K.O.] timing resolve when a Character is trashed by battle?",
            answer=(
                "[On K.O.] activates when the Character is K.O.'d in battle "
                "during damage resolution."
            ),
            source="official_faq",
            authority=2,
            date="2025-09-05",
            rules_version="1.24",
            reviewed=True,
            card_ids=["OP02-106"],
        )

        embed_chunks(conn, embedder, batch_size=4)
        yield (db_path, conn, embedder)
    finally:
        conn.close()
