"""Tests for Comprehensive Rules PDF/text parsing and idempotent ingestion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from optcg_knowledge.db import get_meta
from optcg_knowledge.ingest.rules_parser import (
    ingest_rules,
    parse_rules_pdf,
    parse_rules_text,
)


def test_parse_rules_text_hierarchy_metadata_and_toc_filtering(sample_rules_text: str) -> None:
    doc = parse_rules_text(sample_rules_text)

    assert doc.rules_version == "1.24"
    assert doc.effective_date == "September 1, 2025"

    by_id = {rule.rule_id: rule for rule in doc.rules}

    # Verify top-level sections and deep hierarchy.
    assert "1" in by_id
    assert by_id["1"].parent_id is None
    assert by_id["1-1"].parent_id == "1"
    assert by_id["1-1-1"].parent_id == "1-1"
    assert by_id["1-1-1"].section == "1. Game Overview"

    assert by_id["7-1-4-1"].parent_id == "7-1-4"
    assert by_id["7-1-4-1-1"].parent_id == "7-1-4-1"
    assert by_id["7-1-4-1-1-1"].parent_id == "7-1-4-1-1"
    assert by_id["7-1-4-1-1-1"].section == "7. Card Attacks and Battles"

    # Inline numbered list inside 1-3-2 must stay inside 1-3-2 and not overwrite section 1.
    assert "Perform the first action" in by_id["1-3-2"].text
    assert "Game Overview" in by_id["1"].text

    # Example and ※ notes are preserved on their parent rules.
    assert "Example:" in by_id["1-3-1"].text
    assert "[Rush]" in by_id["1-3-1"].text
    assert "※" in by_id["10-1-1"].text
    assert "[Blocker]" in by_id["10-1-1"].text


def test_parse_rules_pdf_roundtrip(sample_rules_pdf: Path) -> None:
    doc = parse_rules_pdf(sample_rules_pdf)
    rule_ids = {rule.rule_id for rule in doc.rules}

    assert doc.rules_version == "1.24"
    assert "7-1-4-1-1-1" in rule_ids
    assert "10-1-1" in rule_ids


def test_ingest_rules_chunks_and_idempotency(
    db_conn: sqlite3.Connection,
    sample_rules_pdf: Path,
) -> None:
    doc = parse_rules_pdf(sample_rules_pdf)

    first = ingest_rules(db_conn, doc)
    second = ingest_rules(db_conn, doc)

    assert first.rules_upserted == len(doc.rules)
    assert second.rules_upserted == len(doc.rules)
    assert second.rules_removed == 0

    rule_count = db_conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
    chunk_count = db_conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE doc_type = 'rule'"
    ).fetchone()[0]
    fts_count = db_conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    assert rule_count == len(doc.rules)
    assert chunk_count == len(doc.rules)
    assert fts_count == len(doc.rules)

    # Rule chunks include full ancestor path for context.
    chunk_row = db_conn.execute(
        "SELECT text, authority FROM chunks WHERE chunk_id = 'rule:7-1-4-1-1-1'"
    ).fetchone()
    assert chunk_row is not None
    assert chunk_row["authority"] == 1
    assert "7. Card Attacks and Battles" in chunk_row["text"]
    assert "7-1. Attack Steps" in chunk_row["text"]
    assert "7-1-4. Damage Step" in chunk_row["text"]
    assert "0 Life" in chunk_row["text"]

    assert get_meta(db_conn, "rules_version") == "1.24"
    assert get_meta(db_conn, "rules_effective_date") == "September 1, 2025"


def test_ingest_rules_removes_stale_rules_on_version_upgrade(
    db_conn: sqlite3.Connection,
    sample_rules_text: str,
) -> None:
    v1_doc = parse_rules_text(sample_rules_text)
    ingest_rules(db_conn, v1_doc)

    v2_text = """
    ONE PIECE CARD GAME Comprehensive Rules
    Ver. 1.25
    Effective Date: January 1, 2026

    1. Game Overview
    1-1. Number of Players
    1-1-1. Played by two players.
    """
    v2_doc = parse_rules_text(v2_text)
    summary = ingest_rules(db_conn, v2_doc)

    assert summary.rules_upserted == 3
    assert summary.rules_removed == len(v1_doc.rules) - 3
    assert db_conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 3
    assert db_conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_type = 'rule'").fetchone()[0] == 3
    assert get_meta(db_conn, "rules_version") == "1.25"
    db_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
