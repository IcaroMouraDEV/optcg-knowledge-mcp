"""Tests for Comprehensive Rules PDF/text parsing and idempotent ingestion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pymupdf
import pytest

from optcg_knowledge.db import get_meta
from optcg_knowledge.ingest.dates import normalize_official_date
from optcg_knowledge.ingest.rules_parser import (
    ingest_rules,
    parse_rules_pdf,
    parse_rules_text,
)
from optcg_knowledge.search import get_card_rulings
from tests.conftest import insert_test_ruling


def test_parse_rules_text_hierarchy_metadata_and_toc_filtering(sample_rules_text: str) -> None:
    doc = parse_rules_text(sample_rules_text)

    assert doc.rules_version == "1.24"
    assert doc.effective_date == "2025-09-01"

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
    assert get_meta(db_conn, "rules_effective_date") == "2025-09-01"


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
    assert get_meta(db_conn, "rules_effective_date") == "2026-01-01"
    db_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")


def test_parse_rules_real_bandai_version_and_mdy_date_pdf_and_outdated_detection(
    db_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """Regression test for live Bandai PDF header: Version 1.2.1 + Last updated: 8/28/2026."""
    bandai_text = (
        "1\n\n"
        "ONE PIECE CARD GAME Comprehensive Rules\n"
        "Version 1.2.1\n\n"
        "Last updated: 8/28/2026\n\n"
        "Table of Contents\n"
        "1. Game Overview .................................................... 1\n\n"
        "1. Game Overview\n"
        "1-1. Number of Players\n"
        "1-1-1. ONE PIECE CARD GAME is a game played by two players.\n"
    )
    pdf_path = tmp_path / "rule_comprehensive.pdf"
    pdf = pymupdf.open()
    try:
        page = pdf.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), bandai_text, fontsize=10)
        pdf.save(pdf_path)
    finally:
        pdf.close()

    parsed = parse_rules_pdf(pdf_path)
    assert parsed.rules_version == "1.2.1"
    assert parsed.effective_date == "2026-08-28"
    assert len(parsed.rules) == 3
    assert all(r.rules_version == "1.2.1" for r in parsed.rules)
    assert all(r.effective_date == "2026-08-28" for r in parsed.rules)

    summary = ingest_rules(db_conn, parsed)
    assert summary.rules_version == "1.2.1"
    assert summary.effective_date == "2026-08-28"
    assert get_meta(db_conn, "rules_version") == "1.2.1"
    assert get_meta(db_conn, "rules_effective_date") == "2026-08-28"

    # With meta.rules_version populated, get_card_rulings() flags older rulings as outdated.
    insert_test_ruling(
        db_conn,
        ruling_id="faq:op05:old",
        question="Old FAQ question for OP05-119?",
        answer="Old FAQ answer.",
        source="official_faq",
        authority=2,
        date="2025-01-10",
        rules_version="1.20",
        card_ids=["OP05-119"],
    )
    insert_test_ruling(
        db_conn,
        ruling_id="judge:op05:current",
        question="Current judge question for OP05-119?",
        answer="Current judge answer.",
        source="judge",
        authority=3,
        date="2026-09-01",
        rules_version="1.2.1",
        card_ids=["OP05-119"],
    )

    rulings_resp = get_card_rulings(db_conn, "OP05-119")
    assert rulings_resp["current_rules_version"] == "1.2.1"
    by_rid = {item["ruling_id"]: item for item in rulings_resp["rulings"]}
    assert by_rid["faq:op05:old"]["potentially_outdated"] is True
    assert by_rid["judge:op05:current"]["potentially_outdated"] is False


@pytest.mark.parametrize(
    ("header", "expected_version", "expected_date"),
    [
        ("Ver. 1.24\nEffective Date: September 1, 2025", "1.24", "2025-09-01"),
        ("Ver 2.0\nEffective: 2026-03-15", "2.0", "2026-03-15"),
        ("Version 1.2.1\nLast updated: 8/28/2026", "1.2.1", "2026-08-28"),
        ("Version: 3.4.5\nUpdated: 2026/1/5", "3.4.5", "2026-01-05"),
        ("Version - 1.3.0\nLast Updated: Oct. 17, 2025", "1.3.0", "2025-10-17"),
    ],
)
def test_parse_rules_version_and_date_variants(
    header: str,
    expected_version: str,
    expected_date: str,
) -> None:
    text = f"ONE PIECE CARD GAME Comprehensive Rules\n{header}\n\n1. Overview\n1-1. Basics\n"
    doc = parse_rules_text(text)
    assert doc.rules_version == expected_version
    assert doc.effective_date == expected_date


def test_parse_rules_explicit_overrides_normalize_date() -> None:
    text = (
        "ONE PIECE CARD GAME Comprehensive Rules\n"
        "Version 1.2.1\n"
        "Last updated: 8/28/2026\n"
        "1. Overview\n"
    )
    doc = parse_rules_text(text, rules_version=" 2.0.0 ", effective_date="August 30, 2026")
    assert doc.rules_version == "2.0.0"
    assert doc.effective_date == "2026-08-30"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("   ", None),
        ("2026-08-28", "2026-08-28"),
        ("2026/8/5", "2026-08-05"),
        ("2026.08.05", "2026-08-05"),
        ("8/28/2026", "2026-08-28"),
        ("10/01/2025", "2025-10-01"),
        ("October 17, 2025", "2025-10-17"),
        ("Oct. 7, 2025", "2025-10-07"),
        ("13/40/2026", "13/40/2026"),
        ("custom-tag", "custom-tag"),
    ],
)
def test_normalize_official_date(raw: str | None, expected: str | None) -> None:
    assert normalize_official_date(raw) == expected
