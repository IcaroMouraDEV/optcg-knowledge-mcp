"""Offline tests for Bandai Tournament Rules Manual (TRM) parsing, ingestion, and search."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pymupdf
import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.db import get_meta, open_db
from optcg_knowledge.ingest import (
    HashEmbedder,
    embed_chunks,
    ingest_rules,
    ingest_trm,
    parse_rules_text,
    parse_trm_pdf,
    parse_trm_text,
)
from optcg_knowledge.search import detect_query_tokens, get_trm_section, hybrid_search

_SAMPLE_TRM_PAGES: tuple[str, ...] = (
    (
        "1\n"
        "Last Updated: October 17, 2025\n"
        "BANDAI ORGANIZED PLAY\n"
        "One Piece Card Game\n"
        "Tournament Rules Manual\n"
        "The Bandai Organized Play Tournament Rules Manual outlines the requirements of all\n"
        "participants at any Bandai Organized Play event.\n"
    ),
    (
        "2\n"
        "Table of Contents\n"
        "1.0 Event Participants ....................................................... 3\n"
        "1.1\n"
        "Players ...................................................................... 3\n"
        "4.0 Tournament Play .......................................................... 4\n"
        "4.8 Note Taking .............................................................. 4\n"
        "7.0 Tournament Penalties ..................................................... 5\n"
        "7.4 Examples of Infractions .................................................. 5\n"
        "7.4.1 Game Play Error ........................................................ 5\n"
        "7.4.2 Event Errors ........................................................... 6\n"
        "8.0 Online Event Supplement .................................................. 7\n"
        "8.4 Player Setup ............................................................. 7\n"
    ),
    (
        "3\n"
        "1.0 Event Participants\n"
        "All event participants must conduct themselves in a sensible manner.\n"
        "1.1 Players\n"
        "Players have the following responsibilities:\n"
        "•\n"
        "To understand the rules and standards of player behavior.\n"
        "•\n"
        "To communicate clearly with opponents and judges.\n"
        "1. Register a single Bandai TCG+ account.\n"
        "2. Keep your account information up to date.\n"
    ),
    (
        "4\n"
        "4.0 Tournament Play\n"
        "Matches follow official Bandai tournament procedures.\n"
        "4.8 Note Taking\n"
        "Players are not allowed to take notes in official Bandai organized play events.\n"
    ),
    (
        "5\n"
        "7.0 Tournament Penalties\n"
        "Judges issue penalties to educate players and maintain tournament integrity.\n"
        "7.4 Examples of Infractions\n"
        "Infractions occur with differing levels of severity.\n"
        "7.4.1 Game Play Error\n"
        "These can be wrong or inaccurate actions by a player contrary to game rules.\n"
        "1. Failing to Observe Game Rules\n"
        "An infraction in which a player failed to observe a rule or misinterpreted it.\n"
        "Penalty: Warning\n"
        "Resolution: Issue a Warning to the player responsible and reverse the game state\n"
        "to before the error occurred if possible.\n"
        "2. Forgetting to Resolve Automatic Actions\n"
        "An infraction in which a player proceeds with the game after failing to resolve\n"
        "an automatic effect or missed trigger.\n"
        "Penalty: Warning\n"
        "Resolution: Issue a Warning and carry out the mandatory effect from the closest\n"
        "checkpoint. If the effect was voluntary, treat it as declined.\n"
    ),
    (
        "6\n"
        "7.4.2 Event Errors\n"
        "These are errors which occur when players do not adhere to the Floor Rules.\n"
        "1. Tardiness\n"
        "An infraction in which a player was not seated at the beginning of a match.\n"
        "Penalty:\n"
        "Warning - within 5 minutes of match start.\n"
        "Game Loss - 5 or more minutes late.\n"
        "Resolution: Issue the appropriate penalty based on arrival time.\n"
        "2. Slow Play\n"
        "An infraction in which a player plays unreasonably slowly without advancing\n"
        "the game state.\n"
        "Penalty: Warning\n"
        "Resolution: Issue a Warning and instruct the player to maintain a reasonable pace.\n"
    ),
    (
        "7\n"
        "8.0 Online Event Supplement\n"
        "Additional guidelines for official online events.\n"
        "8.4 Player Setup\n"
        "Players must ensure their play area is clearly visible on camera.\n"
        "8.4 Setup Examples\n"
        "The Life area and hand must remain visible within the camera frame.\n"
    ),
)

_SAMPLE_COMP_RULES = """\
ONE PIECE CARD GAME Comprehensive Rules
Ver. 1.24
7. Game Progression
7-1. Battle
7-1-4. Damage Step
7-1-4-1. If the attacking card's power is greater than or equal to the target, deal damage.
"""


def _write_synthetic_trm_pdf(pdf_path: Path, pages: tuple[str, ...] = _SAMPLE_TRM_PAGES) -> Path:
    doc = pymupdf.open()
    try:
        for page_text in pages:
            page = doc.new_page(width=612, height=792)
            page.insert_text((36, 48), page_text, fontsize=9)
        doc.save(pdf_path)
    finally:
        doc.close()
    return pdf_path


def test_parse_trm_pdf_roundtrip_toc_duplicate_heading_and_infraction_split(
    tmp_path: Path,
) -> None:
    pdf_path = _write_synthetic_trm_pdf(tmp_path / "tournament_rules_manual.pdf")
    parsed = parse_trm_pdf(pdf_path)

    assert parsed.effective_date == "2025-10-17"
    assert parsed.trm_version is None

    by_id = {sec.section_id: sec for sec in parsed.sections}

    # Canonical TRM-* IDs with zero TOC duplication.
    expected_ids = {
        "TRM-1.0",
        "TRM-1.1",
        "TRM-4.0",
        "TRM-4.8",
        "TRM-7.0",
        "TRM-7.4",
        "TRM-7.4.1",
        "TRM-7.4.1.1",
        "TRM-7.4.1.2",
        "TRM-7.4.2",
        "TRM-7.4.2.1",
        "TRM-7.4.2.2",
        "TRM-8.0",
        "TRM-8.4",
    }
    assert set(by_id) == expected_ids

    # Hierarchy checks.
    assert by_id["TRM-7.0"].parent_id is None
    assert by_id["TRM-7.4"].parent_id == "TRM-7.0"
    assert by_id["TRM-7.4.1"].parent_id == "TRM-7.4"
    assert by_id["TRM-7.4.1.2"].parent_id == "TRM-7.4.1"
    assert by_id["TRM-7.4.1.2"].title == "Forgetting to Resolve Automatic Actions"
    assert by_id["TRM-7.4.1.2"].section == "7.0 Tournament Penalties"
    assert "Penalty: Warning" in by_id["TRM-7.4.1.2"].text
    assert "closest checkpoint" in by_id["TRM-7.4.1.2"].text

    # Ordinary numbered list in 1.1 Players must NOT be falsely split into TRM-1.1.1.
    assert "TRM-1.1.1" not in by_id
    assert "Register a single Bandai TCG+ account." in by_id["TRM-1.1"].text
    assert "• To understand the rules" in by_id["TRM-1.1"].text

    # Duplicate 8.4 heading (Player Setup + Setup Examples) merges cleanly into TRM-8.4.
    assert "clearly visible on camera" in by_id["TRM-8.4"].text
    assert "Setup Examples" in by_id["TRM-8.4"].text
    assert "camera frame" in by_id["TRM-8.4"].text


def test_ingest_trm_idempotency_stale_cleanup_and_isolation_from_rules(
    db_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    comp_doc = parse_rules_text(_SAMPLE_COMP_RULES)
    ingest_rules(db_conn, comp_doc)

    pdf_path = _write_synthetic_trm_pdf(tmp_path / "trm.pdf")
    trm_doc = parse_trm_pdf(pdf_path)

    first = ingest_trm(db_conn, trm_doc)
    assert first.sections_upserted == 14
    assert first.sections_removed == 0
    assert first.chunks_upserted == 14
    assert get_meta(db_conn, "trm_effective_date") == "2025-10-17"
    assert get_meta(db_conn, "trm_last_ingested_at") is not None

    embedder = HashEmbedder(dimension=64)
    embed_summary = embed_chunks(db_conn, embedder)
    assert embed_summary.embedded_chunks == 18  # 4 Comprehensive Rules + 14 TRM

    # Second run on the exact same TRM PDF is completely idempotent and preserves embeddings.
    second = ingest_trm(db_conn, trm_doc)
    assert second.sections_upserted == 14
    assert second.sections_removed == 0
    assert (
        db_conn.execute("SELECT COUNT(*) FROM trm_sections").fetchone()[0] == 14
    )
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_type = 'tournament_rule'"
        ).fetchone()[0]
        == 14
    )
    reembed_check = embed_chunks(db_conn, embedder)
    assert reembed_check.embedded_chunks == 0
    assert reembed_check.skipped_chunks == 18

    # Re-ingesting Comprehensive Rules must NOT touch TRM sections or chunks.
    ingest_rules(db_conn, comp_doc)
    assert db_conn.execute("SELECT COUNT(*) FROM trm_sections").fetchone()[0] == 14
    assert (
        db_conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_type = 'tournament_rule'"
        ).fetchone()[0]
        == 14
    )

    # Removing Chapter 8 in a newer TRM cleans up stale TRM rows & vectors without touching rules.
    trimmed_doc = parse_trm_text("\n".join(_SAMPLE_TRM_PAGES[:-1]))
    third = ingest_trm(db_conn, trimmed_doc)
    assert third.sections_upserted == 12
    assert third.sections_removed == 2  # TRM-8.0 and TRM-8.4
    assert db_conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0] == 4
    assert (
        db_conn.execute("SELECT COUNT(*) FROM chunks WHERE doc_type = 'rule'").fetchone()[0] == 4
    )
    assert (
        db_conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 16
    )
    db_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")

    # Empty TRM input is rejected cleanly without wiping existing data.
    with pytest.raises(ValueError, match="No TRM sections"):
        ingest_trm(db_conn, parse_trm_text("Cover Page Only\nNo numbered headings"))
    assert db_conn.execute("SELECT COUNT(*) FROM trm_sections").fetchone()[0] == 12


def test_trm_hybrid_search_exact_tokens_doc_types_and_get_trm_section(
    db_conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    ingest_rules(db_conn, parse_rules_text(_SAMPLE_COMP_RULES))
    ingest_trm(db_conn, parse_trm_pdf(_write_synthetic_trm_pdf(tmp_path / "trm.pdf")))
    embedder = HashEmbedder(dimension=64)
    embed_chunks(db_conn, embedder)

    # Query token extraction recognizes TRM IDs without confusing them with Comprehensive Rule IDs.
    tokens = detect_query_tokens("Check trm-7.4.1.2 and rule 7-1-4-1 for OP05-119")
    assert tokens.trm_ids == ("TRM-7.4.1.2",)
    assert tokens.rule_ids == ("7-1-4-1",)
    assert tokens.card_ids == ("OP05-119",)

    # Hybrid semantic/lexical query for missed triggers surfaces TRM-7.4.1.2 with source metadata.
    missed = hybrid_search(
        db_conn,
        "forgetting to resolve automatic actions missed trigger penalty",
        embedder=embedder,
        k=5,
    )
    assert missed["search_mode"] == "hybrid"
    assert missed["results"][0]["doc_type"] == "tournament_rule"
    assert missed["results"][0]["doc_id"] == "TRM-7.4.1.2"
    assert missed["results"][0]["source"] == "tournament_rules"
    assert missed["results"][0]["authority"] == 1

    # Query for Slow Play surfaces TRM-7.4.2.2.
    slow_play = hybrid_search(
        db_conn,
        "slow play unreasonably slowly warning pace",
        embedder=embedder,
        k=5,
    )
    assert slow_play["results"][0]["doc_id"] == "TRM-7.4.2.2"

    # Exact TRM token lookup boosts exact section first and surfaces children via prefix match.
    exact = hybrid_search(db_conn, "TRM-7.4.1", embedder=embedder, k=6)
    assert exact["detected_tokens"]["trm_ids"] == ["TRM-7.4.1"]
    assert exact["results"][0]["doc_id"] == "TRM-7.4.1"
    assert "exact_trm_id" in exact["results"][0]["match_signals"]
    top_ids = [r["doc_id"] for r in exact["results"]]
    assert "TRM-7.4.1.1" in top_ids
    assert "TRM-7.4.1.2" in top_ids

    # doc_types filter keeps Comprehensive Rules and TRM strictly separable.
    only_trm = hybrid_search(
        db_conn, "7-1-4-1 damage penalty", embedder=embedder, doc_types=["tournament_rule"], k=10
    )
    assert only_trm["results"]
    assert all(r["doc_type"] == "tournament_rule" for r in only_trm["results"])

    only_rules = hybrid_search(
        db_conn, "7-1-4-1 damage penalty", embedder=embedder, doc_types=["rule"], k=10
    )
    assert only_rules["results"]
    assert all(r["doc_type"] == "rule" for r in only_rules["results"])

    # Structured lookup via get_trm_section accepts with or without TRM- prefix.
    sec = get_trm_section(db_conn, "7.4.1", include_children=True)
    assert sec["found"] is True
    assert sec["section_id"] == "TRM-7.4.1"
    assert sec["source"] == "tournament_rules"
    assert sec["trm_effective_date"] == "2025-10-17"
    assert [a["section_id"] for a in sec["ancestors"]] == ["TRM-7.0", "TRM-7.4"]
    assert [c["section_id"] for c in sec["children"]] == ["TRM-7.4.1.1", "TRM-7.4.1.2"]

    sec_no_children = get_trm_section(db_conn, "trm-7.4.1", include_children=False)
    assert sec_no_children["found"] is True
    assert sec_no_children["children"] == []

    # Chapter-level lookup (TRM-7.0) returns all descendants under Chapter 7 in natural order.
    chapter = get_trm_section(db_conn, "TRM-7.0", include_children=True)
    assert chapter["found"] is True
    assert [c["section_id"] for c in chapter["children"]] == [
        "TRM-7.4",
        "TRM-7.4.1",
        "TRM-7.4.1.1",
        "TRM-7.4.1.2",
        "TRM-7.4.2",
        "TRM-7.4.2.1",
        "TRM-7.4.2.2",
    ]

    missing = get_trm_section(db_conn, "TRM-99.9")
    assert missing["found"] is False
    assert missing["section"] is None


def test_cli_trm_command_roundtrip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "cli_trm.db"
    pdf_path = _write_synthetic_trm_pdf(tmp_path / "tournament_rules_manual.pdf")

    rc = cli_main(
        [
            "--db",
            str(db_path),
            "trm",
            str(pdf_path),
            "--version",
            "2025.10",
        ]
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "Ingested 14 TRM sections" in stdout
    assert "version=2025.10" in stdout
    assert "effective_date=2025-10-17" in stdout

    with open_db(db_path, init=False) as conn:
        assert get_meta(conn, "trm_version") == "2025.10"
        assert get_meta(conn, "trm_effective_date") == "2025-10-17"
        assert conn.execute("SELECT COUNT(*) FROM trm_sections").fetchone()[0] == 14


def test_parse_trm_supports_mdy_date_and_three_part_version() -> None:
    raw = (
        "BANDAI ORGANIZED PLAY\n"
        "One Piece Card Game Tournament Rules Manual\n"
        "Version: 2.1.0\n"
        "Last updated: 8/28/2026\n"
        "1.0 Event Participants\n"
        "All participants must follow official guidelines.\n"
    )
    parsed = parse_trm_text(raw)
    assert parsed.trm_version == "2.1.0"
    assert parsed.effective_date == "2026-08-28"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].trm_version == "2.1.0"
    assert parsed.sections[0].effective_date == "2026-08-28"
