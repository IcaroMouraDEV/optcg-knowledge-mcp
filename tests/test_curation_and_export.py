"""Offline tests for judge ruling curation, judge_tier precedence, curar prompt, and export."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.curation import (
    EXPORT_SCHEMA_VERSION,
    add_ruling,
    export_rulings,
    list_unreviewed_rulings,
    mark_ruling,
)
from optcg_knowledge.db import SCHEMA_VERSION, connect, get_meta, migrate_schema
from optcg_knowledge.ingest import HashEmbedder, embed_chunks
from optcg_knowledge.mcp import build_curar_prompt, create_server
from optcg_knowledge.search import get_card_rulings, hybrid_search


def _extract_json_payload(call_result: Any) -> dict[str, Any]:
    assert call_result.is_error is False
    assert call_result.content
    parsed = json.loads(call_result.content[0].text)
    assert isinstance(parsed, dict)
    return parsed


def test_schema_v1_to_v2_migration_adds_judge_tier_nondestructively(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy_v1.db"
    conn = connect(db_path, init=False)
    try:
        conn.executescript(
            """
            CREATE TABLE rulings (
                ruling_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT NOT NULL,
                authority INTEGER NOT NULL,
                date TEXT,
                rules_version TEXT,
                url TEXT,
                superseded_by TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0,
                raw_ref TEXT
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta (key, value) VALUES ('schema_version', '1');
            INSERT INTO rulings (ruling_id, question, answer, source, authority, reviewed)
            VALUES ('legacy-1', 'Q?', 'A.', 'official_faq', 2, 1);
            """
        )

        with conn:
            migrated = migrate_schema(conn)
        assert migrated is True
        assert get_meta(conn, "schema_version") == SCHEMA_VERSION

        # Second call is a no-op and preserves existing data.
        with conn:
            migrated_again = migrate_schema(conn)
        assert migrated_again is False

        row = conn.execute(
            "SELECT ruling_id, judge_tier FROM rulings WHERE ruling_id = 'legacy-1'"
        ).fetchone()
        assert row is not None
        assert row["ruling_id"] == "legacy-1"
        assert row["judge_tier"] is None
    finally:
        conn.close()


def test_add_ruling_privacy_validation_and_immediate_fts_indexing(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, _ = populated_db

    res = add_ruling(
        conn,
        question="Can OP05-119 take an extra turn if Sabo OP04-083 is on the field?",
        answer="Yes. Returning 10 DON!! still grants the extra turn during the End Phase.",
        source="judge",
        date="2026-11-18",
        card_ids=["op05-119", "OP04-083"],
        rule_refs=["7-1-4-1"],
        raw_ref="2026-11-forum.md#thread-1",
        judge_tier="T0",
    )
    assert res["created"] is True
    assert res["warnings"] == []
    ruling = res["ruling"]
    assert ruling["ruling_id"].startswith("judge:OP04-083:")
    assert ruling["authority"] == 3
    assert ruling["judge_tier"] == "tier_0"
    assert ruling["reviewed"] is False
    assert ruling["rules_version"] == "1.24"
    assert ruling["card_ids"] == ["OP04-083", "OP05-119"]
    assert ruling["rule_ids"] == ["7-1-4-1"]

    # Immediately searchable via FTS5 without running embed yet.
    search_out = hybrid_search(
        conn,
        "OP05-119 Sabo extra turn End Phase",
        doc_types=["ruling"],
        embedder=None,
    )
    found_ids = [item["doc_id"] for item in search_out["results"]]
    assert ruling["ruling_id"] in found_ids
    matched = next(item for item in search_out["results"] if item["doc_id"] == ruling["ruling_id"])
    assert matched["judge_tier"] == "tier_0"
    assert "[JUDGE RULING | TIER 0]" in matched["text"]

    # Unknown card/rule IDs produce non-blocking warnings.
    warn_res = add_ruling(
        conn,
        question="Does OP99-001 interact with Rule 99-1?",
        answer="Pending set release verification.",
        source="judge",
        date="2026-11-18",
        card_ids=["OP99-001"],
        rule_refs=["99-1"],
    )
    assert len(warn_res["warnings"]) == 2

    # Privacy guards reject personal Discord mentions, @handles, and username#0000 tags.
    with pytest.raises(ValueError, match="Discord user mention"):
        add_ruling(
            conn,
            question="Asked by <@123456789012345678> about Blocker",
            answer="Yes.",
            source="judge",
        )
    with pytest.raises(ValueError, match="@handle"):
        add_ruling(
            conn,
            question="Does Blocker work?",
            answer="Confirmed by @HeadJudge.",
            source="judge",
        )
    with pytest.raises(ValueError, match="username#0000"):
        add_ruling(
            conn,
            question="Does Blocker work?",
            answer="JudgeAlice#1234 said yes.",
            source="judge",
        )

    # Non-judge sources cannot set judge_tier; 'rules' is rejected for curation.
    with pytest.raises(ValueError, match="judge_tier can only be set"):
        add_ruling(
            conn,
            question="Community ruling?",
            answer="Maybe.",
            source="community",
            judge_tier="tier_2",
        )
    with pytest.raises(ValueError, match="Unsupported curation source"):
        add_ruling(
            conn,
            question="Rule?",
            answer="Text.",
            source="rules",
        )


def test_mark_ruling_superseding_cycle_guard_and_list_unreviewed(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, _ = populated_db

    r1 = add_ruling(
        conn,
        question="Early Tier 1 judge ruling on [Blocker] redirect timing?",
        answer="Initial Tier 1 interpretation.",
        source="judge",
        date="2026-10-01",
        card_ids=["OP04-083"],
        judge_tier="tier_1",
    )["ruling_id"]
    r2 = add_ruling(
        conn,
        question="Clarified Tier 0 judge ruling on [Blocker] redirect timing?",
        answer="Definitive Tier 0 ruling.",
        source="judge",
        date="2026-11-01",
        card_ids=["OP04-083"],
        judge_tier="tier_0",
    )["ruling_id"]

    # list_unreviewed_rulings orders Tier 0 before Tier 1 within authority=3.
    pending = list_unreviewed_rulings(conn, limit=10)
    pending_ids = [r["ruling_id"] for r in pending["rulings"]]
    assert pending_ids.index(r2) < pending_ids.index(r1)

    # Self-superseding and unknown target are rejected.
    with pytest.raises(ValueError, match="cannot supersede itself"):
        mark_ruling(conn, r1, superseded_by=r1)
    with pytest.raises(ValueError, match="Target superseded_by ruling not found"):
        mark_ruling(conn, r1, superseded_by="judge:missing:000000")

    # Approve r2 and mark r1 as superseded by r2 (r1 is preserved, not deleted).
    mark_ruling(conn, r2, reviewed=True)
    marked = mark_ruling(conn, r1, reviewed=True, superseded_by=r2)
    assert marked["ruling"]["superseded_by"] == r2
    assert marked["ruling"]["reviewed"] is True

    # Circular superseding chain (r2 -> r1 when r1 -> r2) is rejected.
    with pytest.raises(ValueError, match="circular superseding chain"):
        mark_ruling(conn, r2, superseded_by=r1)

    # Updating judge_tier refreshes chunk metadata & text.
    r3 = add_ruling(
        conn,
        question="Unassigned tier judge ruling on OP04-083?",
        answer="Answer text.",
        source="judge",
        date="2026-11-10",
        card_ids=["OP04-083"],
    )["ruling_id"]
    mark_ruling(conn, r3, judge_tier="tier_2")
    chunk = conn.execute(
        "SELECT text, metadata FROM chunks WHERE doc_type = 'ruling' AND doc_id = ?",
        (r3,),
    ).fetchone()
    assert chunk is not None
    assert "[JUDGE RULING | TIER 2]" in chunk["text"]
    assert json.loads(chunk["metadata"])["judge_tier"] == "tier_2"


def test_judge_tier_precedence_in_get_card_rulings_and_hybrid_search(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db

    # Add 3 judge rulings on OP04-083 with identical text & date across tier_1, tier_2, tier_0.
    t1_id = add_ruling(
        conn,
        question="How does Sabo OP04-083 K.O. protection resolve with simultaneous effects?",
        answer="All simultaneous K.O. effects are prevented by Sabo OP04-083.",
        source="judge",
        date="2026-11-15",
        card_ids=["OP04-083"],
        judge_tier="tier_1",
    )["ruling_id"]
    t2_id = add_ruling(
        conn,
        question="How does Sabo OP04-083 K.O. protection resolve with simultaneous effects?",
        answer="All simultaneous K.O. effects are prevented by Sabo OP04-083.",
        source="judge",
        date="2026-11-16",
        card_ids=["OP04-083"],
        judge_tier="tier_2",
    )["ruling_id"]
    t0_id = add_ruling(
        conn,
        question="How does Sabo OP04-083 K.O. protection resolve with simultaneous effects?",
        answer="All simultaneous K.O. effects are prevented by Sabo OP04-083.",
        source="judge",
        date="2026-11-14",  # Even with an older date, tier_0 outranks tier_2 and tier_1.
        card_ids=["OP04-083"],
        judge_tier="tier_0",
    )["ruling_id"]
    faq_id = add_ruling(
        conn,
        question="How does Sabo OP04-083 K.O. protection resolve with simultaneous effects?",
        answer="All simultaneous K.O. effects are prevented by Sabo OP04-083.",
        source="official_faq",
        date="2026-11-10",
        card_ids=["OP04-083"],
    )["ruling_id"]

    for rid in (t1_id, t2_id, t0_id, faq_id):
        mark_ruling(conn, rid, reviewed=True)
    embed_chunks(conn, embedder)

    # get_card_rulings: authority=2 official_faq first, then tier_0 > tier_2 > tier_1.
    card_rulings = get_card_rulings(conn, "OP04-083")
    assert [r["ruling_id"] for r in card_rulings["rulings"]] == [faq_id, t0_id, t2_id, t1_id]

    # hybrid_search tie-breaks official_faq (auth=2) > tier_0 > tier_2 > tier_1.
    search_res = hybrid_search(
        conn,
        "Sabo OP04-083 K.O. protection simultaneous effects",
        doc_types=["ruling"],
        k=10,
        embedder=embedder,
    )
    ranked_ids = [
        item["doc_id"]
        for item in search_res["results"]
        if item["doc_id"] in {faq_id, t0_id, t2_id, t1_id}
    ]
    assert ranked_ids == [faq_id, t0_id, t2_id, t1_id]


def test_export_rulings_schema_mcp_and_cli_parity(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db

    # Add an unreviewed judge ruling (must be excluded from export by default).
    unreviewed_id = add_ruling(
        conn,
        question="Pending unreviewed judge question on OP05-119?",
        answer="Draft answer awaiting approval.",
        source="judge",
        date="2026-11-19",
        card_ids=["OP05-119"],
        judge_tier="tier_2",
    )["ruling_id"]

    # Direct service export vs MCP tool export vs CLI export.
    direct_payload = export_rulings(conn, card_id="OP05-119", min_authority=3)
    assert direct_payload["schema_version"] == EXPORT_SCHEMA_VERSION
    assert direct_payload["rules_version"] == "1.24"
    assert direct_payload["filters"] == {
        "card_id": "OP05-119",
        "since": None,
        "min_authority": 3,
        "include_superseded": False,
        "include_unreviewed": False,
    }
    exported_ids = [r["ruling_id"] for r in direct_payload["rulings"]]
    assert unreviewed_id not in exported_ids
    assert exported_ids == ["faq:op05:119-1", "judge:2025-09-10:001"]

    # include_unreviewed=True surfaces the pending judge ruling with judge_tier='tier_2'.
    with_drafts = export_rulings(conn, card_id="OP05-119", include_unreviewed=True)
    assert unreviewed_id in [r["ruling_id"] for r in with_drafts["rulings"]]

    # since filter works on ISO YYYY-MM-DD dates.
    since_filtered = export_rulings(conn, card_id="OP05-119", since="2025-09-01")
    assert [r["ruling_id"] for r in since_filtered["rulings"]] == ["judge:2025-09-10:001"]

    # CLI export to file matches MCP export payload identically (except exported_at timestamp).
    out_file = tmp_path / "exported_rulings.json"
    exit_code = cli_main(
        [
            "--db",
            str(db_path),
            "export-rulings",
            "--card-id",
            "OP05-119",
            "--min-authority",
            "3",
            "--out",
            str(out_file),
        ]
    )
    assert exit_code == 0
    cli_msg = capsys.readouterr().out
    assert "Exported 2 rulings" in cli_msg
    cli_payload = json.loads(out_file.read_text(encoding="utf-8"))

    # CLI export to stdout when --out is omitted.
    exit_code_stdout = cli_main(
        ["--db", str(db_path), "export-rulings", "--card-id", "OP05-119", "--min-authority", "3"]
    )
    assert exit_code_stdout == 0
    stdout_payload = json.loads(capsys.readouterr().out)

    server = create_server(db_path=db_path, embedder=embedder)

    async def _mcp_export() -> dict[str, Any]:
        return _extract_json_payload(
            await server.call_tool(
                "export_rulings",
                {"card_id": "OP05-119", "min_authority": 3},
            )
        )

    mcp_payload = asyncio.run(_mcp_export())
    # MCP tool call must never write to stdout.
    assert capsys.readouterr().out == ""

    for payload in (cli_payload, stdout_payload, mcp_payload):
        assert payload["schema_version"] == direct_payload["schema_version"]
        assert payload["rules_version"] == direct_payload["rules_version"]
        assert payload["filters"] == direct_payload["filters"]
        assert payload["total"] == direct_payload["total"]
        assert payload["rulings"] == direct_payload["rulings"]


def test_mcp_curation_tools_and_curar_prompt_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, _, embedder = populated_db
    server = create_server(db_path=db_path, embedder=embedder)

    async def _workflow() -> dict[str, Any]:
        added = _extract_json_payload(
            await server.call_tool(
                "add_ruling",
                {
                    "question": "Can OP05-119 attack on the extra turn?",
                    "answer": "Yes, normal turn rules apply on the extra turn.",
                    "source": "judge",
                    "date": "2026-11-18",
                    "card_ids": ["OP05-119"],
                    "rule_refs": ["7-1-1-1"],
                    "judge_tier": "tier_0",
                    "raw_ref": "2026-11-forum.md#thread-2",
                },
            )
        )
        rid = added["ruling_id"]
        unreviewed = _extract_json_payload(
            await server.call_tool("list_unreviewed_rulings", {"limit": 10})
        )
        marked = _extract_json_payload(
            await server.call_tool("mark_ruling", {"ruling_id": rid, "reviewed": True})
        )
        curar_prompt = await server.get_prompt("curar", {"arquivo": "2026-11-forum.md"})
        return {
            "added": added,
            "unreviewed": unreviewed,
            "marked": marked,
            "curar_prompt": curar_prompt,
        }

    out = asyncio.run(_workflow())

    # Strict stdio protocol integrity: zero stdout leakage across all calls.
    assert capsys.readouterr().out == ""

    rid = out["added"]["ruling_id"]
    assert out["added"]["ruling"]["reviewed"] is False
    assert out["added"]["ruling"]["judge_tier"] == "tier_0"
    assert any(r["ruling_id"] == rid for r in out["unreviewed"]["rulings"])
    assert out["marked"]["ruling"]["reviewed"] is True

    prompt_text = out["curar_prompt"].messages[0].content.text
    assert "data/raw/judge_chats/2026-11-forum.md" in prompt_text
    assert "tier_0" in prompt_text
    assert "add_ruling" in prompt_text
    assert "superseded_by" in prompt_text

    # Path traversal is rejected by build_curar_prompt.
    with pytest.raises(ValueError, match="relative path"):
        build_curar_prompt("../secrets.txt")
    with pytest.raises(ValueError, match="relative path"):
        build_curar_prompt("/etc/passwd")
