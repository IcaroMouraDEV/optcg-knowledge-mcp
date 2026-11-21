"""Offline tests for MCP tools, `juiz` prompt, JSON contracts, and stdio cleanliness."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from optcg_knowledge.curation import add_ruling, mark_ruling
from optcg_knowledge.db import connect
from optcg_knowledge.ingest import (
    HashEmbedder,
    embed_chunks,
    ingest_banlist,
    ingest_trm,
    parse_banlist_json,
    parse_trm_text,
)
from optcg_knowledge.mcp import create_server
from optcg_knowledge.search.queries import MAX_BATCH_CARDS_PAGE_SIZE, get_cards, search_cards


def _extract_json_payload(call_result: Any) -> dict[str, Any]:
    assert call_result.is_error is False
    assert call_result.content
    text = call_result.content[0].text
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


def _extract_resource_json(read_result: Any) -> dict[str, Any]:
    items = list(read_result)
    assert len(items) == 1
    item = items[0]
    assert item.mime_type == "application/json"
    assert isinstance(item.content, str)
    parsed = json.loads(item.content)
    assert isinstance(parsed, dict)
    return parsed


def test_mcp_server_registers_all_tools_and_prompts(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, _, embedder = populated_db
    server = create_server(db_path=db_path, embedder=embedder)

    async def _inspect() -> tuple[list[str], list[str], list[str], list[str]]:
        tools = [t.name for t in await server.list_tools()]
        prompts = [p.name for p in await server.list_prompts()]
        resources = [str(r.uri) for r in await server.list_resources()]
        templates = [t.uri_template for t in await server.list_resource_templates()]
        return (tools, prompts, resources, templates)

    tool_names, prompt_names, resource_uris, template_uris = asyncio.run(_inspect())
    assert set(tool_names) == {
        "search",
        "get_rule",
        "get_trm_section",
        "get_card",
        "get_cards",
        "search_cards",
        "get_banlist",
        "get_card_rulings",
        "get_rule_rulings",
        "add_ruling",
        "mark_ruling",
        "list_unreviewed_rulings",
        "get_status",
        "export_rulings",
    }
    assert set(prompt_names) == {"juiz", "curar"}
    assert set(resource_uris) == {"optcg://status", "optcg://banlist"}
    assert set(template_uris) == {
        "optcg://rules/{rule_id}",
        "optcg://trm/{section_id}",
        "optcg://cards/{card_id}",
    }

    # Strict stdio protocol integrity: nothing written to stdout.
    assert capsys.readouterr().out == ""


def test_mcp_query_tools_contracts_and_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, _, embedder = populated_db
    server = create_server(db_path=db_path, embedder=embedder)

    async def _exercise() -> dict[str, Any]:
        search_res = _extract_json_payload(
            await server.call_tool("search", {"query": "OP05-119 [Blocker]", "k": 5})
        )
        rule_res = _extract_json_payload(
            await server.call_tool("get_rule", {"rule_id": "7-1-4", "include_children": True})
        )
        rule_no_children = _extract_json_payload(
            await server.call_tool("get_rule", {"rule_id": "7-1-4", "include_children": False})
        )
        missing_rule = _extract_json_payload(
            await server.call_tool("get_rule", {"rule_id": "99-9-9"})
        )
        card_by_id = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "op05-119"})
        )
        card_by_name = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "Monkey D Luffy"})
        )
        card_fuzzy = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "Zoro"})
        )
        card_missing = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "OP99-999"})
        )
        cards_filtered = _extract_json_payload(
            await server.call_tool(
                "search_cards",
                {"color": "Purple", "cost": 10, "trait": "Straw Hat", "set_code": "OP05"},
            )
        )
        cards_by_keyword = _extract_json_payload(
            await server.call_tool("search_cards", {"keyword": "Blocker"})
        )
        rulings_default = _extract_json_payload(
            await server.call_tool("get_card_rulings", {"card_id": "op05-119"})
        )
        rulings_all = _extract_json_payload(
            await server.call_tool(
                "get_card_rulings", {"card_id": "OP05-119", "include_superseded": True}
            )
        )
        prompt_result = await server.get_prompt(
            "juiz", {"question": "Does OP05-119 work with 0 other Characters?"}
        )
        return {
            "search": search_res,
            "rule": rule_res,
            "rule_no_children": rule_no_children,
            "missing_rule": missing_rule,
            "card_by_id": card_by_id,
            "card_by_name": card_by_name,
            "card_fuzzy": card_fuzzy,
            "card_missing": card_missing,
            "cards_filtered": cards_filtered,
            "cards_by_keyword": cards_by_keyword,
            "rulings_default": rulings_default,
            "rulings_all": rulings_all,
            "prompt": prompt_result,
        }

    out = asyncio.run(_exercise())

    # Zero stdout leakage across all tool and prompt invocations.
    assert capsys.readouterr().out == ""

    # 1. search contract
    assert out["search"]["search_mode"] == "hybrid"
    assert out["search"]["rules_version"] == "1.24"
    assert out["search"]["detected_tokens"]["card_ids"] == ["OP05-119"]
    assert out["search"]["results"][0]["doc_id"] == "OP05-119"

    # 2. get_rule contract
    assert out["rule"]["found"] is True
    assert out["rule"]["rule"]["rule_id"] == "7-1-4"
    assert [a["rule_id"] for a in out["rule"]["ancestors"]] == ["7", "7-1"]
    assert [c["rule_id"] for c in out["rule"]["children"]] == [
        "7-1-4-1",
        "7-1-4-1-1",
        "7-1-4-1-1-1",
    ]
    assert out["rule_no_children"]["children"] == []
    assert out["missing_rule"]["found"] is False
    assert out["missing_rule"]["rule"] is None

    # 3. get_card contract
    assert out["card_by_id"]["match_type"] == "exact_id"
    assert out["card_by_id"]["total"] == 1
    assert out["card_by_id"]["cards"][0]["card_id"] == "OP05-119"
    assert out["card_by_id"]["cards"][0]["block_icon"] == 3
    assert out["card_by_id"]["cards"][0]["variant_count"] == 2
    assert "raw_json" not in out["card_by_id"]["cards"][0]

    assert out["card_by_name"]["match_type"] == "exact_name"
    assert {c["card_id"] for c in out["card_by_name"]["cards"]} == {"OP05-119", "P-001"}

    assert out["card_fuzzy"]["match_type"] == "fuzzy_name"
    assert out["card_fuzzy"]["cards"][0]["card_id"] == "OP01-001"

    assert out["card_missing"]["match_type"] == "not_found"
    assert out["card_missing"]["cards"] == []

    # 4. search_cards contract
    assert out["cards_filtered"]["total_matched"] == 1
    assert out["cards_filtered"]["cards"][0]["card_id"] == "OP05-119"
    assert out["cards_by_keyword"]["cards"][0]["card_id"] == "OP04-083"

    # 5. get_card_rulings contract
    assert out["rulings_default"]["card_id"] == "OP05-119"
    assert out["rulings_default"]["card_found"] is True
    assert out["rulings_default"]["current_rules_version"] == "1.24"
    default_rulings = out["rulings_default"]["rulings"]
    assert [r["ruling_id"] for r in default_rulings] == [
        "faq:op05:119-1",
        "judge:2025-09-10:001",
    ]
    assert default_rulings[0]["authority"] == 2
    assert default_rulings[0]["potentially_outdated"] is True
    assert default_rulings[1]["authority"] == 3
    assert default_rulings[1]["potentially_outdated"] is False

    assert len(out["rulings_all"]["rulings"]) == 3
    assert any(r["superseded_by"] == "faq:op05:119-1" for r in out["rulings_all"]["rulings"])

    # 6. juiz prompt contract
    prompt_text = out["prompt"].messages[0].content.text
    assert "authority = 1" in prompt_text
    assert "get_card_rulings" in prompt_text
    assert "get_rule_rulings" in prompt_text
    assert "get_cards" in prompt_text
    assert "get_trm_section" in prompt_text
    assert "get_banlist" in prompt_text
    assert "tournament_rules" in prompt_text
    assert "ban_status" in prompt_text
    assert "block_icon" in prompt_text
    assert "min_block" in prompt_text
    assert "paired_card_ids" in prompt_text
    assert "potentially_outdated" in prompt_text
    assert "optcg-engine" in prompt_text
    assert "OP05-119" in prompt_text


def test_mcp_banlist_legality_in_get_card_and_search_cards_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db
    doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "upcoming_effective_date": "2026-11-27",
            "restrictions": [
                {"card_id": "OP01-001", "ban_status": "banned"},
                {"card_id": "OP04-083", "ban_status": "restricted"},
                {
                    "card_id": "OP05-119",
                    "ban_status": "pair_banned",
                    "paired_card_ids": ["P-001"],
                },
            ],
            "upcoming_banned_pairs": [["OP02-106", "ST10-001"]],
        }
    )
    ingest_banlist(conn, doc)

    server = create_server(db_path=db_path, embedder=embedder)
    assert "banlist" in (server.instructions or "").lower()
    assert "get_banlist" in (server.instructions or "")

    async def _run() -> dict[str, Any]:
        zoro = _extract_json_payload(await server.call_tool("get_card", {"id_or_name": "OP01-001"}))
        luffy = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "OP05-119"})
        )
        tsuru = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "OP02-106"})
        )
        full_banlist = _extract_json_payload(await server.call_tool("get_banlist", {}))
        filtered_banlist = _extract_json_payload(
            await server.call_tool("get_banlist", {"ban_status": "pair_banned"})
        )
        banned_list = _extract_json_payload(
            await server.call_tool("search_cards", {"ban_status": "banned"})
        )
        pair_list = _extract_json_payload(
            await server.call_tool("search_cards", {"ban_status": "pair_banned"})
        )
        restricted_list = _extract_json_payload(
            await server.call_tool("search_cards", {"ban_status": "restricted"})
        )
        legal_list = _extract_json_payload(
            await server.call_tool("search_cards", {"ban_status": "legal"})
        )
        return {
            "zoro": zoro,
            "luffy": luffy,
            "tsuru": tsuru,
            "full_banlist": full_banlist,
            "filtered_banlist": filtered_banlist,
            "banned_list": banned_list,
            "pair_list": pair_list,
            "restricted_list": restricted_list,
            "legal_list": legal_list,
        }

    res = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    assert res["zoro"]["banlist_loaded"] is True
    assert res["zoro"]["banlist_effective_date"] == "2026-10-12"
    assert res["zoro"]["banlist_upcoming_effective_date"] == "2026-11-27"
    assert res["zoro"]["cards"][0]["ban_status"] == "banned"
    assert res["zoro"]["cards"][0]["max_copies"] == 0
    assert res["zoro"]["cards"][0]["effective_date"] == "2026-10-12"
    assert res["zoro"]["cards"][0]["upcoming_restriction"] is None

    assert res["luffy"]["cards"][0]["ban_status"] == "pair_banned"
    assert res["luffy"]["cards"][0]["paired_card_ids"] == ["P-001"]

    # OP02-106 is currently legal today, with a scheduled upcoming pair ban.
    assert res["tsuru"]["cards"][0]["ban_status"] == "legal"
    assert res["tsuru"]["cards"][0]["max_copies"] == 4
    assert res["tsuru"]["cards"][0]["upcoming_restriction"]["ban_status"] == "pair_banned"
    assert res["tsuru"]["cards"][0]["upcoming_restriction"]["paired_card_ids"] == ["ST10-001"]
    assert res["tsuru"]["cards"][0]["upcoming_restriction"]["effective_date"] == "2026-11-27"

    assert res["full_banlist"]["banlist_loaded"] is True
    assert res["full_banlist"]["banlist_effective_date"] == "2026-10-12"
    assert res["full_banlist"]["banlist_upcoming_effective_date"] == "2026-11-27"
    assert res["full_banlist"]["counts"] == {
        "banned": 1,
        "restricted": 1,
        "pair_banned": 2,
        "total": 4,
    }
    assert res["full_banlist"]["banned_pairs"] == [["OP05-119", "P-001"]]
    assert res["full_banlist"]["upcoming"]["effective_date"] == "2026-11-27"
    assert res["full_banlist"]["upcoming"]["counts"] == {
        "banned": 0,
        "restricted": 0,
        "pair_banned": 2,
        "total": 2,
    }
    assert res["full_banlist"]["upcoming"]["banned_pairs"] == [["OP02-106", "ST10-001"]]

    assert [r["card_id"] for r in res["filtered_banlist"]["restrictions"]] == [
        "OP05-119",
        "P-001",
    ]
    assert [r["card_id"] for r in res["filtered_banlist"]["upcoming"]["restrictions"]] == [
        "OP02-106",
        "ST10-001",
    ]

    assert [c["card_id"] for c in res["banned_list"]["cards"]] == ["OP01-001"]
    assert res["banned_list"]["filters"]["ban_status"] == "banned"
    assert [c["card_id"] for c in res["restricted_list"]["cards"]] == ["OP04-083"]
    assert [c["card_id"] for c in res["pair_list"]["cards"]] == ["OP05-119", "P-001"]
    assert [c["card_id"] for c in res["legal_list"]["cards"]] == [
        "OP02-106",
        "ST01-014",
        "ST10-001",
    ]


def test_mcp_trm_tool_and_search_filtering_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db
    trm_doc = parse_trm_text(
        """\
Last Updated: October 17, 2025
7.0 Tournament Penalties
Judges maintain event fairness.
7.4 Examples of Infractions
Infractions and remedies.
7.4.1 Game Play Error
Accidental rules violations.
1. Failing to Observe Game Rules
Failure to follow a game rule.
Penalty: Warning
Resolution: Reverse the game state if possible.
2. Forgetting to Resolve Automatic Actions
Missed mandatory trigger.
Penalty: Warning
Resolution: Carry out the mandatory effect from the closest checkpoint.
"""
    )
    ingest_trm(conn, trm_doc)
    embed_chunks(conn, embedder)

    server = create_server(db_path=db_path, embedder=embedder)

    async def _run() -> dict[str, Any]:
        trm_section = _extract_json_payload(
            await server.call_tool(
                "get_trm_section",
                {"section_id": "TRM-7.4.1", "include_children": True},
            )
        )
        trm_missing = _extract_json_payload(
            await server.call_tool("get_trm_section", {"section_id": "TRM-99.9"})
        )
        trm_search = _extract_json_payload(
            await server.call_tool(
                "search",
                {
                    "query": "missed mandatory trigger warning",
                    "doc_types": ["tournament_rule"],
                    "k": 5,
                },
            )
        )
        return {
            "trm_section": trm_section,
            "trm_missing": trm_missing,
            "trm_search": trm_search,
        }

    res = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    assert res["trm_section"]["found"] is True
    assert res["trm_section"]["section_id"] == "TRM-7.4.1"
    assert res["trm_section"]["source"] == "tournament_rules"
    assert [a["section_id"] for a in res["trm_section"]["ancestors"]] == ["TRM-7.0", "TRM-7.4"]
    assert [c["section_id"] for c in res["trm_section"]["children"]] == [
        "TRM-7.4.1.1",
        "TRM-7.4.1.2",
    ]
    assert res["trm_missing"]["found"] is False
    assert res["trm_search"]["results"][0]["doc_id"] == "TRM-7.4.1.2"
    assert res["trm_search"]["results"][0]["source"] == "tournament_rules"
    assert all(r["doc_type"] == "tournament_rule" for r in res["trm_search"]["results"])


def test_mcp_server_degrades_gracefully_without_vectors_or_embedder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "empty_optcg.db"
    # Fresh DB path with no vectors -> must initialize safely and run lexical_only without torch.
    server = create_server(db_path=db_path, search_mode="auto")

    async def _run() -> dict[str, Any]:
        return _extract_json_payload(await server.call_tool("search", {"query": "Blocker"}))

    res = asyncio.run(_run())
    assert res["search_mode"] == "lexical_only"
    assert res["total"] == 0
    assert capsys.readouterr().out == ""

    # Confirm DB schema exists and was cleanly initialized.
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks'"
        ).fetchone()
        assert row is not None


def test_mcp_get_status_contract_and_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db
    doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "restrictions": [
                {"card_id": "OP01-001", "ban_status": "banned"},
            ],
        }
    )
    ingest_banlist(conn, doc)

    server = create_server(db_path=db_path, embedder=embedder)
    assert "get_status" in (server.instructions or "")

    async def _run() -> dict[str, Any]:
        return _extract_json_payload(await server.call_tool("get_status", {}))

    status = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    assert status["schema_version"] == "6"
    assert status["rules_version"] == "1.24"
    assert status["banlist_loaded"] is True
    assert status["banlist_effective_date"] == "2026-10-12"
    assert status["banlist_upcoming_effective_date"] is None
    assert status["embedding_model"] == embedder.model_name
    assert status["embedding_dim"] == embedder.dimension

    assert status["counts"]["rules"] > 0
    assert status["counts"]["cards"] > 0
    assert status["counts"]["card_restrictions"] == {
        "total": 1,
        "banned": 1,
        "restricted": 0,
        "pair_banned": 0,
        "upcoming": {
            "total": 0,
            "banned": 0,
            "restricted": 0,
            "pair_banned": 0,
        },
    }
    assert status["counts"]["rulings"]["total"] == 5
    assert status["counts"]["rulings"]["superseded"] == 1
    assert status["counts"]["rulings"]["active"] == 4
    assert status["counts"]["rulings"]["unreviewed"] == 0
    assert status["counts"]["rulings"]["by_source"]["official_faq"] == 2
    assert status["counts"]["rulings"]["by_source"]["judge"] == 2
    assert status["counts"]["rulings"]["by_source"]["community"] == 1
    assert status["counts"]["rulings"]["judge_by_tier"]["unspecified"] == 2

    assert status["embeddings"]["total_chunks"] == status["counts"]["chunks"]["total"]
    assert status["embeddings"]["embedded_chunks"] == status["embeddings"]["total_chunks"]
    assert status["embeddings"]["pending_chunks"] == 0
    assert status["embeddings"]["coverage_pct"] == 100.0
    assert status["embeddings"]["ready"] is True

    assert status["timestamps"]["last_ingested_at"] is not None
    assert status["timestamps"]["last_embed_at"] is not None
    assert status["timestamps"]["banlist_last_ingested_at"] is not None


def test_mcp_get_rule_rulings_ordering_children_and_superseded(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db

    # Add judge rulings citing parent rule 7-1-4 and child rule 7-1-4-1 across judge tiers.
    t1_res = add_ruling(
        conn,
        question="Does [Blocker] require resting the Character under rule 7-1-4-1?",
        answer="Yes, activating [Blocker] requires resting the Character.",
        source="judge",
        date="2026-01-10",
        card_ids=["OP04-083"],
        rule_refs=["7-1-4-1"],
        judge_tier="tier_1",
    )
    t0_res = add_ruling(
        conn,
        question="Can [Blocker] be activated if the Character is already rested?",
        answer="No. Under 7-1-4 and 7-1-4-1, only an active Character with [Blocker] can rest.",
        source="judge",
        date="2026-01-05",
        card_ids=["OP04-083"],
        rule_refs=["7-1-4", "7-1-4-1"],
        judge_tier="tier_0",
    )
    old_res = add_ruling(
        conn,
        question="Superseded blocker question on 7-1-4-1.",
        answer="Superseded answer.",
        source="judge",
        date="2025-05-01",
        card_ids=["OP04-083"],
        rule_refs=["7-1-4-1"],
        judge_tier="tier_2",
    )
    mark_ruling(
        conn,
        old_res["ruling"]["ruling_id"],
        superseded_by=t0_res["ruling"]["ruling_id"],
    )

    server = create_server(db_path=db_path, embedder=embedder)
    assert "get_rule_rulings" in (server.instructions or "")

    async def _run() -> dict[str, Any]:
        rule_132 = _extract_json_payload(
            await server.call_tool("get_rule_rulings", {"rule_id": "1-3-2"})
        )
        parent_exact = _extract_json_payload(
            await server.call_tool(
                "get_rule_rulings",
                {"rule_id": "7-1-4", "include_children": False},
            )
        )
        parent_with_children = _extract_json_payload(
            await server.call_tool(
                "get_rule_rulings",
                {"rule_id": "7-1-4", "include_children": True},
            )
        )
        child_with_superseded = _extract_json_payload(
            await server.call_tool(
                "get_rule_rulings",
                {"rule_id": "7-1-4-1", "include_superseded": True},
            )
        )
        missing_rule = _extract_json_payload(
            await server.call_tool("get_rule_rulings", {"rule_id": "99-9-9"})
        )
        return {
            "rule_132": rule_132,
            "parent_exact": parent_exact,
            "parent_with_children": parent_with_children,
            "child_with_superseded": child_with_superseded,
            "missing_rule": missing_rule,
        }

    out = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    # 1-3-2 has 2 active rulings in populated_db: authority 2 (official_faq) before authority 3
    assert out["rule_132"]["rule_id"] == "1-3-2"
    assert out["rule_132"]["rule_found"] is True
    assert out["rule_132"]["include_children"] is False
    assert out["rule_132"]["matched_rule_ids"] == ["1-3-2"]
    assert [r["ruling_id"] for r in out["rule_132"]["rulings"]] == [
        "faq:op05:119-1",
        "judge:2025-09-10:001",
    ]

    # Exact match on 7-1-4 returns only the ruling explicitly citing 7-1-4
    assert out["parent_exact"]["rule_found"] is True
    assert out["parent_exact"]["matched_rule_ids"] == ["7-1-4"]
    assert [r["ruling_id"] for r in out["parent_exact"]["rulings"]] == [
        t0_res["ruling"]["ruling_id"]
    ]

    # include_children=True on 7-1-4 includes 7-1-4-1, deduplicates t0_res, orders tier_0 > tier_1
    assert out["parent_with_children"]["include_children"] is True
    assert out["parent_with_children"]["matched_rule_ids"] == ["7-1-4", "7-1-4-1"]
    assert [r["ruling_id"] for r in out["parent_with_children"]["rulings"]] == [
        t0_res["ruling"]["ruling_id"],
        t1_res["ruling"]["ruling_id"],
    ]

    # include_superseded=True includes superseded tier_2 ruling in tier_0 > tier_2 > tier_1 order
    assert [r["ruling_id"] for r in out["child_with_superseded"]["rulings"]] == [
        t0_res["ruling"]["ruling_id"],
        old_res["ruling"]["ruling_id"],
        t1_res["ruling"]["ruling_id"],
    ]

    assert out["missing_rule"]["rule_found"] is False
    assert out["missing_rule"]["total"] == 0
    assert out["missing_rule"]["rulings"] == []


def test_mcp_get_cards_batch_pagination_banlist_rulings_and_validation(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db
    doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "upcoming_effective_date": "2026-11-27",
            "restrictions": [
                {"card_id": "OP04-083", "ban_status": "restricted"},
            ],
            "upcoming_banned_pairs": [["OP05-119", "ST10-001"]],
        }
    )
    ingest_banlist(conn, doc)

    server = create_server(db_path=db_path, embedder=embedder)
    assert "get_cards" in (server.instructions or "")

    requested = ["op05-119", "OP04-083", "OP05-119", "OP99-999", "st10-001"]

    async def _run() -> dict[str, Any]:
        page1 = _extract_json_payload(
            await server.call_tool(
                "get_cards",
                {
                    "card_ids": requested,
                    "include_rulings": True,
                    "include_superseded": False,
                    "page": 1,
                    "page_size": 2,
                },
            )
        )
        page2 = _extract_json_payload(
            await server.call_tool(
                "get_cards",
                {
                    "card_ids": requested,
                    "include_rulings": False,
                    "page": 2,
                    "page_size": 2,
                },
            )
        )
        page3_empty = _extract_json_payload(
            await server.call_tool(
                "get_cards",
                {
                    "card_ids": requested,
                    "page": 3,
                    "page_size": 2,
                },
            )
        )
        with_superseded = _extract_json_payload(
            await server.call_tool(
                "get_cards",
                {
                    "card_ids": ["OP05-119"],
                    "include_rulings": True,
                    "include_superseded": True,
                },
            )
        )
        return {
            "page1": page1,
            "page2": page2,
            "page3_empty": page3_empty,
            "with_superseded": with_superseded,
        }

    out = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    # Deduplicated total_requested is 4: ["OP05-119", "OP04-083", "OP99-999", "ST10-001"]
    p1 = out["page1"]
    assert p1["total_requested"] == 4
    assert p1["requested_ids"] == ["OP05-119", "OP04-083"]
    assert p1["page"] == 1
    assert p1["page_size"] == 2
    assert p1["total_pages"] == 2
    assert p1["has_more"] is True
    assert p1["next_page"] == 2
    assert p1["found"] == 2
    assert p1["not_found"] == []
    assert p1["banlist_loaded"] is True
    assert p1["banlist_effective_date"] == "2026-10-12"
    assert p1["banlist_upcoming_effective_date"] == "2026-11-27"
    assert [c["card_id"] for c in p1["cards"]] == ["OP05-119", "OP04-083"]

    # OP05-119 is legal today with an upcoming pair ban, and has 2 active rulings
    luffy_card = p1["cards"][0]
    assert luffy_card["ban_status"] == "legal"
    assert luffy_card["upcoming_restriction"]["ban_status"] == "pair_banned"
    assert luffy_card["upcoming_restriction"]["paired_card_ids"] == ["ST10-001"]
    assert [r["ruling_id"] for r in luffy_card["rulings"]] == [
        "faq:op05:119-1",
        "judge:2025-09-10:001",
    ]

    # OP04-083 is actively restricted today and has 0 rulings
    sabo_card = p1["cards"][1]
    assert sabo_card["ban_status"] == "restricted"
    assert sabo_card["max_copies"] == 1
    assert sabo_card["rulings"] == []

    # Page 2 contains ["OP99-999", "ST10-001"] with include_rulings=False
    p2 = out["page2"]
    assert p2["requested_ids"] == ["OP99-999", "ST10-001"]
    assert p2["page"] == 2
    assert p2["total_pages"] == 2
    assert p2["has_more"] is False
    assert p2["next_page"] is None
    assert p2["found"] == 1
    assert p2["not_found"] == ["OP99-999"]
    assert [c["card_id"] for c in p2["cards"]] == ["ST10-001"]
    assert "rulings" not in p2["cards"][0]

    # Out-of-range page 3 returns empty slice cleanly
    p3 = out["page3_empty"]
    assert p3["requested_ids"] == []
    assert p3["found"] == 0
    assert p3["has_more"] is False
    assert p3["next_page"] is None
    assert p3["cards"] == []

    # include_superseded=True includes the 3rd superseded ruling on OP05-119
    assert len(out["with_superseded"]["cards"][0]["rulings"]) == 3

    # Direct query-layer validation & page_size cap tests
    capped = get_cards(conn, ["OP05-119"], page_size=999)
    assert capped["page_size"] == MAX_BATCH_CARDS_PAGE_SIZE

    with pytest.raises(ValueError, match="at least one non-empty card_id"):
        get_cards(conn, ["  ", ""])
    with pytest.raises(ValueError, match="not a string"):
        get_cards(conn, "OP05-119")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="page must be a positive integer"):
        get_cards(conn, ["OP05-119"], page=0)
    with pytest.raises(ValueError, match="page_size must be a positive integer"):
        get_cards(conn, ["OP05-119"], page_size=0)


def test_mcp_resources_list_and_read_contracts_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db
    trm_doc = parse_trm_text(
        """\
Last Updated: October 17, 2025
7.0 Tournament Penalties
Judges maintain event fairness.
7.4 Examples of Infractions
Infractions and remedies.
7.4.1 Game Play Error
Accidental rules violations.
1. Failing to Observe Game Rules
Failure to follow a game rule.
Penalty: Warning
Resolution: Reverse the game state if possible.
2. Forgetting to Resolve Automatic Actions
Missed mandatory trigger.
Penalty: Warning
Resolution: Carry out the mandatory effect from the closest checkpoint.
"""
    )
    ingest_trm(conn, trm_doc)

    banlist_doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "upcoming_effective_date": "2026-11-27",
            "restrictions": [
                {"card_id": "OP04-083", "ban_status": "restricted"},
            ],
            "upcoming_banned_pairs": [["OP05-119", "ST10-001"]],
        }
    )
    ingest_banlist(conn, banlist_doc)

    server = create_server(db_path=db_path, embedder=embedder)
    instructions = server.instructions or ""
    assert "optcg://status" in instructions
    assert "optcg://banlist" in instructions
    assert "optcg://rules/{rule_id}" in instructions
    assert "optcg://trm/{section_id}" in instructions
    assert "optcg://cards/{card_id}" in instructions

    async def _run() -> dict[str, Any]:
        static_resources = await server.list_resources()
        resource_templates = await server.list_resource_templates()

        res_status = _extract_resource_json(await server.read_resource("optcg://status"))
        tool_status = _extract_json_payload(await server.call_tool("get_status", {}))

        res_banlist = _extract_resource_json(await server.read_resource("optcg://banlist"))
        tool_banlist = _extract_json_payload(await server.call_tool("get_banlist", {}))

        res_rule = _extract_resource_json(await server.read_resource("optcg://rules/7-1-4"))
        tool_rule = _extract_json_payload(
            await server.call_tool("get_rule", {"rule_id": "7-1-4", "include_children": True})
        )
        res_rule_missing = _extract_resource_json(
            await server.read_resource("optcg://rules/99-9-9")
        )

        res_trm = _extract_resource_json(await server.read_resource("optcg://trm/TRM-7.4.1"))
        tool_trm = _extract_json_payload(
            await server.call_tool(
                "get_trm_section",
                {"section_id": "TRM-7.4.1", "include_children": True},
            )
        )
        res_trm_missing = _extract_resource_json(
            await server.read_resource("optcg://trm/TRM-99.9")
        )

        res_card = _extract_resource_json(await server.read_resource("optcg://cards/OP05-119"))
        tool_card = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "OP05-119"})
        )
        res_card_missing = _extract_resource_json(
            await server.read_resource("optcg://cards/OP99-999")
        )

        return {
            "static_resources": static_resources,
            "resource_templates": resource_templates,
            "res_status": res_status,
            "tool_status": tool_status,
            "res_banlist": res_banlist,
            "tool_banlist": tool_banlist,
            "res_rule": res_rule,
            "tool_rule": tool_rule,
            "res_rule_missing": res_rule_missing,
            "res_trm": res_trm,
            "tool_trm": tool_trm,
            "res_trm_missing": res_trm_missing,
            "res_card": res_card,
            "tool_card": tool_card,
            "res_card_missing": res_card_missing,
        }

    out = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    static_by_uri = {str(r.uri): r for r in out["static_resources"]}
    assert set(static_by_uri) == {"optcg://status", "optcg://banlist"}
    assert all(r.mime_type == "application/json" for r in static_by_uri.values())

    templates_by_uri = {t.uri_template: t for t in out["resource_templates"]}
    assert set(templates_by_uri) == {
        "optcg://rules/{rule_id}",
        "optcg://trm/{section_id}",
        "optcg://cards/{card_id}",
    }
    assert all(t.mime_type == "application/json" for t in templates_by_uri.values())

    # Exact payload parity with the delegated query tools
    assert out["res_status"] == out["tool_status"]
    assert out["res_banlist"] == out["tool_banlist"]
    assert out["res_rule"] == out["tool_rule"]
    assert out["res_trm"] == out["tool_trm"]
    assert out["res_card"] == out["tool_card"]

    # Spot-check domain fields and missing-entity behavior
    assert out["res_rule"]["found"] is True
    assert [c["rule_id"] for c in out["res_rule"]["children"]] == [
        "7-1-4-1",
        "7-1-4-1-1",
        "7-1-4-1-1-1",
    ]
    assert out["res_rule_missing"]["found"] is False

    assert out["res_trm"]["found"] is True
    assert [c["section_id"] for c in out["res_trm"]["children"]] == [
        "TRM-7.4.1.1",
        "TRM-7.4.1.2",
    ]
    assert out["res_trm_missing"]["found"] is False

    assert out["res_card"]["match_type"] == "exact_id"
    assert out["res_card"]["cards"][0]["card_id"] == "OP05-119"
    assert out["res_card"]["cards"][0]["block_icon"] == 3
    assert out["res_card"]["cards"][0]["ban_status"] == "legal"
    assert out["res_card"]["cards"][0]["upcoming_restriction"]["ban_status"] == "pair_banned"
    assert out["res_card_missing"]["match_type"] == "not_found"
    assert out["res_card_missing"]["cards"] == []


def test_mcp_block_icon_surfacing_and_search_cards_filtering_zero_stdout_leak(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, embedder = populated_db
    banlist_doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "restrictions": [
                {"card_id": "OP01-001", "ban_status": "banned"},
                {"card_id": "ST10-001", "ban_status": "restricted"},
            ],
        }
    )
    ingest_banlist(conn, banlist_doc)

    server = create_server(db_path=db_path, embedder=embedder)

    async def _run() -> dict[str, Any]:
        card_promo = _extract_json_payload(
            await server.call_tool("get_card", {"id_or_name": "P-001"})
        )
        batch_cards = _extract_json_payload(
            await server.call_tool("get_cards", {"card_ids": ["OP05-119", "ST10-001", "P-001"]})
        )
        banlist = _extract_json_payload(await server.call_tool("get_banlist", {}))
        block_1 = _extract_json_payload(
            await server.call_tool("search_cards", {"block_icon": 1})
        )
        block_2 = _extract_json_payload(
            await server.call_tool("search_cards", {"block_icon": 2})
        )
        min_block_2 = _extract_json_payload(
            await server.call_tool("search_cards", {"min_block": 2})
        )
        min_block_2_legal = _extract_json_payload(
            await server.call_tool("search_cards", {"min_block": 2, "ban_status": "legal"})
        )
        return {
            "card_promo": card_promo,
            "batch_cards": batch_cards,
            "banlist": banlist,
            "block_1": block_1,
            "block_2": block_2,
            "min_block_2": min_block_2,
            "min_block_2_legal": min_block_2_legal,
        }

    out = asyncio.run(_run())
    assert capsys.readouterr().out == ""

    # Nullable block_icon preserved cleanly on get_card and get_cards.
    assert out["card_promo"]["cards"][0]["block_icon"] is None
    assert [(c["card_id"], c["block_icon"]) for c in out["batch_cards"]["cards"]] == [
        ("OP05-119", 3),
        ("ST10-001", 2),
        ("P-001", None),
    ]

    # get_banlist surfaces joined card block_icon alongside restriction metadata.
    banlist_blocks = {r["card_id"]: r["block_icon"] for r in out["banlist"]["restrictions"]}
    assert banlist_blocks == {"OP01-001": 1, "ST10-001": 2}

    # Exact block_icon and Standard rotation min_block filters exclude NULL block_icon cards.
    assert out["block_1"]["filters"]["block_icon"] == 1
    assert out["block_1"]["filters"]["min_block"] is None
    assert [c["card_id"] for c in out["block_1"]["cards"]] == [
        "OP01-001",
        "OP02-106",
        "OP04-083",
        "ST01-014",
    ]
    assert [c["card_id"] for c in out["block_2"]["cards"]] == ["ST10-001"]

    assert out["min_block_2"]["filters"]["min_block"] == 2
    assert [c["card_id"] for c in out["min_block_2"]["cards"]] == ["OP05-119", "ST10-001"]

    # Combining min_block with ban_status isolates Standard-legal unrestricted cards.
    assert [c["card_id"] for c in out["min_block_2_legal"]["cards"]] == ["OP05-119"]

    # Query-layer input validation rejects non-positive or boolean block filters.
    with pytest.raises(ValueError, match="block_icon must be a positive integer"):
        search_cards(conn, block_icon=0)
    with pytest.raises(ValueError, match="block_icon must be a positive integer"):
        search_cards(conn, block_icon=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_block must be a positive integer"):
        search_cards(conn, min_block=0)
    with pytest.raises(ValueError, match="min_block must be a positive integer"):
        search_cards(conn, min_block=False)  # type: ignore[arg-type]
