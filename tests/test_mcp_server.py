"""Offline tests for MCP tools, `juiz` prompt, JSON contracts, and stdio cleanliness."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from optcg_knowledge.db import connect
from optcg_knowledge.ingest import HashEmbedder
from optcg_knowledge.mcp import create_server


def _extract_json_payload(call_result: Any) -> dict[str, Any]:
    assert call_result.is_error is False
    assert call_result.content
    text = call_result.content[0].text
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


def test_mcp_server_registers_all_tools_and_prompts(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, _, embedder = populated_db
    server = create_server(db_path=db_path, embedder=embedder)

    async def _inspect() -> tuple[list[str], list[str]]:
        tools = [t.name for t in await server.list_tools()]
        prompts = [p.name for p in await server.list_prompts()]
        return (tools, prompts)

    tool_names, prompt_names = asyncio.run(_inspect())
    assert set(tool_names) == {
        "search",
        "get_rule",
        "get_card",
        "search_cards",
        "get_card_rulings",
        "add_ruling",
        "mark_ruling",
        "list_unreviewed_rulings",
        "export_rulings",
    }
    assert set(prompt_names) == {"juiz", "curar"}

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
    assert "potentially_outdated" in prompt_text
    assert "optcg-engine" in prompt_text
    assert "OP05-119" in prompt_text


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
