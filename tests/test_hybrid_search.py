"""Offline golden tests for OPTCG query parsing, FTS5 safety, and hybrid search."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from optcg_knowledge.ingest import HashEmbedder
from optcg_knowledge.search import build_fts_queries, detect_query_tokens, hybrid_search


def test_detect_query_tokens_extracts_card_ids_rule_ids_and_keywords() -> None:
    tokens = detect_query_tokens(
        "Como op05-119 e st10-001 (ou p-001) interagem com [Blocker], [On K.O.] e [DON!! x1] "
        "na regra 7-1-4-1 e 10-1?"
    )
    assert tokens.card_ids == ("OP05-119", "ST10-001", "P-001")
    assert tokens.rule_ids == ("7-1-4-1", "10-1")
    assert tokens.keywords == ("[Blocker]", "[On K.O.]", "[DON!! x1]")


def test_detect_query_tokens_does_not_confuse_card_ids_with_rule_ids() -> None:
    tokens = detect_query_tokens("EB01-001 PRB01-002 OP05-119 P-001")
    assert tokens.card_ids == ("EB01-001", "PRB01-002", "OP05-119", "P-001")
    assert tokens.rule_ids == ()


@pytest.mark.parametrize(
    "weird_query",
    [
        "",
        "   ",
        "!!! ??? --- :::",
        '"""',
        "[] [   ]",
        "NOT OR AND NEAR",
        "[DON!! x1] [On K.O.] [Activate: Main] OP05-119 7-1-4-1",
        'card:"OP05-119" -blocker (rush*)',
    ],
)
def test_fts5_sanitization_never_crashes_on_special_characters(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    weird_query: str,
) -> None:
    _, conn, embedder = populated_db
    strict_q, relaxed_q = build_fts_queries(weird_query)
    if weird_query.strip() and any(ch.isalnum() for ch in weird_query):
        assert strict_q is not None
    else:
        assert strict_q is None
        assert relaxed_q is None

    # Must never raise sqlite3.OperationalError regardless of input characters.
    response = hybrid_search(conn, weird_query, embedder=embedder, k=5)
    assert isinstance(response["results"], list)


def test_golden_exact_card_id_ranks_card_first_and_surfaces_rulings(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db
    res = hybrid_search(conn, "OP05-119", embedder=embedder, k=6)

    assert res["search_mode"] == "hybrid"
    assert res["detected_tokens"]["card_ids"] == ["OP05-119"]
    assert res["results"][0]["doc_type"] == "card"
    assert res["results"][0]["doc_id"] == "OP05-119"
    assert "exact_card_id" in res["results"][0]["match_signals"]

    # Linked active rulings should also be surfaced, while superseded ruling stays hidden.
    ruling_ids = [r["doc_id"] for r in res["results"] if r["doc_type"] == "ruling"]
    assert "faq:op05:119-1" in ruling_ids
    assert "judge:2025-09-10:001" in ruling_ids
    assert "judge:2023-12-01:old" not in ruling_ids

    # ST10-001 and promo P-001 also rank #1 for exact ID lookups.
    for card_id in ("ST10-001", "P-001"):
        card_res = hybrid_search(conn, card_id, embedder=embedder, k=3)
        assert card_res["results"][0]["doc_id"] == card_id


def test_golden_exact_rule_id_and_prefix_ranking(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db

    exact = hybrid_search(conn, "7-1-4-1", embedder=embedder, k=5)
    assert exact["detected_tokens"]["rule_ids"] == ["7-1-4-1"]
    assert exact["results"][0]["doc_type"] == "rule"
    assert exact["results"][0]["doc_id"] == "7-1-4-1"
    assert "exact_rule_id" in exact["results"][0]["match_signals"]

    prefix = hybrid_search(conn, "7-1-4", embedder=embedder, k=6)
    top_ids = [r["doc_id"] for r in prefix["results"]]
    assert top_ids[0] == "7-1-4"
    assert "7-1-4-1" in top_ids
    assert "7-1-4-1-1" in top_ids


def test_golden_bracketed_keywords_surface_matching_chunks(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db

    blocker = hybrid_search(conn, "[Blocker]", embedder=embedder, k=6)
    assert blocker["detected_tokens"]["keywords"] == ["[Blocker]"]
    assert blocker["results"]
    assert all("exact_keyword" in r["match_signals"] for r in blocker["results"][:3])
    assert any(r["doc_id"] == "10-1-1" for r in blocker["results"])

    on_ko = hybrid_search(conn, "[On K.O.]", embedder=embedder, k=5)
    assert on_ko["detected_tokens"]["keywords"] == ["[On K.O.]"]
    assert any(r["doc_id"] == "OP02-106" for r in on_ko["results"])

    don_x1 = hybrid_search(conn, "[DON!! x1]", embedder=embedder, k=5)
    assert don_x1["detected_tokens"]["keywords"] == ["[DON!! x1]"]
    assert don_x1["results"][0]["doc_id"] == "OP01-001"


def test_authority_tie_breaking_and_min_authority_filter(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db

    # Identical text exists at authority=2 (official_faq) and authority=4 (community).
    tied = hybrid_search(
        conn,
        "[On K.O.] timing resolve trashed by battle",
        embedder=embedder,
        doc_types=["ruling"],
        k=5,
    )
    ids = [r["doc_id"] for r in tied["results"]]
    assert ids.index("faq:on-ko:tie") < ids.index("community:on-ko:tie")

    # min_authority=2 means authority <= 2 (rules + official_faq/errata only).
    official_only = hybrid_search(
        conn,
        "OP05-119 extra turn",
        embedder=embedder,
        min_authority=2,
        k=8,
    )
    assert official_only["results"]
    assert all(r["authority"] <= 2 for r in official_only["results"])
    assert all(r["doc_id"] != "judge:2025-09-10:001" for r in official_only["results"])


def test_superseded_rulings_hidden_by_default_and_included_when_requested(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db

    default_res = hybrid_search(
        conn,
        "OP05-119 outdated claiming required",
        embedder=embedder,
        doc_types=["ruling"],
        k=10,
    )
    assert all(r["doc_id"] != "judge:2023-12-01:old" for r in default_res["results"])

    with_superseded = hybrid_search(
        conn,
        "OP05-119 outdated claiming required",
        embedder=embedder,
        doc_types=["ruling"],
        include_superseded=True,
        k=10,
    )
    superseded_hits = [
        r for r in with_superseded["results"] if r["doc_id"] == "judge:2023-12-01:old"
    ]
    assert len(superseded_hits) == 1
    assert superseded_hits[0]["superseded_by"] == "faq:op05:119-1"
    assert superseded_hits[0]["potentially_outdated"] is True


def test_lexical_fallback_when_no_embedder_supplied(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, _ = populated_db
    res = hybrid_search(conn, "OP05-119 extra turn", embedder=None, k=5)
    assert res["search_mode"] == "lexical_only"
    assert res["results"][0]["doc_id"] == "OP05-119"
    assert "vector" not in res["results"][0]["match_signals"]
