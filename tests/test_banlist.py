"""Offline tests for Bandai banlist HTML/JSON parsing, ingestion, queries, and CLI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.db import connect, get_meta
from optcg_knowledge.ingest.banlist_parser import (
    ingest_banlist,
    parse_banlist_html,
    parse_banlist_json,
    parse_banlist_path,
)
from optcg_knowledge.ingest.embeddings import HashEmbedder
from optcg_knowledge.search.queries import get_banlist, get_card, search_cards

_REALISTIC_BANDAI_HTML = """\
<!doctype html>
<html lang="en">
<body>
  <header><nav><a href="/rules/">OP99-999 Nav Noise</a></nav></header>
  <article class="articleCol">
    <div class="pageTitCol">
      <h2 class="pageTit">Banned/Restricted Card Addition Notice</h2>
      <span class="pageTitDate">November 17, 2026</span>
    </div>
    <div class="blocks">
      <div><h3>Regarding the Policy of Banned/Restricted Card Revision</h3>
        <p>Up to 4 copies of the same card number in a deck.</p>
      </div>
      <div>
        <h3>Banned/Restricted Cards effective from November 27, 2026</h3>
        <h4>Announced at November 17, 2026</h4>
        <h4>Banned Pair Cards</h4>
        <h5>Card A:</h5>
        <p><strong>EB05-001 Jewelry Bonney</strong></p>
        <h5>Card B:</h5>
        <p><strong>OP13-001 Monkey.D.Luffy<br> OP01-002 Trafalgar Law</strong></p>
        <img src="/onepiececg/EB05-001_p1.webp" alt="EB05-001 Jewelry Bonney">
        <img src="/onepiececg/OP13-001_p1.webp" alt="OP13-001 Monkey.D.Luffy">
      </div>
      <div>
        <h3>Cards with Active Restrictions</h3>
        <h4>Banned Cards</h4>
        <p>The following card(s) cannot be included in any deck.</p>
        <div class="editor_row"><ul>
          <li>・<a href="/cardlist/?freewords=OP06-047">OP06-047 Charlotte Pudding</a></li>
          <li>・<a href="/cardlist/?freewords=ST10-001">ST10-001 Trafalgar Law</a></li>
        </ul></div>
        <h4>Restricted Cards</h4>
        <p>There are currently no cards in this category.</p>
        <h4>Banned Pair Cards</h4>
        <p>Card A and Card B cannot be included in the same deck.</p>
        <div class="editor_row"><ul>
          <li>・<a href="/cardlist/?freewords=OP11-040">OP11-040 Monkey.D.Luffy</a></li>
          <li>・<a href="/cardlist/?freewords=OP11-067">OP11-067 Charlotte Katakuri</a></li>
        </ul></div>
        <div class="editor_row"><ul>
          <li>・<a href="/cardlist/?freewords=OP11-040">OP11-040 Monkey.D.Luffy</a></li>
          <li>・<a href="/cardlist/?freewords=OP08-069">OP08-069 Charlotte Linlin</a></li>
        </ul></div>
      </div>
      <div>
        <a class="commonBtn" href="https://en.onepiece-cardgame.com/news/restriction-261001.html">
          <span>Banned/Restricted Cards effective<br>from October 12, 2026</span>
        </a>
        <a class="commonBtn" href="https://en.onepiece-cardgame.com/news/restriction-260501.html">
          <span>Banned/Restricted Cards effective<br>from April 10, 2026</span>
        </a>
      </div>
    </div>
  </article>
</body>
</html>
"""


def test_parse_bandai_banlist_html_active_vs_upcoming() -> None:
    active_doc = parse_banlist_html(_REALISTIC_BANDAI_HTML)
    by_id = {rec.card_id: rec for rec in active_doc.restrictions}

    assert active_doc.effective_date == "2026-10-12"
    assert active_doc.upcoming_effective_date == "2026-11-27"
    assert active_doc.upcoming_included is False
    # Nav noise and upcoming announcements must NOT leak into active restrictions.
    assert "OP99-999" not in by_id
    assert "EB05-001" not in by_id
    assert "OP13-001" not in by_id
    assert set(by_id) == {"OP06-047", "ST10-001", "OP11-040", "OP11-067", "OP08-069"}

    assert by_id["OP06-047"].ban_status == "banned"
    assert by_id["OP06-047"].max_copies == 0
    assert by_id["OP06-047"].effective_date == "2026-10-12"

    # OP11-040 participates in TWO distinct banned pairs -> both partners sorted canonically.
    assert by_id["OP11-040"].ban_status == "pair_banned"
    assert by_id["OP11-040"].max_copies == 4
    assert by_id["OP11-040"].paired_card_ids == ("OP08-069", "OP11-067")
    assert "OP08-069, OP11-067" in (by_id["OP11-040"].restriction_note or "")
    assert by_id["OP11-067"].paired_card_ids == ("OP11-040",)
    assert by_id["OP08-069"].paired_card_ids == ("OP11-040",)

    # With include_upcoming=True, scheduled Card A x Card B pairs are merged symmetrically.
    upcoming_doc = parse_banlist_html(_REALISTIC_BANDAI_HTML, include_upcoming=True)
    up_by_id = {rec.card_id: rec for rec in upcoming_doc.restrictions}
    assert upcoming_doc.upcoming_included is True
    assert upcoming_doc.effective_date == "2026-11-27"
    assert up_by_id["EB05-001"].ban_status == "pair_banned"
    assert up_by_id["EB05-001"].paired_card_ids == ("OP01-002", "OP13-001")
    assert up_by_id["EB05-001"].effective_date == "2026-11-27"
    assert up_by_id["OP13-001"].paired_card_ids == ("EB05-001",)
    assert up_by_id["OP01-002"].paired_card_ids == ("EB05-001",)


def test_parse_bandai_banlist_html_restricted_and_precedence() -> None:
    html = """\
<html><body>
  <h3>Cards with Active Restrictions</h3>
  <h4>Banned Cards</h4>
  <ul><li>OP01-001 Roronoa Zoro</li></ul>
  <h4>Restricted Cards</h4>
  <ul>
    <li>OP01-001 Roronoa Zoro</li>
    <li>OP05-119 Monkey.D.Luffy</li>
  </ul>
  <h4>Banned Pair Cards</h4>
  <ul><li>OP05-119 Monkey.D.Luffy</li><li>P-001 Promo Luffy</li></ul>
</body></html>
"""
    doc = parse_banlist_html(html, effective_date="2026-09-01")
    by_id = {rec.card_id: rec for rec in doc.restrictions}

    # banned > restricted > pair_banned precedence
    assert by_id["OP01-001"].ban_status == "banned"
    assert by_id["OP01-001"].max_copies == 0
    assert by_id["OP05-119"].ban_status == "restricted"
    assert by_id["OP05-119"].max_copies == 1
    assert by_id["OP05-119"].paired_card_ids == ("P-001",)
    assert by_id["P-001"].ban_status == "pair_banned"
    assert by_id["P-001"].paired_card_ids == ("OP05-119",)


def test_parse_banlist_json_and_directory_overlay(tmp_path: Path) -> None:
    json_payload = {
        "effective_date": "2026-10-15",
        "source_url": "https://en.onepiece-cardgame.com/rules/restriction/",
        "restrictions": [
            {"card_id": "op04-083_p1", "ban_status": "restricted", "restriction_note": "Max 1."},
            {"card_id": "OP05-119", "ban_status": "pair-banned", "paired_card_ids": ["P-001"]},
        ],
        "banned_pairs": [["OP01-001", "OP05-119"]],
    }
    doc, legal_ids = parse_banlist_json(json_payload)
    assert legal_ids == set()
    by_id = {r.card_id: r for r in doc.restrictions}
    assert by_id["OP04-083"].ban_status == "restricted"
    assert by_id["OP04-083"].max_copies == 1
    assert by_id["OP04-083"].restriction_note == "Max 1."
    assert by_id["OP05-119"].paired_card_ids == ("OP01-001", "P-001")
    assert by_id["P-001"].paired_card_ids == ("OP05-119",)
    assert by_id["OP01-001"].paired_card_ids == ("OP05-119",)

    # Validation errors on malformed manual JSON.
    with pytest.raises(ValueError, match="Unknown ban_status"):
        parse_banlist_json([{"card_id": "OP01-001", "ban_status": "super_banned"}])
    with pytest.raises(ValueError, match="cannot be pair_banned with itself"):
        parse_banlist_json(
            [{"card_id": "OP01-001", "ban_status": "pair_banned", "paired_card_ids": ["OP01-001"]}]
        )
    with pytest.raises(ValueError, match="without paired_card_ids"):
        parse_banlist_json([{"card_id": "OP01-001", "ban_status": "pair_banned"}])

    # Directory merge: HTML base + JSON overlay unbanning one card and one half of a pair.
    ban_dir = tmp_path / "banlist"
    ban_dir.mkdir()
    (ban_dir / "restriction.html").write_text(_REALISTIC_BANDAI_HTML, encoding="utf-8")
    (ban_dir / "override.json").write_text(
        json.dumps(
            {
                "restrictions": [
                    {"card_id": "ST10-001", "ban_status": "legal"},
                    {"card_id": "OP08-069", "ban_status": "legal"},
                    {"card_id": "OP05-119", "ban_status": "restricted"},
                ]
            }
        ),
        encoding="utf-8",
    )
    merged = parse_banlist_path(ban_dir)
    merged_by_id = {r.card_id: r for r in merged.restrictions}
    assert "ST10-001" not in merged_by_id
    assert "OP08-069" not in merged_by_id
    assert merged_by_id["OP11-040"].paired_card_ids == ("OP11-067",)
    assert merged_by_id["OP05-119"].ban_status == "restricted"


def test_ingest_banlist_idempotency_unban_and_query_contracts(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, db_conn, _ = populated_db
    pre_vec_count = db_conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    assert pre_vec_count > 0

    # Before ingesting banlist, cards default to legal and banlist_loaded is False.
    before = get_card(db_conn, "OP05-119")
    assert before["banlist_loaded"] is False
    assert before["banlist_effective_date"] is None
    assert before["cards"][0]["ban_status"] == "legal"
    assert before["cards"][0]["max_copies"] == 4
    assert before["cards"][0]["paired_card_ids"] == []
    assert before["cards"][0]["restriction_note"] is None

    doc_v1, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "restrictions": [
                {"card_id": "OP01-001", "ban_status": "banned"},
                {"card_id": "OP04-083", "ban_status": "restricted"},
                {
                    "card_id": "OP05-119",
                    "ban_status": "pair_banned",
                    "paired_card_ids": ["P-001"],
                },
            ],
        }
    )
    first = ingest_banlist(db_conn, doc_v1)
    assert first.restrictions_upserted == 4
    assert first.restrictions_removed == 0
    assert first.banned_count == 1
    assert first.restricted_count == 1
    assert first.pair_banned_count == 2
    assert get_meta(db_conn, "banlist_effective_date") == "2026-10-12"

    # Second run is idempotent and leaves chunks_vec untouched.
    second = ingest_banlist(db_conn, doc_v1)
    assert second.restrictions_upserted == 4
    assert second.restrictions_removed == 0
    assert db_conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == pre_vec_count

    # Verify get_card contracts across exact_id, exact_name, and fuzzy_name.
    zoro = get_card(db_conn, "OP01-001")
    assert zoro["banlist_loaded"] is True
    assert zoro["banlist_effective_date"] == "2026-10-12"
    assert zoro["cards"][0]["ban_status"] == "banned"
    assert zoro["cards"][0]["max_copies"] == 0
    assert zoro["cards"][0]["effective_date"] == "2026-10-12"

    luffys = get_card(db_conn, "Monkey D Luffy")
    assert luffys["match_type"] == "exact_name"
    luffy_by_id = {c["card_id"]: c for c in luffys["cards"]}
    assert luffy_by_id["OP05-119"]["ban_status"] == "pair_banned"
    assert luffy_by_id["OP05-119"]["paired_card_ids"] == ["P-001"]
    assert luffy_by_id["P-001"]["ban_status"] == "pair_banned"
    assert luffy_by_id["P-001"]["paired_card_ids"] == ["OP05-119"]

    # Verify search_cards ban_status filtering.
    assert [c["card_id"] for c in search_cards(db_conn, ban_status="banned")["cards"]] == [
        "OP01-001"
    ]
    assert [c["card_id"] for c in search_cards(db_conn, ban_status="restricted")["cards"]] == [
        "OP04-083"
    ]
    assert [c["card_id"] for c in search_cards(db_conn, ban_status="pair_banned")["cards"]] == [
        "OP05-119",
        "P-001",
    ]
    assert [c["card_id"] for c in search_cards(db_conn, ban_status="legal")["cards"]] == [
        "OP02-106",
        "ST01-014",
        "ST10-001",
    ]

    with pytest.raises(ValueError, match="Unknown ban_status"):
        search_cards(db_conn, ban_status="invalid_status")

    # Verify get_banlist returns all categories + deduplicated banned_pairs in one call,
    # including cards not yet present in the cards table.
    doc_with_new_card, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "restrictions": [
                {"card_id": "OP01-001", "ban_status": "banned"},
                {"card_id": "OP99-001", "ban_status": "banned"},
                {"card_id": "OP04-083", "ban_status": "restricted"},
                {
                    "card_id": "OP05-119",
                    "ban_status": "pair_banned",
                    "paired_card_ids": ["P-001"],
                },
            ],
        }
    )
    ingest_banlist(db_conn, doc_with_new_card)

    full_banlist = get_banlist(db_conn)
    assert full_banlist["banlist_loaded"] is True
    assert full_banlist["banlist_effective_date"] == "2026-10-12"
    assert full_banlist["counts"] == {
        "banned": 2,
        "restricted": 1,
        "pair_banned": 2,
        "total": 5,
    }
    assert full_banlist["banned_pairs"] == [["OP05-119", "P-001"]]
    by_cid = {r["card_id"]: r for r in full_banlist["restrictions"]}
    assert by_cid["OP01-001"]["card_found"] is True
    assert by_cid["OP01-001"]["name"] == "Roronoa Zoro"
    assert by_cid["OP99-001"]["card_found"] is False
    assert by_cid["OP99-001"]["name"] is None

    banned_only = get_banlist(db_conn, ban_status="banned")
    assert [r["card_id"] for r in banned_only["restrictions"]] == ["OP01-001", "OP99-001"]
    assert banned_only["banned_pairs"] == []

    with pytest.raises(ValueError, match="only returns restricted cards"):
        get_banlist(db_conn, ban_status="legal")

    # Unbanning OP01-001 removes its stale row and restores it to legal.
    doc_v2, _ = parse_banlist_json(
        {
            "effective_date": "2026-11-01",
            "restrictions": [{"card_id": "OP04-083", "ban_status": "restricted"}],
        }
    )
    third = ingest_banlist(db_conn, doc_v2)
    assert third.restrictions_upserted == 1
    assert third.restrictions_removed == 4
    assert get_card(db_conn, "OP01-001")["cards"][0]["ban_status"] == "legal"
    assert [c["card_id"] for c in search_cards(db_conn, ban_status="legal")["cards"]] == [
        "OP01-001",
        "OP02-106",
        "OP05-119",
        "P-001",
        "ST01-014",
        "ST10-001",
    ]


def test_cli_banlist_command_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "banlist_cli.db"
    html_path = tmp_path / "restriction.html"
    html_path.write_text(_REALISTIC_BANDAI_HTML, encoding="utf-8")

    exit_code = cli_main(["--db", str(db_path), "banlist", str(html_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Ingested 5 card restriction(s)" in out
    assert "banned=2, restricted=0, pair_banned=3" in out
    assert "effective_date=2026-10-12" in out

    # Re-run with --include-upcoming adds EB05-001, OP13-001, OP01-002.
    exit_code_up = cli_main(
        ["--db", str(db_path), "banlist", str(html_path), "--include-upcoming"]
    )
    assert exit_code_up == 0
    out_up = capsys.readouterr().out
    assert "Ingested 8 card restriction(s)" in out_up
    assert "effective_date=2026-11-27" in out_up

    with connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM card_restrictions").fetchone()[0]
        assert count == 8
