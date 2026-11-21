"""Offline tests for Bandai banlist HTML/JSON parsing, ingestion, queries, and CLI."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.db import (
    LEADER_MAX_COPIES,
    MAX_MAIN_DECK_COPIES,
    connect,
    get_meta,
    has_unlimited_copies_rule,
    resolve_card_max_copies,
)
from optcg_knowledge.ingest.banlist_parser import (
    ingest_banlist,
    parse_banlist_html,
    parse_banlist_json,
    parse_banlist_path,
)
from optcg_knowledge.ingest.cards_importer import ingest_cards
from optcg_knowledge.ingest.embeddings import HashEmbedder
from optcg_knowledge.ingest.sources import LocalCardSource
from optcg_knowledge.search.queries import get_banlist, get_card, get_cards, search_cards

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
    doc = parse_banlist_html(_REALISTIC_BANDAI_HTML)
    by_id = {rec.card_id: rec for rec in doc.restrictions}
    up_by_id = {rec.card_id: rec for rec in doc.upcoming_restrictions}

    assert doc.effective_date == "2026-10-12"
    assert doc.upcoming_effective_date == "2026-11-27"
    assert doc.upcoming_included is True
    # Nav noise and upcoming announcements must NOT leak into active restrictions.
    assert "OP99-999" not in by_id
    assert "EB05-001" not in by_id
    assert "OP13-001" not in by_id
    assert set(by_id) == {"OP06-047", "ST10-001", "OP11-040", "OP11-067", "OP08-069"}
    assert all(rec.scope == "active" for rec in doc.restrictions)

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

    # Scheduled Card A x Card B pairs are extracted into upcoming_restrictions separately.
    assert set(up_by_id) == {"EB05-001", "OP01-002", "OP13-001"}
    assert all(rec.scope == "upcoming" for rec in doc.upcoming_restrictions)
    assert up_by_id["EB05-001"].ban_status == "pair_banned"
    assert up_by_id["EB05-001"].paired_card_ids == ("OP01-002", "OP13-001")
    assert up_by_id["EB05-001"].effective_date == "2026-11-27"
    assert up_by_id["OP13-001"].paired_card_ids == ("EB05-001",)
    assert up_by_id["OP01-002"].paired_card_ids == ("EB05-001",)

    # Passing include_upcoming=False omits upcoming_restrictions while keeping active intact.
    active_only_doc = parse_banlist_html(_REALISTIC_BANDAI_HTML, include_upcoming=False)
    assert active_only_doc.effective_date == "2026-10-12"
    assert active_only_doc.upcoming_effective_date == "2026-11-27"
    assert active_only_doc.upcoming_included is False
    assert active_only_doc.upcoming_restrictions == ()
    assert {r.card_id for r in active_only_doc.restrictions} == set(by_id)


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


def test_upcoming_restrictions_do_not_pollute_active_legality(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, db_conn, _ = populated_db

    # OP04-083 is currently restricted (max 1) and scheduled to be banned on 2026-11-27.
    # OP02-106 and ST01-014 are currently legal and scheduled to be pair_banned on 2026-11-27.
    doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "upcoming_effective_date": "2026-11-27",
            "restrictions": [
                {"card_id": "OP04-083", "ban_status": "restricted"},
            ],
            "upcoming_restrictions": [
                {"card_id": "OP04-083", "ban_status": "banned"},
            ],
            "upcoming_banned_pairs": [["OP02-106", "ST01-014"]],
        }
    )
    summary = ingest_banlist(db_conn, doc)
    assert summary.restrictions_upserted == 1
    assert summary.upcoming_restrictions_upserted == 3
    assert summary.effective_date == "2026-10-12"
    assert summary.upcoming_effective_date == "2026-11-27"

    # OP04-083 remains "restricted" today, with upcoming_restriction="banned".
    sabo = get_card(db_conn, "OP04-083")
    assert sabo["banlist_effective_date"] == "2026-10-12"
    assert sabo["banlist_upcoming_effective_date"] == "2026-11-27"
    assert sabo["cards"][0]["ban_status"] == "restricted"
    assert sabo["cards"][0]["max_copies"] == 1
    assert sabo["cards"][0]["upcoming_restriction"] == {
        "ban_status": "banned",
        "max_copies": 0,
        "paired_card_ids": [],
        "restriction_note": "Cannot be included in any deck.",
        "effective_date": "2026-11-27",
        "source_url": None,
    }

    # OP02-106 remains "legal" (max 4) today in both get_card and search_cards(ban_status="legal"),
    # while surfacing its scheduled upcoming pair ban.
    tsuru = get_card(db_conn, "OP02-106")
    assert tsuru["cards"][0]["ban_status"] == "legal"
    assert tsuru["cards"][0]["max_copies"] == 4
    assert tsuru["cards"][0]["upcoming_restriction"] == {
        "ban_status": "pair_banned",
        "max_copies": 4,
        "paired_card_ids": ["ST01-014"],
        "restriction_note": "Cannot be included in the same deck as ST01-014.",
        "effective_date": "2026-11-27",
        "source_url": None,
    }

    legal_cards = search_cards(db_conn, ban_status="legal")
    legal_by_id = {c["card_id"]: c for c in legal_cards["cards"]}
    assert "OP02-106" in legal_by_id
    assert "ST01-014" in legal_by_id
    assert "OP04-083" not in legal_by_id
    assert legal_by_id["OP02-106"]["upcoming_restriction"]["ban_status"] == "pair_banned"
    assert legal_by_id["OP01-001"]["upcoming_restriction"] is None

    # get_banlist separates active restrictions from the dedicated upcoming section.
    banlist = get_banlist(db_conn)
    assert banlist["banlist_effective_date"] == "2026-10-12"
    assert banlist["banlist_upcoming_effective_date"] == "2026-11-27"
    assert banlist["counts"] == {
        "banned": 0,
        "restricted": 1,
        "pair_banned": 0,
        "total": 1,
    }
    assert [r["card_id"] for r in banlist["restrictions"]] == ["OP04-083"]
    assert banlist["upcoming"]["effective_date"] == "2026-11-27"
    assert banlist["upcoming"]["counts"] == {
        "banned": 1,
        "restricted": 0,
        "pair_banned": 2,
        "total": 3,
    }
    assert banlist["upcoming"]["banned_pairs"] == [["OP02-106", "ST01-014"]]
    assert [r["card_id"] for r in banlist["upcoming"]["restrictions"]] == [
        "OP04-083",
        "OP02-106",
        "ST01-014",
    ]


def test_cli_banlist_command_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "banlist_cli.db"
    html_path = tmp_path / "restriction.html"
    html_path.write_text(_REALISTIC_BANDAI_HTML, encoding="utf-8")

    # By default, ingests active (5) and upcoming (3) separately.
    exit_code = cli_main(["--db", str(db_path), "banlist", str(html_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Ingested 5 card restriction(s)" in out
    assert "banned=2, restricted=0, pair_banned=3, upcoming=3" in out
    assert "effective_date=2026-10-12" in out
    assert "upcoming_effective_date=2026-11-27" in out

    with connect(db_path) as conn:
        active_count = conn.execute(
            "SELECT COUNT(*) FROM card_restrictions WHERE scope = 'active'"
        ).fetchone()[0]
        upcoming_count = conn.execute(
            "SELECT COUNT(*) FROM card_restrictions WHERE scope = 'upcoming'"
        ).fetchone()[0]
        assert active_count == 5
        assert upcoming_count == 3

    # Re-run with --no-include-upcoming removes the 3 upcoming rows while keeping the 5 active rows.
    exit_code_no_up = cli_main(
        ["--db", str(db_path), "banlist", str(html_path), "--no-include-upcoming"]
    )
    assert exit_code_no_up == 0
    out_no_up = capsys.readouterr().out
    assert "Ingested 5 card restriction(s)" in out_no_up
    assert "removed=3" in out_no_up
    assert "upcoming=0" in out_no_up
    assert "upcoming_effective_date=none" in out_no_up

    with connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM card_restrictions").fetchone()[0]
        assert count == 5


def test_card_max_copies_respects_unlimited_rule_overrides_and_leader_limits(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    tmp_path: Path,
) -> None:
    _, db_conn, _ = populated_db

    # Unit checks for whitespace/case tolerance and non-matching "Under the rules" clauses.
    assert has_unlimited_copies_rule(
        "Under the rules of this game,\n  you may have any number of this card in your deck."
    )
    assert not has_unlimited_copies_rule(
        "Under the rules of this game, you cannot include cards with a cost of 5 or more."
    )
    assert (
        resolve_card_max_copies("legal", category="Leader", effect_text=None) == LEADER_MAX_COPIES
    )
    assert (
        resolve_card_max_copies("banned", category="Leader", effect_text=None) == 0
    )

    # Ingest all four canonical unlimited-copy cards plus non-copy "Under the rules" cards.
    override_cards_path = tmp_path / "override_cards.json"
    override_cards_path.write_text(
        json.dumps(
            [
                {
                    "id": "OP01-075",
                    "name": "Pacifista",
                    "rarity": "Common",
                    "block_number": 1,
                    "category": "Character",
                    "colors": ["Blue"],
                    "cost": 4,
                    "power": 5000,
                    "counter": None,
                    "attributes": ["Special"],
                    "types": ["Biological Weapon", "Navy"],
                    "effect": (
                        "Under the rules of this game, you may have any number of this card "
                        "in your deck. [Blocker]"
                    ),
                },
                {
                    "id": "OP08-072",
                    "name": "Biscuit Warrior",
                    "rarity": "Common",
                    "block_number": 2,
                    "category": "Character",
                    "colors": ["Purple"],
                    "cost": 4,
                    "power": 5000,
                    "counter": 1000,
                    "attributes": ["Slash"],
                    "types": ["Big Mom Pirates"],
                    "effect": (
                        "Under the rules of this game, you may have any number of this card "
                        "in your deck. [Blocker]"
                    ),
                },
                {
                    "id": "OP16-042",
                    "name": "Prisoner of Impel Down",
                    "rarity": "Common",
                    "block_number": 4,
                    "category": "Character",
                    "colors": ["Blue"],
                    "cost": 2,
                    "power": 3000,
                    "counter": 1000,
                    "attributes": ["Strike"],
                    "types": ["Impel Down"],
                    "effect": (
                        "Under the rules of this game, you may have any number of this card "
                        "in your deck."
                    ),
                },
                {
                    "id": "OP18-093",
                    "name": "Sleeptids",
                    "rarity": "Common",
                    "block_number": 4,
                    "category": "Character",
                    "colors": ["Black"],
                    "cost": 3,
                    "power": 4000,
                    "counter": 1000,
                    "attributes": ["Special"],
                    "types": ["Animal"],
                    "effect": (
                        "Under the rules of this game, you may have any number of this card "
                        "in your deck. [Blocker]"
                    ),
                },
                {
                    "id": "EB04-038",
                    "name": "Rosinante & Law",
                    "rarity": "SuperRare",
                    "block_number": 4,
                    "category": "Character",
                    "colors": ["Blue"],
                    "cost": 4,
                    "power": 5000,
                    "counter": 1000,
                    "attributes": ["Slash"],
                    "types": ["Donquixote Pirates"],
                    "effect": (
                        "Under the rules of this game, also treat this card's name as "
                        "[Trafalgar Law] and [Donquixote Rosinante]."
                    ),
                },
                {
                    "id": "OP12-001",
                    "name": "Silvers Rayleigh",
                    "rarity": "Leader",
                    "block_number": 3,
                    "category": "Leader",
                    "colors": ["Red"],
                    "cost": 5,
                    "power": 5000,
                    "counter": None,
                    "attributes": ["Slash"],
                    "types": ["Former Roger Pirates"],
                    "effect": (
                        "Under the rules of this game, you cannot include cards with a cost "
                        "of 5 or more in your deck."
                    ),
                },
            ]
        ),
        encoding="utf-8",
    )
    ingest_cards(db_conn, LocalCardSource(override_cards_path))

    # Legal baseline: unlimited-copy -> 50, normal Character -> 4, Leaders -> 1.
    for cid in ("OP01-075", "OP08-072", "OP16-042", "OP18-093"):
        res = get_card(db_conn, cid)
        assert res["cards"][0]["ban_status"] == "legal"
        assert res["cards"][0]["max_copies"] == MAX_MAIN_DECK_COPIES

    assert get_card(db_conn, "EB04-038")["cards"][0]["max_copies"] == 4
    assert get_card(db_conn, "OP05-119")["cards"][0]["max_copies"] == 4
    for leader_id in ("OP01-001", "ST10-001", "OP12-001"):
        leader_res = get_card(db_conn, leader_id)
        assert leader_res["cards"][0]["category"] == "Leader"
        assert leader_res["cards"][0]["max_copies"] == LEADER_MAX_COPIES

    # Apply banlist restrictions testing precedence against unlimited-copy cards and Leaders.
    doc, _ = parse_banlist_json(
        {
            "effective_date": "2026-10-12",
            "upcoming_effective_date": "2026-11-27",
            "restrictions": [
                {"card_id": "OP01-075", "ban_status": "banned"},
                {"card_id": "OP08-072", "ban_status": "restricted"},
                {"card_id": "ST10-001", "ban_status": "banned"},
            ],
            "banned_pairs": [["OP16-042", "OP01-001"]],
            "upcoming_banned_pairs": [["OP18-093", "OP12-001"]],
        }
    )
    ingest_banlist(db_conn, doc)

    # Batch get_cards reflects every precedence rule cleanly.
    batch = get_cards(
        db_conn,
        ["OP01-075", "OP08-072", "OP16-042", "OP18-093", "OP01-001", "ST10-001", "OP12-001"],
    )
    by_id = {c["card_id"]: c for c in batch["cards"]}
    assert by_id["OP01-075"]["ban_status"] == "banned"
    assert by_id["OP01-075"]["max_copies"] == 0
    assert by_id["OP08-072"]["ban_status"] == "restricted"
    assert by_id["OP08-072"]["max_copies"] == 1
    assert by_id["OP16-042"]["ban_status"] == "pair_banned"
    assert by_id["OP16-042"]["max_copies"] == MAX_MAIN_DECK_COPIES
    assert by_id["OP01-001"]["ban_status"] == "pair_banned"
    assert by_id["OP01-001"]["max_copies"] == LEADER_MAX_COPIES
    assert by_id["ST10-001"]["ban_status"] == "banned"
    assert by_id["ST10-001"]["max_copies"] == 0
    assert by_id["OP18-093"]["ban_status"] == "legal"
    assert by_id["OP18-093"]["max_copies"] == MAX_MAIN_DECK_COPIES
    assert by_id["OP18-093"]["upcoming_restriction"]["ban_status"] == "pair_banned"
    assert by_id["OP18-093"]["upcoming_restriction"]["max_copies"] == MAX_MAIN_DECK_COPIES
    assert by_id["OP12-001"]["ban_status"] == "legal"
    assert by_id["OP12-001"]["max_copies"] == LEADER_MAX_COPIES
    assert by_id["OP12-001"]["upcoming_restriction"]["ban_status"] == "pair_banned"
    assert by_id["OP12-001"]["upcoming_restriction"]["max_copies"] == LEADER_MAX_COPIES

    # search_cards returns identical effective max_copies.
    pair_cards = {
        c["card_id"]: c["max_copies"]
        for c in search_cards(db_conn, ban_status="pair_banned")["cards"]
    }
    assert pair_cards == {"OP01-001": 1, "OP16-042": 50}

    # get_banlist aligns active and upcoming max_copies with joined card rules.
    banlist = get_banlist(db_conn)
    active_max = {r["card_id"]: r["max_copies"] for r in banlist["restrictions"]}
    assert active_max == {
        "OP01-075": 0,
        "ST10-001": 0,
        "OP08-072": 1,
        "OP01-001": 1,
        "OP16-042": 50,
    }
    upcoming_max = {r["card_id"]: r["max_copies"] for r in banlist["upcoming"]["restrictions"]}
    assert upcoming_max == {"OP12-001": 1, "OP18-093": 50}
