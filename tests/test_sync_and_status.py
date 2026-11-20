"""Offline tests for `optcg-ingest sync`, `optcg-ingest status`, and `get_status()`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pymupdf
import pytest

from optcg_knowledge import cli
from optcg_knowledge.config import get_settings
from optcg_knowledge.curation import add_ruling, mark_ruling
from optcg_knowledge.ingest import (
    HashEmbedder,
    embed_chunks,
    ingest_banlist,
    ingest_trm,
    parse_banlist_json,
    parse_trm_text,
)
from optcg_knowledge.ingest.sources import LocalCardSource
from optcg_knowledge.search import get_status
from optcg_knowledge.sync import run_sync


def _make_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    lines = text.splitlines()
    for start in range(0, max(len(lines), 1), 16):
        page = doc.new_page(width=595, height=842)
        chunk = "\n".join(lines[start : start + 16])
        page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), chunk, fontsize=10)
    data = bytes(doc.tobytes())
    doc.close()
    return data


def _sample_trm_text() -> str:
    return """\
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
"""


def _sample_faq_text() -> str:
    return """\
Revision Date: 2025-09-01
OP05-119 Monkey.D.Luffy
Q: Can I activate OP05-119 with 0 other Characters?
A: Yes, placing 0 other Characters still grants the extra turn.
"""


def _sample_errata_html() -> str:
    return """\
<!DOCTYPE html>
<html>
<body>
<section class="contentsLCol">
  <h4 class="mediumTit">August 10, 2025</h4>
  <div class="detailCol" id="errata_01">
    <h5 class="smallTitRed">OP01-001 Roronoa Zoro</h5>
    <dl>
      <dt>Before:</dt>
      <dd>All of your Characters gain +1000 power.</dd>
      <dt>After:</dt>
      <dd>[DON!! x1] All of your Characters gain +1000 power during your turn.</dd>
    </dl>
  </div>
</section>
</body>
</html>
"""


def _sample_banlist_html() -> str:
    return """\
<!DOCTYPE html>
<html>
<body>
<main>
  <h3>Cards with Active Restrictions</h3>
  <h4>Banned Cards</h4>
  <div class="editor_row"><ul>
    <li>・<a href="/cardlist/?freewords=OP01-001">OP01-001 Roronoa Zoro</a></li>
  </ul></div>
  <h4>Banned Pair Cards</h4>
  <div class="editor_row"><ul>
    <li>・<a href="/cardlist/?freewords=OP05-119">OP05-119 Monkey.D.Luffy</a></li>
    <li>・<a href="/cardlist/?freewords=P-001">P-001 Monkey.D.Luffy</a></li>
  </ul></div>
  <div>
    <a class="commonBtn" href="https://en.onepiece-cardgame.com/news/restriction-261001.html">
      <span>Banned/Restricted Cards effective<br>from October 12, 2026</span>
    </a>
  </div>
</main>
</body>
</html>
"""


def _sample_curated_rulings_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "exported_at": "2026-10-15T00:00:00+00:00",
        "rules_version": "1.24",
        "total": 2,
        "rulings": [
            {
                "ruling_id": "judge:2026-10-10:t0-001",
                "question": "Does OP05-119 trigger during an extra turn?",
                "answer": "Yes, normal turn-start effects resolve during the extra turn.",
                "source": "judge",
                "authority": 3,
                "judge_tier": "tier_0",
                "date": "2026-10-10",
                "rules_version": "1.24",
                "url": None,
                "superseded_by": None,
                "reviewed": True,
                "card_ids": ["OP05-119"],
                "rule_ids": ["1-3-2"],
            },
            {
                "ruling_id": "judge:2026-10-11:t2-002",
                "question": "Can [Blocker] activate after Counter Step?",
                "answer": "No, [Blocker] can only activate during the Block Step.",
                "source": "judge",
                "authority": 3,
                "judge_tier": "tier_2",
                "date": "2026-10-11",
                "rules_version": "1.24",
                "url": None,
                "superseded_by": None,
                "reviewed": True,
                "card_ids": ["OP04-083"],
                "rule_ids": ["7-1-4"],
            },
        ],
    }


def _build_mock_official_client(sample_rules_text: str) -> httpx.Client:
    rules_pdf_bytes = _make_pdf_bytes(sample_rules_text)
    trm_pdf_bytes = _make_pdf_bytes(_sample_trm_text())
    faq_pdf_bytes = _make_pdf_bytes(_sample_faq_text())
    errata_html = _sample_errata_html()
    banlist_html = _sample_banlist_html()

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/rules/":
            return httpx.Response(
                200,
                text=(
                    '<a href="/pdf/rule_comprehensive.pdf?20250901">Rules</a>'
                    '<a href="/pdf/tournament_rules_manual.pdf?20251017">TRM</a>'
                ),
            )
        if path == "/pdf/rule_comprehensive.pdf":
            return httpx.Response(200, content=rules_pdf_bytes)
        if path == "/pdf/tournament_rules_manual.pdf":
            return httpx.Response(200, content=trm_pdf_bytes)
        if path == "/rules/faq/":
            return httpx.Response(200, text='<a href="/pdf/qa_op05.pdf?20250901">OP05 FAQ</a>')
        if path == "/pdf/qa_op05.pdf":
            return httpx.Response(200, content=faq_pdf_bytes)
        if path == "/rules/errata_card/":
            return httpx.Response(200, text=errata_html)
        if path == "/rules/restriction/":
            return httpx.Response(200, text=banlist_html)
        return httpx.Response(404, text=f"Not found: {path}")

    return httpx.Client(transport=httpx.MockTransport(_handler))


def test_get_status_empty_db(db_conn: sqlite3.Connection) -> None:
    status = get_status(db_conn)
    assert status["schema_version"] == "4"
    assert status["rules_version"] is None
    assert status["trm_version"] is None
    assert status["banlist_loaded"] is False
    assert status["banlist_effective_date"] is None
    assert status["embedding_model"] == "test/hash-embedder"
    assert status["embedding_dim"] == 64

    assert status["counts"]["rules"] == 0
    assert status["counts"]["trm_sections"] == 0
    assert status["counts"]["cards"] == 0
    assert status["counts"]["card_restrictions"] == {
        "total": 0,
        "banned": 0,
        "restricted": 0,
        "pair_banned": 0,
    }
    assert status["counts"]["rulings"]["total"] == 0
    assert status["counts"]["rulings"]["unreviewed"] == 0
    assert status["counts"]["chunks"]["total"] == 0

    assert status["embeddings"]["total_chunks"] == 0
    assert status["embeddings"]["embedded_chunks"] == 0
    assert status["embeddings"]["pending_chunks"] == 0
    assert status["embeddings"]["coverage_pct"] == 0.0
    assert status["embeddings"]["ready"] is False
    assert status["timestamps"]["last_ingested_at"] is None
    assert status["timestamps"]["last_embed_at"] is None


def test_get_status_breakdown_by_tier_unreviewed_and_partial_embeddings(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db
    ingest_trm(conn, parse_trm_text(_sample_trm_text()))

    ban_doc, _ = parse_banlist_json(
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
    ingest_banlist(conn, ban_doc)

    # Add curated rulings (unreviewed by default) across judge tiers and errata.
    t0 = add_ruling(
        conn,
        question="Tier 0 judge question about OP05-119?",
        answer="Tier 0 judge answer.",
        source="judge",
        judge_tier="tier_0",
        card_ids=["OP05-119"],
    )
    add_ruling(
        conn,
        question="Tier 1 judge question about OP04-083?",
        answer="Tier 1 judge answer.",
        source="judge",
        judge_tier="tier_1",
        card_ids=["OP04-083"],
    )
    add_ruling(
        conn,
        question="Official errata correction for OP01-001?",
        answer="Updated text applies during your turn.",
        source="errata",
        card_ids=["OP01-001"],
    )
    mark_ruling(conn, t0["ruling_id"], reviewed=True)

    partial = get_status(conn)
    assert partial["counts"]["trm_sections"] > 0
    assert partial["counts"]["card_restrictions"] == {
        "total": 4,
        "banned": 1,
        "restricted": 1,
        "pair_banned": 2,
    }
    assert partial["counts"]["rulings"]["by_source"] == {
        "rules": 0,
        "official_faq": 2,
        "errata": 1,
        "judge": 4,
        "community": 1,
    }
    assert partial["counts"]["rulings"]["by_authority"] == {
        "1": 0,
        "2": 3,
        "3": 4,
        "4": 1,
    }
    assert partial["counts"]["rulings"]["judge_by_tier"] == {
        "tier_0": 1,
        "tier_2": 0,
        "tier_1": 1,
        "unspecified": 2,
    }
    assert partial["counts"]["rulings"]["unreviewed"] == 2
    assert partial["counts"]["rulings"]["superseded"] == 1
    assert partial["embeddings"]["pending_chunks"] > 0
    assert partial["embeddings"]["ready"] is False
    assert 0.0 < partial["embeddings"]["coverage_pct"] < 100.0

    # Embed remaining chunks and verify 100% coverage.
    embed_chunks(conn, embedder, batch_size=8)
    complete = get_status(conn)
    assert complete["embeddings"]["pending_chunks"] == 0
    assert complete["embeddings"]["coverage_pct"] == 100.0
    assert complete["embeddings"]["ready"] is True
    assert complete["timestamps"]["trm_last_ingested_at"] is not None
    assert complete["timestamps"]["banlist_last_ingested_at"] is not None


def test_run_sync_full_offline_pipeline_and_idempotency(
    tmp_path: Path,
    sample_rules_text: str,
    fixtures_dir: Path,
) -> None:
    db_path = tmp_path / "data" / "optcg.db"
    raw_dir = tmp_path / "data" / "raw"
    rulings_dir = tmp_path / "data" / "rulings"
    rulings_dir.mkdir(parents=True)
    (rulings_dir / "judge_rulings.json").write_text(
        json.dumps(_sample_curated_rulings_payload()),
        encoding="utf-8",
    )

    settings = get_settings(db_path=db_path, raw_data_dir=raw_dir)
    embedder = HashEmbedder(dimension=64)
    card_source = LocalCardSource(fixtures_dir / "sample_cards_punk_records.json")

    with _build_mock_official_client(sample_rules_text) as mock_client:
        first = run_sync(
            settings,
            fetch_client=mock_client,
            card_source=card_source,
            embedder=embedder,
        )
        second = run_sync(
            settings,
            fetch_client=mock_client,
            card_source=card_source,
            embedder=embedder,
        )

    assert [step.name for step in first.steps] == [
        "fetch-official",
        "rules",
        "trm",
        "cards",
        "banlist",
        "faq",
        "import-rulings",
        "embed",
    ]
    assert all(step.status == "completed" for step in first.steps)

    snap1 = first.status_snapshot
    assert snap1["schema_version"] == "4"
    assert snap1["rules_version"] == "1.24"
    assert snap1["banlist_loaded"] is True
    assert snap1["banlist_effective_date"] == "2026-10-12"
    assert snap1["counts"]["rules"] > 0
    assert snap1["counts"]["trm_sections"] > 0
    assert snap1["counts"]["cards"] > 0
    assert snap1["counts"]["card_restrictions"]["banned"] == 1
    assert snap1["counts"]["card_restrictions"]["pair_banned"] == 2
    assert snap1["counts"]["rulings"]["by_source"]["official_faq"] >= 1
    assert snap1["counts"]["rulings"]["by_source"]["errata"] >= 1
    assert snap1["counts"]["rulings"]["by_source"]["judge"] == 2
    assert snap1["counts"]["rulings"]["judge_by_tier"]["tier_0"] == 1
    assert snap1["counts"]["rulings"]["judge_by_tier"]["tier_2"] == 1
    assert snap1["embeddings"]["pending_chunks"] == 0
    assert snap1["embeddings"]["coverage_pct"] == 100.0
    assert snap1["embeddings"]["ready"] is True
    assert snap1["timestamps"]["last_faq_ingested_at"] is not None
    assert snap1["timestamps"]["last_embed_at"] is not None

    # Second run must be fully idempotent: no duplicate rows, 0 re-embedded chunks, 0 re-downloads.
    fetch_step2 = second.step_by_name("fetch-official")
    embed_step2 = second.step_by_name("embed")
    assert fetch_step2 is not None and fetch_step2.metrics["downloaded"] == 0
    assert embed_step2 is not None and embed_step2.metrics["embedded_chunks"] == 0
    assert second.status_snapshot["counts"] == snap1["counts"]
    assert (
        second.status_snapshot["embeddings"]["total_chunks"]
        == snap1["embeddings"]["total_chunks"]
    )


def test_run_sync_auto_skips_embed_when_embeddings_extra_missing(
    tmp_path: Path,
    sample_rules_pdf: Path,
) -> None:
    db_path = tmp_path / "optcg.db"
    raw_dir = tmp_path / "raw"
    (raw_dir / "rules").mkdir(parents=True)
    (raw_dir / "rules" / "rule_comprehensive.pdf").write_bytes(sample_rules_pdf.read_bytes())

    settings = get_settings(db_path=db_path, raw_data_dir=raw_dir)
    summary = run_sync(
        settings,
        skip_fetch=True,
        skip_cards=True,
        embeddings_available=lambda: False,
    )

    fetch_step = summary.step_by_name("fetch-official")
    rules_step = summary.step_by_name("rules")
    trm_step = summary.step_by_name("trm")
    cards_step = summary.step_by_name("cards")
    banlist_step = summary.step_by_name("banlist")
    faq_step = summary.step_by_name("faq")
    rulings_step = summary.step_by_name("import-rulings")
    embed_step = summary.step_by_name("embed")

    assert fetch_step is not None and fetch_step.status == "skipped"
    assert rules_step is not None and rules_step.status == "completed"
    assert trm_step is not None and trm_step.status == "skipped"
    assert cards_step is not None and cards_step.status == "skipped"
    assert banlist_step is not None and banlist_step.status == "skipped"
    assert faq_step is not None and faq_step.status == "skipped"
    assert rulings_step is not None and rulings_step.status == "skipped"
    assert embed_step is not None and embed_step.status == "skipped"
    assert "uv sync --extra embeddings" in embed_step.detail

    assert summary.status_snapshot["counts"]["rules"] > 0
    assert summary.status_snapshot["embeddings"]["embedded_chunks"] == 0
    assert summary.status_snapshot["embeddings"]["pending_chunks"] > 0
    assert summary.status_snapshot["embeddings"]["ready"] is False


def test_cli_sync_and_status_offline(
    tmp_path: Path,
    sample_rules_pdf: Path,
    fixtures_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "cli_optcg.db"
    raw_dir = tmp_path / "raw"
    (raw_dir / "rules").mkdir(parents=True)
    (raw_dir / "rules" / "rule_comprehensive.pdf").write_bytes(sample_rules_pdf.read_bytes())

    # 1. `status` before DB exists must report cleanly without creating an empty SQLite file.
    exit_missing = cli.main(["--db", str(db_path), "status"])
    assert exit_missing == 0
    assert not db_path.exists()
    assert "optcg-ingest sync" in capsys.readouterr().out

    exit_missing_json = cli.main(["--db", str(db_path), "status", "--json"])
    assert exit_missing_json == 0
    assert not db_path.exists()
    missing_payload = json.loads(capsys.readouterr().out)
    assert missing_payload["exists"] is False

    # 2. `sync` with local files and `--no-embed`.
    exit_sync = cli.main(
        [
            "--db",
            str(db_path),
            "sync",
            "--raw-dir",
            str(raw_dir),
            "--skip-fetch",
            "--cards-source",
            "local",
            "--cards-path",
            str(fixtures_dir / "sample_cards_punk_records.json"),
            "--no-embed",
        ]
    )
    assert exit_sync == 0
    sync_out = capsys.readouterr().out
    assert "[OK] rules:" in sync_out
    assert "[OK] cards:" in sync_out
    assert "[SKIP] embed:" in sync_out

    # 3. `status` human-readable output and `--json` output.
    exit_status = cli.main(["--db", str(db_path), "status"])
    assert exit_status == 0
    status_text = capsys.readouterr().out
    assert "OPTCG Knowledge Base Status" in status_text
    assert "schema_version:" in status_text
    assert "rules_version:" in status_text
    assert "coverage:" in status_text

    exit_status_json = cli.main(["--db", str(db_path), "status", "--json"])
    assert exit_status_json == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["schema_version"] == "4"
    assert status_payload["rules_version"] == "1.24"
    assert status_payload["counts"]["rules"] > 0
    assert status_payload["counts"]["cards"] > 0
    assert status_payload["embeddings"]["embedded_chunks"] == 0
    assert status_payload["embeddings"]["pending_chunks"] == status_payload["counts"]["chunks"][
        "total"
    ]
