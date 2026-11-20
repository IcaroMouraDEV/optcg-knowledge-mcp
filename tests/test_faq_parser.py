"""Offline tests for official Bandai FAQ & errata parsing, ingestion, and search integration."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pymupdf
import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.db import open_db
from optcg_knowledge.ingest.cards_importer import ingest_cards
from optcg_knowledge.ingest.embed_pipeline import embed_chunks
from optcg_knowledge.ingest.embeddings import HashEmbedder
from optcg_knowledge.ingest.faq_parser import (
    ingest_faqs,
    parse_faq_html,
    parse_faq_path,
    parse_faq_pdf,
    parse_faq_text,
)
from optcg_knowledge.ingest.rules_parser import ingest_rules, parse_rules_text
from optcg_knowledge.ingest.sources import LocalCardSource
from optcg_knowledge.search.hybrid import hybrid_search
from optcg_knowledge.search.queries import get_card_rulings
from tests.conftest import insert_test_ruling


def _make_pdf_from_text(pdf_path: Path, text: str, *, lines_per_page: int = 12) -> Path:
    """Write a real multi-page PDF on disk using PyMuPDF for offline roundtrip tests."""
    doc = pymupdf.open()
    lines = text.splitlines()
    for start in range(0, len(lines), lines_per_page):
        page = doc.new_page(width=595, height=842)
        block = "\n".join(lines[start : start + lines_per_page])
        page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), block, fontsize=10)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_parse_faq_pdf_roundtrip(tmp_path: Path, fixtures_dir: Path) -> None:
    raw_text = (fixtures_dir / "sample_faq_op05.txt").read_text(encoding="utf-8")
    pdf_path = _make_pdf_from_text(tmp_path / "qa_op05.pdf", raw_text, lines_per_page=11)

    parsed = parse_faq_pdf(pdf_path)

    assert parsed.source == "official_faq"
    assert parsed.date == "2025-09-01"
    # Rules version was not explicitly declared in the FAQ PDF -> must remain None.
    assert parsed.rules_version is None
    assert len(parsed.rulings) == 4

    q1, q2, q3, q4 = parsed.rulings

    # Q1: Card section OP05-119 + inline ST10-001 + explicit Rule 7-1-4-1.
    assert q1.ruling_id.startswith("faq:OP05-119:")
    assert q1.source == "official_faq"
    assert q1.authority == 2
    assert q1.reviewed is True
    assert q1.date == "2025-09-01"
    assert q1.rules_version is None
    assert q1.card_ids == ("OP05-119", "ST10-001")
    assert q1.rule_ids == ("7-1-4-1",)
    assert q1.raw_ref == "qa_op05.pdf#p1"
    assert "extra turn" in q1.question
    assert "returned 10 DON!!" in q1.answer

    # Q2: Standalone 2-column Q2/A2 layout; date 2025-09-01 & range "1-2" must NOT link as rules.
    assert q2.ruling_id.startswith("faq:OP05-119:")
    assert q2.card_ids == ("OP05-119",)
    assert q2.rule_ids == ()
    assert "1-2 cards" in q2.question
    assert "explicitly permits" in q2.answer

    # Q3: Event card header with "!!" (OP01-029 Radical Beam!!) + P-001/EB01-001/PRB01-001.
    assert q3.ruling_id.startswith("faq:OP01-029:")
    assert q3.card_ids == ("OP01-029", "P-001", "EB01-001", "PRB01-001")
    assert set(q3.rule_ids) == {"9-9-9-9", "10-1-1"}

    # Q4: General Rules Q&A resets section card -> general scope; "A Character..." stays in answer.
    assert q4.ruling_id.startswith("faq:general:")
    assert q4.card_ids == ()
    assert q4.rule_ids == ("7-1-4", "7-1-5")
    assert "A Character with [Blocker] must be rested" in q4.answer


def test_parse_errata_pdf_roundtrip(tmp_path: Path, fixtures_dir: Path) -> None:
    raw_text = (fixtures_dir / "sample_errata.txt").read_text(encoding="utf-8")
    pdf_path = _make_pdf_from_text(tmp_path / "card_errata_sheet.pdf", raw_text, lines_per_page=14)

    parsed = parse_faq_pdf(pdf_path)

    assert parsed.source == "errata"
    assert parsed.date == "2025-09-05"
    assert parsed.rules_version == "1.24"
    assert len(parsed.rulings) == 2

    e1, e2 = parsed.rulings

    assert e1.ruling_id.startswith("errata:OP01-016:")
    assert e1.source == "errata"
    assert e1.authority == 2
    assert e1.reviewed is True
    assert e1.date == "2023-03-10"
    assert e1.rules_version == "1.24"
    assert e1.card_ids == ("OP01-016",)
    assert "OP01-016 (Nami)" in e1.question
    assert "Before: [On Play] Look at 5 cards" in e1.answer
    assert "After: [On Play] Look at 5 cards" in e1.answer
    assert 'Note: Added "other than [Nami]"' in e1.answer

    assert e2.ruling_id.startswith("errata:ST10-001:")
    assert e2.date == "2025-09-05"
    assert e2.card_ids == ("ST10-001",)
    assert e2.rule_ids == ("7-1-2-1",)
    assert "ST10-001 (Trafalgar Law)" in e2.question
    assert "DON!! -3" in e2.answer


def test_directory_ingestion_and_conservative_rule_filtering(
    db_conn: sqlite3.Connection,
    tmp_path: Path,
    fixtures_dir: Path,
    sample_rules_text: str,
) -> None:
    # Populate Comprehensive Rules first so standalone deep rule IDs are verified against `rules`.
    ingest_rules(db_conn, parse_rules_text(sample_rules_text))

    faq_dir = tmp_path / "faq_bundle"
    faq_dir.mkdir()
    qa_text = (fixtures_dir / "sample_faq_op05.txt").read_text(encoding="utf-8")
    errata_text = (fixtures_dir / "sample_errata.txt").read_text(encoding="utf-8")
    _make_pdf_from_text(faq_dir / "01_qa_op05.pdf", qa_text, lines_per_page=14)
    (faq_dir / "02_errata.txt").write_text(errata_text, encoding="utf-8")

    docs = parse_faq_path(faq_dir)
    assert len(docs) == 2

    summary = ingest_faqs(db_conn, docs)
    assert summary.files_processed == 2
    assert summary.rulings_upserted == 6
    assert summary.official_faq_count == 4
    assert summary.errata_count == 2
    assert summary.chunks_upserted == 6

    # Verify conservative filtering: standalone bogus "9-9-9-9" is dropped when `rules` is loaded,
    # while real rule "10-1-1" (present in sample_rules_text) is linked.
    op01_029_ruling = db_conn.execute(
        "SELECT ruling_id FROM rulings WHERE ruling_id LIKE 'faq:OP01-029:%'"
    ).fetchone()
    assert op01_029_ruling is not None
    linked_rules = [
        row[0]
        for row in db_conn.execute(
            "SELECT rule_id FROM ruling_rules WHERE ruling_id = ? ORDER BY rule_id",
            (op01_029_ruling["ruling_id"],),
        ).fetchall()
    ]
    assert linked_rules == ["10-1-1"]

    # Chunk text and metadata match the domain contract.
    chunk_row = db_conn.execute(
        "SELECT text, authority, metadata FROM chunks WHERE doc_type = 'ruling' AND doc_id = ?",
        (op01_029_ruling["ruling_id"],),
    ).fetchone()
    assert chunk_row is not None
    assert chunk_row["authority"] == 2
    assert chunk_row["text"].startswith("[OFFICIAL FAQ] Cards: OP01-029")
    assert "Q: " in chunk_row["text"]
    assert "\nA: " in chunk_row["text"]
    metadata = json.loads(chunk_row["metadata"])
    assert metadata["rule_ids"] == ["10-1-1"]
    assert "OP01-029" in metadata["card_ids"]


def test_faq_ingestion_idempotency_preserves_superseded_and_embeddings(
    db_conn: sqlite3.Connection,
    fixtures_dir: Path,
) -> None:
    qa_text = (fixtures_dir / "sample_faq_op05.txt").read_text(encoding="utf-8")
    doc = parse_faq_text(qa_text, source_name="qa_op05.txt")
    embedder = HashEmbedder(dimension=64)

    first_summary = ingest_faqs(db_conn, doc)
    assert first_summary.rulings_upserted == 4

    embed_first = embed_chunks(db_conn, embedder)
    assert embed_first.embedded_chunks == 4

    # Mark Q1 as superseded by Q2 to verify re-ingestion never wipes superseded_by.
    q1_id = doc.rulings[0].ruling_id
    q2_id = doc.rulings[1].ruling_id
    with db_conn:
        db_conn.execute(
            "UPDATE rulings SET superseded_by = ? WHERE ruling_id = ?",
            (q2_id, q1_id),
        )

    counts_before = {
        table: db_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("rulings", "ruling_cards", "ruling_rules", "chunks", "chunks_fts")
    }

    # Re-ingest identical document.
    second_summary = ingest_faqs(db_conn, doc)
    assert second_summary.rulings_upserted == 4

    counts_after = {
        table: db_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("rulings", "ruling_cards", "ruling_rules", "chunks", "chunks_fts")
    }
    assert counts_after == counts_before

    # superseded_by is preserved.
    superseded_row = db_conn.execute(
        "SELECT superseded_by, reviewed FROM rulings WHERE ruling_id = ?",
        (q1_id,),
    ).fetchone()
    assert superseded_row["superseded_by"] == q2_id
    assert superseded_row["reviewed"] == 1

    # Unchanged chunks keep their embeddings (incremental embed skips all 4).
    embed_second = embed_chunks(db_conn, embedder)
    assert embed_second.embedded_chunks == 0
    assert embed_second.skipped_chunks == 4

    # Updating only the answer to Q2 keeps the SAME deterministic ruling_id & re-embeds only Q2.
    updated_text = qa_text.replace(
        "Only when the card text explicitly permits choosing up to that many targets.",
        "Only when the card text explicitly permits choosing up to two targets.",
    )
    updated_doc = parse_faq_text(updated_text, source_name="qa_op05.txt")
    assert updated_doc.rulings[1].ruling_id == q2_id

    ingest_faqs(db_conn, updated_doc)
    assert db_conn.execute("SELECT COUNT(*) FROM rulings").fetchone()[0] == 4

    embed_third = embed_chunks(db_conn, embedder)
    assert embed_third.embedded_chunks == 1
    assert embed_third.skipped_chunks == 3


def test_faq_integration_with_hybrid_search_and_get_card_rulings(
    db_conn: sqlite3.Connection,
    tmp_path: Path,
    fixtures_dir: Path,
    sample_rules_text: str,
) -> None:
    ingest_rules(db_conn, parse_rules_text(sample_rules_text))
    ingest_cards(db_conn, LocalCardSource(fixtures_dir / "sample_cards_punk_records.json"))

    qa_pdf = _make_pdf_from_text(
        tmp_path / "qa_op05.pdf",
        (fixtures_dir / "sample_faq_op05.txt").read_text(encoding="utf-8"),
    )
    errata_pdf = _make_pdf_from_text(
        tmp_path / "errata.pdf",
        (fixtures_dir / "sample_errata.txt").read_text(encoding="utf-8"),
    )
    ingest_faqs(db_conn, [parse_faq_pdf(qa_pdf), parse_faq_pdf(errata_pdf)])

    # Add a lower-authority judge ruling on OP05-119 to test authority ordering & filtering.
    insert_test_ruling(
        db_conn,
        ruling_id="judge-op05-119-extra",
        question="Can OP05-119 take an extra turn if 10 DON!! were returned?",
        answer="Judge note confirming extra turn timing.",
        source="judge",
        authority=3,
        date="2025-10-01",
        rules_version="1.24",
        reviewed=True,
        card_ids=["OP05-119"],
    )

    embedder = HashEmbedder(dimension=64)
    embed_chunks(db_conn, embedder)

    # get_card_rulings returns official FAQ (authority=2) before judge ruling (authority=3).
    luffy_rulings = get_card_rulings(db_conn, "OP05-119")
    assert luffy_rulings["card_found"] is True
    assert luffy_rulings["total"] == 3
    authorities = [r["authority"] for r in luffy_rulings["rulings"]]
    assert authorities == [2, 2, 3]
    assert luffy_rulings["rulings"][0]["source"] == "official_faq"

    nami_rulings = get_card_rulings(db_conn, "OP01-016")
    assert nami_rulings["total"] == 1
    assert nami_rulings["rulings"][0]["source"] == "errata"
    assert nami_rulings["rulings"][0]["authority"] == 2

    # hybrid_search surfaces official FAQ with exact_card_ruling & respects min_authority=2.
    search_resp = hybrid_search(
        db_conn,
        "OP05-119 extra turn",
        k=8,
        doc_types=["ruling"],
        min_authority=2,
        embedder=embedder,
    )
    assert search_resp["results"]
    assert all(item["authority"] <= 2 for item in search_resp["results"])
    top = search_resp["results"][0]
    assert top["doc_type"] == "ruling"
    assert top["source"] == "official_faq"
    assert "exact_card_ruling" in top["match_signals"]


def test_cli_faq_command_offline(
    tmp_path: Path,
    fixtures_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "cli_faq.db"
    faq_dir = tmp_path / "faq_inputs"
    faq_dir.mkdir()
    _make_pdf_from_text(
        faq_dir / "qa_op05.pdf",
        (fixtures_dir / "sample_faq_op05.txt").read_text(encoding="utf-8"),
    )

    exit_code = cli_main(
        [
            "--db",
            str(db_path),
            "faq",
            str(faq_dir),
            "--rules-version",
            "1.25",
            "--url",
            "https://example.invalid/official-faq",
        ]
    )
    assert exit_code == 0

    stdout = capsys.readouterr().out
    assert "Ingested 4 official rulings from 1 file(s)" in stdout
    assert "official_faq=4" in stdout

    with open_db(db_path, init=False) as conn:
        row = conn.execute(
            "SELECT rules_version, url, authority, reviewed FROM rulings LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["rules_version"] == "1.25"
        assert row["url"] == "https://example.invalid/official-faq"
        assert row["authority"] == 2
        assert row["reviewed"] == 1
        meta_ts = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_faq_ingested_at'"
        ).fetchone()
        assert meta_ts is not None


def test_faq_parser_edge_cases(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_faq"
    empty_dir.mkdir()
    (empty_dir / ".gitkeep").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No official FAQ/errata files"):
        parse_faq_path(empty_dir)

    unsupported = tmp_path / "notes.csv"
    unsupported.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported FAQ file type"):
        parse_faq_path(unsupported)

    # Identical question repeated under the same card gets distinct deterministic IDs.
    dup_text = """
OP05-119 Monkey.D.Luffy
Q1. Can I activate this effect?
A1. Yes, during your Main Phase.
Q2. Can I activate this effect?
A2. No, if you have fewer than 10 DON!! cards.
"""
    parsed = parse_faq_text(dup_text, source_name="dup.txt")
    assert len(parsed.rulings) == 2
    assert parsed.rulings[0].ruling_id != parsed.rulings[1].ruling_id
    # Re-parsing produces the exact same IDs deterministically.
    parsed_again = parse_faq_text(dup_text, source_name="dup.txt")
    assert [r.ruling_id for r in parsed_again.rulings] == [r.ruling_id for r in parsed.rulings]


def _make_bandai_table_pdf(
    pdf_path: Path,
    rows: list[list[str]],
    col_widths: list[float],
    *,
    mod_date: str = "D:20250226182404+09'00'",
) -> Path:
    """Create a synthetic PDF matching Bandai's official Excel-exported table grid."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    x_positions = [20.0]
    for w in col_widths:
        x_positions.append(x_positions[-1] + w)
    row_height = 60.0
    y_top = 40.0

    for row_idx, row in enumerate(rows):
        y0 = y_top + row_idx * row_height
        y1 = y0 + row_height
        for col_idx, cell_text in enumerate(row):
            rect = pymupdf.Rect(x_positions[col_idx], y0, x_positions[col_idx + 1], y1)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            inner = pymupdf.Rect(rect.x0 + 3, rect.y0 + 3, rect.x1 - 3, rect.y1 - 3)
            page.insert_textbox(inner, cell_text, fontsize=8)

    doc.set_metadata({"modDate": mod_date})
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_parse_bandai_tabular_card_and_rules_pdfs(tmp_path: Path) -> None:
    card_pdf = _make_bandai_table_pdf(
        tmp_path / "qa_op05.pdf",
        rows=[
            ["Card No.", "Card Name", "Question", "Answer"],
            [
                "OP05-001",
                "Sabo",
                "Can I use this [Opponent's Turn] effect if ST10-001 is on the field?",
                "Yes, you can.<br>See Rule 7-1-2-1 for timing.",
            ],
            ["EB04-038", "Rosinante & Law", "", ""],
        ],
        col_widths=[65.0, 90.0, 200.0, 200.0],
        mod_date="D:20240403182748+09'00'",
    )
    parsed_cards = parse_faq_pdf(card_pdf)
    assert parsed_cards.source == "official_faq"
    assert parsed_cards.date == "2024-04-03"
    assert parsed_cards.rules_version is None
    # Blank placeholder row (EB04-038) is skipped cleanly.
    assert len(parsed_cards.rulings) == 1
    r0 = parsed_cards.rulings[0]
    assert r0.ruling_id.startswith("faq:OP05-001:")
    assert r0.card_ids == ("OP05-001", "ST10-001")
    assert r0.rule_ids == ("7-1-2-1",)
    assert "<br>" not in r0.answer
    assert r0.answer == "Yes, you can.\nSee Rule 7-1-2-1 for timing."
    assert r0.raw_ref == "qa_op05.pdf#p1"

    rules_pdf = _make_bandai_table_pdf(
        tmp_path / "qa_rules.pdf",
        rows=[
            ["Category", "Question", "Answer"],
            [
                "Basic Rules",
                "What happens when a Character's power becomes 0?",
                "The Character remains on the field.<br>It is not K.O.'d by having 0 power.",
            ],
        ],
        col_widths=[95.0, 230.0, 230.0],
        mod_date="D:20250226182404+09'00'",
    )
    parsed_rules = parse_faq_pdf(rules_pdf)
    assert parsed_rules.source == "official_faq"
    assert parsed_rules.date == "2025-02-26"
    assert len(parsed_rules.rulings) == 1
    gr = parsed_rules.rulings[0]
    assert gr.ruling_id.startswith("faq:general:")
    assert gr.card_ids == ()
    assert "<br>" not in gr.answer
    assert "\n" in gr.answer


def test_parse_bandai_errata_html(tmp_path: Path) -> None:
    sample_html = """<!doctype html>
<html><body>
  <section class="contentsLCol">
    <h4 class="mediumTit">May 16, 2025</h4>
    <div class="detailCol mtS" id="errata_15">
      <h5 class="smallTitRed">OP07-097 Vegapunk</h5>
      <dl>
        <dt>Before:</dt>
        <dd>This Leader cannot attack.<br>[Activate: Main] Typo text.</dd>
        <dt>After:</dt>
        <dd>This Leader cannot attack.<br>[Activate: Main] Corrected text.</dd>
      </dl>
    </div>
    <ul><li>
      <div class="errataModal" id="nov_11_2022_OP01-002">
        <h5 class="smallTitRed">February 17, 2023<br>OP01-002 Trafalgar Law</h5>
        <dl>
          <dt>Note:</dt>
          <dd>Errata will be corrected in reprints.</dd>
          <dt>Before:</dt>
          <dd>Old Law effect.</dd>
          <dt>After:</dt>
          <dd>New Law effect.</dd>
        </dl>
      </div>
    </li></ul>
    <div class="dateText"><p>Effective as of September 30, 2022</p></div>
    <section id="errata_01" class="detailCol">
      <p class="cardNum">ST04-001</p>
      <h6 class="cardName">Kaido</h6>
      <div class="impNoticeBox">
        <dt>After the rule revision, this card will be treated as having the following effect.</dt>
        <dd><p>[Activate: Main] DON!! -7: Trash up to 1 Life card.</p></dd>
      </div>
    </section>
  </section>
</body></html>"""

    parsed = parse_faq_html(sample_html, source_name="errata_card.html")
    assert parsed.source == "errata"
    assert parsed.date == "2025-05-16"
    assert len(parsed.rulings) == 3

    r1, r2, r3 = parsed.rulings
    assert r1.ruling_id.startswith("errata:OP07-097:")
    assert r1.date == "2025-05-16"
    assert r1.card_ids == ("OP07-097",)
    assert r1.raw_ref == "errata_card.html#errata_15"
    assert "Before: This Leader cannot attack.\n[Activate: Main] Typo text." in r1.answer
    assert "After: This Leader cannot attack.\n[Activate: Main] Corrected text." in r1.answer

    assert r2.ruling_id.startswith("errata:OP01-002:")
    assert r2.date == "2023-02-17"
    assert r2.card_ids == ("OP01-002",)
    assert "Note: Errata will be corrected in reprints." in r2.answer

    assert r3.ruling_id.startswith("errata:ST04-001:")
    assert r3.date == "2022-09-30"
    assert r3.card_ids == ("ST04-001",)
    assert "ST04-001 (Kaido)" in r3.question
    assert "After: [Activate: Main] DON!! -7" in r3.answer
    assert "Note: After the rule revision" in r3.answer

    # Also verify parse_faq_path handles .html files cleanly.
    html_file = tmp_path / "errata_card.html"
    html_file.write_text(sample_html, encoding="utf-8")
    docs = parse_faq_path(html_file)
    assert len(docs) == 1
    assert len(docs[0].rulings) == 3
