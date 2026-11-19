"""End-to-end CLI tests for optcg-ingest."""

from __future__ import annotations

from pathlib import Path

import pytest

from optcg_knowledge.cli import main
from optcg_knowledge.db import open_db


def test_cli_init_rules_and_cards_end_to_end(
    tmp_path: Path,
    sample_rules_pdf: Path,
    fixtures_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "cli_test.db"

    assert main(["--db", str(db_path), "init-db"]) == 0
    assert "Initialized database" in capsys.readouterr().out

    assert main(["--db", str(db_path), "rules", str(sample_rules_pdf)]) == 0
    rules_out = capsys.readouterr().out
    assert "Ingested" in rules_out
    assert "version=1.24" in rules_out

    cards_file = fixtures_dir / "sample_cards_punk_records.json"
    assert (
        main(["--db", str(db_path), "cards", "--source", "local", "--path", str(cards_file)]) == 0
    )
    cards_out = capsys.readouterr().out
    assert "Ingested 5 canonical cards" in cards_out

    with open_db(db_path) as conn:
        rules_count = conn.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
        cards_count = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
        chunks_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    assert rules_count > 0
    assert cards_count == 5
    assert chunks_count == rules_count + cards_count
