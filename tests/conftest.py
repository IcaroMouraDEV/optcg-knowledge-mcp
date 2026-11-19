"""Shared pytest fixtures for offline optcg-knowledge tests."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest

from optcg_knowledge.db import connect

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def sample_rules_text() -> str:
    return (FIXTURES_DIR / "sample_rules_text.txt").read_text(encoding="utf-8")


@pytest.fixture()
def sample_rules_pdf(tmp_path: Path, sample_rules_text: str) -> Path:
    """Create a real multi-page PDF on disk from the sample rules text."""
    pdf_path = tmp_path / "rule_comprehensive.pdf"
    doc = pymupdf.open()
    lines = sample_rules_text.splitlines()
    lines_per_page = 14
    for start in range(0, len(lines), lines_per_page):
        page = doc.new_page(width=595, height=842)
        chunk = "\n".join(lines[start : start + lines_per_page])
        page.insert_textbox(pymupdf.Rect(36, 36, 559, 806), chunk, fontsize=10)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture()
def db_conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Provide an initialized temporary SQLite database with FTS5 + sqlite-vec."""
    db_path = tmp_path / "test_optcg.db"
    conn = connect(db_path, init=True, embedding_model="test/hash-embedder", embedding_dim=64)
    try:
        yield conn
    finally:
        conn.close()
