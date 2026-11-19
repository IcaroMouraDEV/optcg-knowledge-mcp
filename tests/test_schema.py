"""Tests for SQLite schema, FTS5 sync triggers, and sqlite-vec integration."""

from __future__ import annotations

import sqlite3

import pytest

from optcg_knowledge.db import (
    SCHEMA_VERSION,
    authority_for_source,
    ensure_vec_table,
    get_meta,
    init_schema,
)
from optcg_knowledge.db.models import ChunkRecord
from optcg_knowledge.ingest.chunking import compute_content_hash, upsert_chunks
from optcg_knowledge.ingest.embeddings import HashEmbedder, serialize_vector


def test_schema_creates_expected_tables_and_meta(db_conn: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    expected = {
        "rules",
        "cards",
        "rulings",
        "ruling_cards",
        "ruling_rules",
        "chunks",
        "chunks_fts",
        "chunks_vec",
        "meta",
    }
    assert expected.issubset(tables)
    assert get_meta(db_conn, "schema_version") == SCHEMA_VERSION
    assert get_meta(db_conn, "embedding_model") == "test/hash-embedder"
    assert get_meta(db_conn, "embedding_dim") == "64"


def test_schema_init_is_idempotent(db_conn: sqlite3.Connection) -> None:
    init_schema(db_conn, embedding_dim=64, embedding_model="test/hash-embedder")
    init_schema(db_conn, embedding_dim=64, embedding_model="test/hash-embedder")
    assert get_meta(db_conn, "schema_version") == SCHEMA_VERSION


def test_authority_hierarchy_and_constraints(db_conn: sqlite3.Connection) -> None:
    assert authority_for_source("rules") == 1
    assert authority_for_source("official_faq") == 2
    assert authority_for_source("errata") == 2
    assert authority_for_source("judge") == 3
    assert authority_for_source("community") == 4

    with pytest.raises(ValueError):
        authority_for_source("unknown_source")

    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            """
            INSERT INTO rulings (ruling_id, question, answer, source, authority)
            VALUES ('r1', 'Q?', 'A.', 'judge', 99)
            """
        )


def test_fts5_triggers_stay_in_sync_on_insert_update_delete(db_conn: sqlite3.Connection) -> None:
    chunk_v1 = ChunkRecord(
        chunk_id="rule:7-1-4-1",
        doc_type="rule",
        doc_id="7-1-4-1",
        text="7-1-4-1. Damage is dealt to the opposing Leader.",
        authority=1,
        metadata={"rule_id": "7-1-4-1"},
        content_hash=compute_content_hash("7-1-4-1. Damage is dealt to the opposing Leader."),
    )
    with db_conn:
        upsert_chunks(db_conn, [chunk_v1])

    hits = db_conn.execute(
        """
        SELECT c.chunk_id
        FROM chunks_fts f
        JOIN chunks c ON c.rowid = f.rowid
        WHERE chunks_fts MATCH '"Leader"'
        """
    ).fetchall()
    assert [row["chunk_id"] for row in hits] == ["rule:7-1-4-1"]

    # Update text and verify old term is gone while new term matches.
    updated_text = "7-1-4-1. Blocker redirects the attack."
    chunk_v2 = ChunkRecord(
        chunk_id="rule:7-1-4-1",
        doc_type="rule",
        doc_id="7-1-4-1",
        text=updated_text,
        authority=1,
        metadata={"rule_id": "7-1-4-1"},
        content_hash=compute_content_hash(updated_text),
    )
    with db_conn:
        upsert_chunks(db_conn, [chunk_v2])

    old_hits = db_conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH '\"Leader\"'"
    ).fetchall()
    new_hits = db_conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH '\"Blocker\"'"
    ).fetchall()
    assert old_hits == []
    assert len(new_hits) == 1

    # Delete and verify FTS5 integrity check passes.
    with db_conn:
        db_conn.execute("DELETE FROM chunks WHERE chunk_id = 'rule:7-1-4-1'")

    db_conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")


def test_sqlite_vec_insert_knn_and_rebuild_on_dim_change(db_conn: sqlite3.Connection) -> None:
    embedder = HashEmbedder(dimension=64)
    blocker_text = "[Blocker] redirects an attack during the Block Step."
    luffy_text = "OP05-119 Monkey.D.Luffy take an extra turn."
    chunks = [
        ChunkRecord(
            chunk_id="rule:10-1-1",
            doc_type="rule",
            doc_id="10-1-1",
            text=blocker_text,
            authority=1,
            metadata={},
            content_hash=compute_content_hash(blocker_text),
        ),
        ChunkRecord(
            chunk_id="card:OP05-119",
            doc_type="card",
            doc_id="OP05-119",
            text=luffy_text,
            authority=1,
            metadata={},
            content_hash=compute_content_hash(luffy_text),
        ),
    ]
    with db_conn:
        upsert_chunks(db_conn, chunks)
        rows = db_conn.execute("SELECT rowid, chunk_id, text FROM chunks ORDER BY rowid").fetchall()
        vectors = embedder.embed([row["text"] for row in rows])
        for row, vec in zip(rows, vectors, strict=True):
            db_conn.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (row["rowid"], serialize_vector(vec)),
            )
            db_conn.execute(
                """
                UPDATE chunks
                SET embedded_hash = content_hash, embedded_model = ?
                WHERE rowid = ?
                """,
                (embedder.model_name, row["rowid"]),
            )

    query_vec = serialize_vector(embedder.embed(["[Blocker] Block Step"])[0])
    results = db_conn.execute(
        """
        SELECT c.chunk_id, v.distance
        FROM chunks_vec v
        JOIN chunks c ON c.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = 2
        ORDER BY v.distance
        """,
        (query_vec,),
    ).fetchall()
    assert results[0]["chunk_id"] == "rule:10-1-1"

    # Switching embedding model/dim rebuilds chunks_vec and invalidates embedded_hash.
    with db_conn:
        rebuilt = ensure_vec_table(db_conn, embedding_dim=128, embedding_model="test/new-model")
    assert rebuilt is True
    assert get_meta(db_conn, "embedding_dim") == "128"
    assert get_meta(db_conn, "embedding_model") == "test/new-model"
    embedded_count = db_conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded_hash IS NOT NULL"
    ).fetchone()[0]
    assert embedded_count == 0
