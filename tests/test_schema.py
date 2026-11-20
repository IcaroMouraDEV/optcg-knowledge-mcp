"""Tests for SQLite schema, FTS5 sync triggers, and sqlite-vec integration."""

from __future__ import annotations

import sqlite3

import pytest

from optcg_knowledge.db import (
    SCHEMA_VERSION,
    authority_for_source,
    connect,
    ensure_vec_table,
    get_meta,
    init_schema,
    migrate_schema,
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
        "trm_sections",
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


def test_migrate_v2_to_v3_preserves_chunk_rowids_fts_and_vectors(tmp_path) -> None:
    db_path = tmp_path / "legacy_v2.db"
    embedder = HashEmbedder(dimension=64)

    with connect(db_path) as conn:
        # Construct an authentic schema v2 DB (no trm_sections, old chunks CHECK constraint).
        conn.executescript(
            """
            CREATE TABLE rules (
                rule_id TEXT PRIMARY KEY,
                parent_id TEXT,
                section TEXT NOT NULL,
                text TEXT NOT NULL,
                rules_version TEXT,
                effective_date TEXT
            );
            CREATE TABLE rulings (
                ruling_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT NOT NULL CHECK (
                    source IN ('rules', 'official_faq', 'errata', 'judge', 'community')
                ),
                authority INTEGER NOT NULL CHECK (authority IN (1, 2, 3, 4)),
                date TEXT,
                rules_version TEXT,
                url TEXT,
                superseded_by TEXT,
                reviewed INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0, 1)),
                raw_ref TEXT,
                judge_tier TEXT CHECK (
                    judge_tier IS NULL OR judge_tier IN ('tier_0', 'tier_2', 'tier_1')
                )
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL CHECK (doc_type IN ('rule', 'ruling', 'card')),
                doc_id TEXT NOT NULL,
                text TEXT NOT NULL,
                authority INTEGER NOT NULL CHECK (authority IN (1, 2, 3, 4)),
                metadata TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL,
                embedded_hash TEXT,
                embedded_model TEXT
            );
            CREATE UNIQUE INDEX idx_chunks_doc ON chunks(doc_type, doc_id);
            CREATE INDEX idx_chunks_authority ON chunks(authority);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                text, content='chunks', content_rowid='rowid', tokenize='unicode61'
            );
            CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
            END;
            CREATE TRIGGER chunks_au AFTER UPDATE OF text ON chunks
            WHEN old.text IS NOT new.text BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text)
                VALUES ('delete', old.rowid, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta (key, value) VALUES ('schema_version', '2');
            """
        )
        with conn:
            ensure_vec_table(conn, embedding_dim=64, embedding_model=embedder.model_name)
            # Insert 3 chunks and delete middle one so surviving rowids have a gap (1 and 3).
            for cid, doc_id, text in [
                ("rule:1-1", "1-1", "1-1. Fundamental game rule about Leader cards."),
                ("rule:temp", "temp", "Temporary deleted chunk."),
                ("card:OP05-119", "OP05-119", "OP05-119 Monkey.D.Luffy extra turn."),
            ]:
                chash = compute_content_hash(text)
                conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_type, doc_id, text, authority, metadata,
                        content_hash, embedded_hash, embedded_model
                    ) VALUES (?, ?, ?, ?, 1, '{}', ?, ?, ?)
                    """,
                    (
                        cid,
                        "rule" if cid.startswith("rule:") else "card",
                        doc_id,
                        text,
                        chash,
                        chash,
                        embedder.model_name,
                    ),
                )
            conn.execute("DELETE FROM chunks WHERE chunk_id = 'rule:temp'")
            rows = conn.execute(
                "SELECT rowid, chunk_id, text FROM chunks ORDER BY rowid"
            ).fetchall()
            pre_rowids = [(int(r["rowid"]), str(r["chunk_id"])) for r in rows]
            assert pre_rowids == [(1, "rule:1-1"), (3, "card:OP05-119")]
            vectors = embedder.embed([str(r["text"]) for r in rows])
            for row, vec in zip(rows, vectors, strict=True):
                conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (row["rowid"], serialize_vector(vec)),
                )

        # Confirm v2 rejects tournament_rule before migration.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_type, doc_id, text, authority, content_hash)
                VALUES ('trm:TRM-1.0', 'tournament_rule', 'TRM-1.0', 'text', 1, 'abc')
                """
            )

        # Apply v2 -> v3 migration.
        with conn:
            migrated = migrate_schema(conn)
        assert migrated is True
        assert get_meta(conn, "schema_version") == SCHEMA_VERSION

        # Exact rowids preserved -> existing chunks_vec KNN search still resolves accurately!
        post_rowids = [
            (int(r["rowid"]), str(r["chunk_id"]))
            for r in conn.execute("SELECT rowid, chunk_id FROM chunks ORDER BY rowid").fetchall()
        ]
        assert post_rowids == [(1, "rule:1-1"), (3, "card:OP05-119")]

        qvec = serialize_vector(embedder.embed_query("Monkey.D.Luffy extra turn"))
        knn = conn.execute(
            """
            SELECT c.chunk_id
            FROM chunks_vec v
            JOIN chunks c ON c.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = 2
            ORDER BY v.distance ASC
            """,
            (qvec,),
        ).fetchall()
        assert knn[0]["chunk_id"] == "card:OP05-119"

        # FTS5 stays intact and tournament_rule chunks are now accepted.
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
        with conn:
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, doc_type, doc_id, text, authority, content_hash)
                VALUES ('trm:TRM-7.4.1', 'tournament_rule', 'TRM-7.4.1', 'Game Play Error', 1, 'h1')
                """
            )
        fts_hits = conn.execute(
            """
            SELECT c.chunk_id
            FROM chunks_fts f
            JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH '"Game" AND "Play"'
            """
        ).fetchall()
        assert [r["chunk_id"] for r in fts_hits] == ["trm:TRM-7.4.1"]

        # Running migrate_schema again is a clean no-op.
        with conn:
            assert migrate_schema(conn) is False
