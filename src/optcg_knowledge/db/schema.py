"""SQLite schema definition with FTS5 and sqlite-vec support."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

SCHEMA_VERSION = "1"

BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rules (
    rule_id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES rules(rule_id) ON DELETE SET NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    rules_version TEXT,
    effective_date TEXT
);

CREATE INDEX IF NOT EXISTS idx_rules_parent_id ON rules(parent_id);
CREATE INDEX IF NOT EXISTS idx_rules_section ON rules(section);

CREATE TABLE IF NOT EXISTS cards (
    card_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    set_code TEXT NOT NULL,
    category TEXT NOT NULL,
    colors TEXT NOT NULL DEFAULT '[]',
    cost INTEGER,
    life INTEGER,
    power INTEGER,
    counter INTEGER,
    attributes TEXT NOT NULL DEFAULT '[]',
    traits TEXT NOT NULL DEFAULT '[]',
    effect_text TEXT,
    trigger_text TEXT,
    rarity TEXT,
    image_url TEXT,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_cards_set_code ON cards(set_code);
CREATE INDEX IF NOT EXISTS idx_cards_category ON cards(category);

CREATE TABLE IF NOT EXISTS rulings (
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
    superseded_by TEXT REFERENCES rulings(ruling_id) ON DELETE SET NULL,
    reviewed INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0, 1)),
    raw_ref TEXT
);

CREATE INDEX IF NOT EXISTS idx_rulings_authority ON rulings(authority);
CREATE INDEX IF NOT EXISTS idx_rulings_superseded_by ON rulings(superseded_by);
CREATE INDEX IF NOT EXISTS idx_rulings_reviewed ON rulings(reviewed);

CREATE TABLE IF NOT EXISTS ruling_cards (
    ruling_id TEXT NOT NULL REFERENCES rulings(ruling_id) ON DELETE CASCADE,
    card_id TEXT NOT NULL,
    PRIMARY KEY (ruling_id, card_id)
);

CREATE INDEX IF NOT EXISTS idx_ruling_cards_card_id ON ruling_cards(card_id);

CREATE TABLE IF NOT EXISTS ruling_rules (
    ruling_id TEXT NOT NULL REFERENCES rulings(ruling_id) ON DELETE CASCADE,
    rule_id TEXT NOT NULL,
    PRIMARY KEY (ruling_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_ruling_rules_rule_id ON ruling_rules(rule_id);

CREATE TABLE IF NOT EXISTS chunks (
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

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_type, doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_authority ON chunks(authority);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks
WHEN old.text IS NOT new.text
BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Read a metadata value by key, returning None when absent."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return str(row[0]) if row is not None else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update a key/value entry in the meta table."""
    conn.execute(
        """
        INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _create_vec_table(conn: sqlite3.Connection, embedding_dim: int) -> None:
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(f"Invalid embedding_dim: {embedding_dim!r}")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
            embedding float[{embedding_dim}] distance_metric=cosine
        )
        """
    )


def ensure_vec_table(
    conn: sqlite3.Connection,
    *,
    embedding_dim: int,
    embedding_model: str,
) -> bool:
    """Ensure chunks_vec matches the configured model/dimension.

    Returns True if chunks_vec was rebuilt due to a dimension or model change.
    """
    existing_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'"
    ).fetchone()
    recorded_dim = get_meta(conn, "embedding_dim")
    recorded_model = get_meta(conn, "embedding_model")

    needs_rebuild = existing_table is not None and (
        (recorded_dim is not None and int(recorded_dim) != embedding_dim)
        or (recorded_model is not None and recorded_model != embedding_model)
    )

    if needs_rebuild:
        conn.execute("DROP TABLE IF EXISTS chunks_vec")
        conn.execute("UPDATE chunks SET embedded_hash = NULL, embedded_model = NULL")

    _create_vec_table(conn, embedding_dim)
    set_meta(conn, "embedding_dim", str(embedding_dim))
    set_meta(conn, "embedding_model", embedding_model)
    return needs_rebuild


def init_schema(
    conn: sqlite3.Connection,
    *,
    embedding_dim: int = 1024,
    embedding_model: str = "BAAI/bge-m3",
) -> None:
    """Idempotently initialize all SQLite tables, FTS5 triggers, sqlite-vec, and metadata."""
    conn.executescript(BASE_SCHEMA_SQL)
    with conn:
        set_meta(conn, "schema_version", SCHEMA_VERSION)
        ensure_vec_table(
            conn,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
        )
