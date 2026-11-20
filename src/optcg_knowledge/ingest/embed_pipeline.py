"""Incremental local embedding pipeline for SQLite chunks + sqlite-vec."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

from optcg_knowledge.db.schema import ensure_vec_table, set_meta, utc_now_iso
from optcg_knowledge.ingest.embeddings import Embedder, serialize_vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbedSummary:
    """Counts and metadata produced by an incremental embedding run."""

    total_chunks: int
    embedded_chunks: int
    skipped_chunks: int
    orphans_removed: int
    rebuilt_vec_table: bool
    model_name: str
    dimension: int


def embed_chunks(
    conn: sqlite3.Connection,
    embedder: Embedder,
    *,
    batch_size: int = 8,
    force: bool = False,
) -> EmbedSummary:
    """Incrementally embed new or changed chunks into sqlite-vec.

    Skips chunks whose ``embedded_hash == content_hash`` and ``embedded_model == model_name``
    unless ``force=True`` or ``chunks_vec`` was rebuilt due to a model/dimension change.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    with conn:
        rebuilt = ensure_vec_table(
            conn,
            embedding_dim=embedder.dimension,
            embedding_model=embedder.model_name,
        )
        orphan_cursor = conn.execute(
            "DELETE FROM chunks_vec WHERE rowid NOT IN (SELECT rowid FROM chunks)"
        )
        orphans_removed = orphan_cursor.rowcount if orphan_cursor.rowcount > 0 else 0

    total_row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    total_chunks = int(total_row[0]) if total_row is not None else 0

    if force:
        pending_rows = conn.execute(
            "SELECT rowid, text, content_hash FROM chunks ORDER BY rowid"
        ).fetchall()
    else:
        pending_rows = conn.execute(
            """
            SELECT rowid, text, content_hash
            FROM chunks
            WHERE embedded_hash IS NULL
               OR embedded_hash != content_hash
               OR embedded_model IS NULL
               OR embedded_model != ?
            ORDER BY rowid
            """,
            (embedder.model_name,),
        ).fetchall()

    embedded_count = 0
    total_pending = len(pending_rows)
    if total_pending > 0:
        logger.info(
            "Embedding %d/%d pending chunks with %s (dim=%d, batch_size=%d)",
            total_pending,
            total_chunks,
            embedder.model_name,
            embedder.dimension,
            batch_size,
        )

    active_dim = embedder.dimension
    started_at = time.monotonic()
    log_every = max(batch_size * 10, 64)
    next_log_at = log_every

    for start in range(0, total_pending, batch_size):
        batch = pending_rows[start : start + batch_size]
        texts = [str(row["text"]) for row in batch]
        vectors = embedder.embed(texts)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(batch)} input chunks"
            )

        first_dim = len(vectors[0]) if vectors else active_dim
        if start == 0 and first_dim != active_dim:
            with conn:
                rebuilt = (
                    ensure_vec_table(
                        conn,
                        embedding_dim=first_dim,
                        embedding_model=embedder.model_name,
                    )
                    or rebuilt
                )
            active_dim = first_dim

        with conn:
            for row, vector in zip(batch, vectors, strict=True):
                if len(vector) != active_dim:
                    raise ValueError(
                        f"Expected vector dimension {active_dim}, got {len(vector)}"
                    )
                rowid = int(row["rowid"])
                content_hash = str(row["content_hash"])
                conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
                conn.execute(
                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, serialize_vector(vector)),
                )
                conn.execute(
                    """
                    UPDATE chunks
                    SET embedded_hash = ?, embedded_model = ?
                    WHERE rowid = ?
                    """,
                    (content_hash, embedder.model_name, rowid),
                )
        embedded_count += len(batch)

        if embedded_count >= next_log_at or embedded_count == total_pending:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            rate = embedded_count / elapsed
            pct = (embedded_count / total_pending) * 100.0
            logger.info(
                "Embedding progress: %d/%d chunks (%.1f%%, %.1f chunks/s)",
                embedded_count,
                total_pending,
                pct,
                rate,
            )
            next_log_at = embedded_count + log_every

    with conn:
        set_meta(conn, "last_embed_at", utc_now_iso())

    return EmbedSummary(
        total_chunks=total_chunks,
        embedded_chunks=embedded_count,
        skipped_chunks=total_chunks - embedded_count,
        orphans_removed=orphans_removed,
        rebuilt_vec_table=rebuilt,
        model_name=embedder.model_name,
        dimension=active_dim,
    )
