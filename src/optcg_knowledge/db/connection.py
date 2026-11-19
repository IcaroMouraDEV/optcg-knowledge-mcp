"""SQLite connection management with sqlite-vec extension loading."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import sqlite_vec

from optcg_knowledge.config import DEFAULT_EMBEDDING_MODEL, resolve_embedding_dim
from optcg_knowledge.db.schema import init_schema


def connect(
    db_path: Path | str,
    *,
    init: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dim: int | None = None,
) -> sqlite3.Connection:
    """Open a SQLite connection configured with WAL, foreign keys, and sqlite-vec."""
    path_str = str(db_path)
    if path_str != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if path_str != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")

    if not hasattr(conn, "enable_load_extension"):
        conn.close()
        raise RuntimeError(
            "This Python interpreter was built without SQLite loadable extension support "
            "(missing sqlite3.Connection.enable_load_extension). "
            "Recreate the virtualenv with a uv-managed Python 3.12 (`uv sync --reinstall`)."
        )

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    if init:
        resolved_dim = resolve_embedding_dim(embedding_model, override=embedding_dim)
        init_schema(conn, embedding_dim=resolved_dim, embedding_model=embedding_model)

    return conn


@contextmanager
def open_db(
    db_path: Path | str,
    *,
    init: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dim: int | None = None,
) -> Iterator[sqlite3.Connection]:
    """Context manager wrapping connect() and guaranteeing connection cleanup."""
    conn = connect(
        db_path,
        init=init,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    try:
        yield conn
    finally:
        conn.close()
