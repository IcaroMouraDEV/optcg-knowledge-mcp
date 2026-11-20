"""Offline tests for incremental local embeddings and `optcg-ingest embed`."""

from __future__ import annotations

import builtins
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.db import ChunkRecord, connect, get_meta
from optcg_knowledge.ingest import (
    HashEmbedder,
    SentenceTransformerEmbedder,
    compute_content_hash,
    embed_chunks,
    ingest_rules,
    parse_rules_text,
    upsert_chunks,
)


def _count_vec_rows(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()
    return int(row[0]) if row is not None else 0


def test_embed_chunks_is_incremental_and_tracks_content_hash(
    db_conn: sqlite3.Connection,
    sample_rules_text: str,
) -> None:
    ingest_rules(db_conn, parse_rules_text(sample_rules_text))
    embedder = HashEmbedder(dimension=64)

    first = embed_chunks(db_conn, embedder, batch_size=3)
    assert first.total_chunks > 0
    assert first.embedded_chunks == first.total_chunks
    assert first.skipped_chunks == 0
    assert first.rebuilt_vec_table is False
    assert _count_vec_rows(db_conn) == first.total_chunks
    assert get_meta(db_conn, "last_embed_at") is not None

    # Second run skips everything because content_hash and model match.
    second = embed_chunks(db_conn, embedder, batch_size=3)
    assert second.embedded_chunks == 0
    assert second.skipped_chunks == first.total_chunks

    # Mutate exactly one chunk's text -> only that chunk gets re-embedded.
    updated_text = "7-1-4-1. Updated damage step rule text for incremental re-embedding."
    with db_conn:
        upsert_chunks(
            db_conn,
            [
                ChunkRecord(
                    chunk_id="rule:7-1-4-1",
                    doc_type="rule",
                    doc_id="7-1-4-1",
                    text=updated_text,
                    authority=1,
                    metadata={"rule_id": "7-1-4-1"},
                    content_hash=compute_content_hash(updated_text),
                )
            ],
        )

    third = embed_chunks(db_conn, embedder, batch_size=3)
    assert third.embedded_chunks == 1
    assert third.skipped_chunks == first.total_chunks - 1
    assert _count_vec_rows(db_conn) == first.total_chunks

    # Force re-embeds all chunks even when unchanged.
    forced = embed_chunks(db_conn, embedder, batch_size=4, force=True)
    assert forced.embedded_chunks == first.total_chunks
    assert forced.skipped_chunks == 0


def test_embed_chunks_cleans_orphans_and_rebuilds_on_model_change(
    db_conn: sqlite3.Connection,
    sample_rules_text: str,
) -> None:
    ingest_rules(db_conn, parse_rules_text(sample_rules_text))
    embedder_v1 = HashEmbedder(dimension=64, model_name="test/model-v1")
    first = embed_chunks(db_conn, embedder_v1, batch_size=5)
    assert _count_vec_rows(db_conn) == first.total_chunks

    # Delete a chunk directly -> its chunks_vec row becomes an orphan.
    with db_conn:
        db_conn.execute("DELETE FROM chunks WHERE chunk_id = 'rule:1-1-1'")

    cleanup = embed_chunks(db_conn, embedder_v1, batch_size=5)
    assert cleanup.orphans_removed == 1
    assert cleanup.embedded_chunks == 0
    assert _count_vec_rows(db_conn) == first.total_chunks - 1

    # Switch model & dimension -> chunks_vec rebuilt and all remaining chunks re-embedded.
    embedder_v2 = HashEmbedder(dimension=32, model_name="test/model-v2")
    rebuilt = embed_chunks(db_conn, embedder_v2, batch_size=5)
    assert rebuilt.rebuilt_vec_table is True
    assert rebuilt.embedded_chunks == first.total_chunks - 1
    assert rebuilt.dimension == 32
    assert get_meta(db_conn, "embedding_model") == "test/model-v2"
    assert get_meta(db_conn, "embedding_dim") == "32"
    assert _count_vec_rows(db_conn) == first.total_chunks - 1


def test_cli_embed_command_offline(
    tmp_path: Path,
    sample_rules_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "cli_embed.db"
    fake_embedder = HashEmbedder(dimension=64, model_name="test/cli-embedder")

    monkeypatch.setattr("optcg_knowledge.cli._create_embedder", lambda _settings: fake_embedder)

    assert cli_main(["--db", str(db_path), "rules", str(sample_rules_pdf)]) == 0
    capsys.readouterr()

    assert cli_main(["--db", str(db_path), "embed", "--batch-size", "4", "--device", "cpu"]) == 0
    first_out = capsys.readouterr().out
    assert "Embedded" in first_out
    assert "model=test/cli-embedder" in first_out

    assert cli_main(["--db", str(db_path), "embed"]) == 0
    second_out = capsys.readouterr().out
    assert "Embedded 0/" in second_out

    with connect(db_path) as conn:
        assert _count_vec_rows(conn) > 0


def test_sentence_transformer_embedder_friendly_error_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"sentence_transformers", "torch"}:
            raise ImportError(f"Simulated missing module: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    with pytest.raises(RuntimeError, match="uv sync --extra embeddings"):
        embedder.embed(["test chunk"])


def test_sentence_transformer_embedder_prefers_local_cache_and_redirects_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys
    import types

    calls: list[dict[str, Any]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            calls.append({"model_name": model_name, **kwargs})
            sys.stdout.write("noisy hf stdout warning\n")

        def get_embedding_dimension(self) -> int:
            return 1024

        def encode(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
            sys.stdout.write("noisy encode stdout\n")
            return [[0.5] * 1024 for _ in texts]

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float16="fp16",
    )
    fake_st = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    vectors = embedder.embed(["[Blocker]"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    assert calls[0]["local_files_only"] is True
    assert calls[0]["device"] == "cpu"

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "noisy hf stdout warning" in captured.err
    assert "noisy encode stdout" in captured.err
