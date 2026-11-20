"""Hybrid search combining OPTCG exact matches, FTS5 BM25, sqlite-vec KNN, and RRF."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from optcg_knowledge.db.schema import get_meta
from optcg_knowledge.ingest.embeddings import Embedder, serialize_vector
from optcg_knowledge.search.query_parser import (
    DetectedTokens,
    build_fts_queries,
    detect_query_tokens,
)

logger = logging.getLogger(__name__)

VALID_DOC_TYPES = frozenset({"rule", "ruling", "card"})

RRF_K = 60
EXACT_DOC_ID_BOOST = 0.12
EXACT_RULE_PREFIX_BOOST = 0.05
EXACT_LINKED_RULING_BOOST = 0.045
EXACT_KEYWORD_BOOST = 0.035
EXACT_KEYWORD_MAX_BOOST = 0.07
AUTHORITY_STEP_BOOST = 0.0005


@dataclass(slots=True)
class _Candidate:
    rowid: int
    chunk_id: str
    doc_type: str
    doc_id: str
    text: str
    authority: int
    metadata: dict[str, Any]
    source: str
    superseded_by: str | None
    ruling_date: str | None
    ruling_rules_version: str | None
    ruling_reviewed: bool | None
    ruling_url: str | None
    bm25_rank: int | None = None
    vec_rank: int | None = None
    exact_boost: float = 0.0
    signals: list[str] = field(default_factory=list)

    def add_signal(self, signal: str) -> None:
        if signal not in self.signals:
            self.signals.append(signal)


def _validate_doc_types(doc_types: Sequence[str] | None) -> tuple[str, ...] | None:
    if doc_types is None:
        return None
    normalized: list[str] = []
    for dt in doc_types:
        cleaned = dt.strip().lower()
        if cleaned not in VALID_DOC_TYPES:
            valid = ", ".join(sorted(VALID_DOC_TYPES))
            raise ValueError(f"Unsupported doc_type {dt!r}; expected one of: {valid}")
        if cleaned not in normalized:
            normalized.append(cleaned)
    return tuple(normalized) if normalized else None


def _validate_min_authority(min_authority: int | None) -> int | None:
    if min_authority is None:
        return None
    if not isinstance(min_authority, int) or min_authority not in (1, 2, 3, 4):
        raise ValueError("min_authority must be an integer between 1 (highest) and 4 (lowest)")
    return min_authority


def _build_filter_sql(
    *,
    doc_types: tuple[str, ...] | None,
    min_authority: int | None,
    include_superseded: bool,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if doc_types is not None:
        placeholders = ", ".join("?" for _ in doc_types)
        clauses.append(f"c.doc_type IN ({placeholders})")
        params.extend(doc_types)

    if min_authority is not None:
        # Authority 1 is highest (Comprehensive Rules) and 4 is lowest (community).
        clauses.append("c.authority <= ?")
        params.append(min_authority)

    if not include_superseded:
        clauses.append("(c.doc_type != 'ruling' OR r.superseded_by IS NULL)")

    sql = " AND ".join(clauses) if clauses else "1 = 1"
    return (sql, params)


def _row_to_candidate(row: sqlite3.Row) -> _Candidate:
    doc_type = str(row["doc_type"])
    raw_meta = row["metadata"]
    try:
        metadata = json.loads(raw_meta) if raw_meta else {}
    except json.JSONDecodeError:
        metadata = {}

    if doc_type == "ruling":
        source = str(row["ruling_source"] or metadata.get("source") or "ruling")
        superseded_by = row["superseded_by"]
        ruling_date = row["ruling_date"]
        ruling_rules_version = row["ruling_rules_version"]
        raw_reviewed = row["ruling_reviewed"]
        ruling_reviewed = bool(raw_reviewed) if raw_reviewed is not None else None
        ruling_url = row["ruling_url"]
        if ruling_date is not None:
            metadata["date"] = ruling_date
        if ruling_rules_version is not None:
            metadata["rules_version"] = ruling_rules_version
        if ruling_reviewed is not None:
            metadata["reviewed"] = ruling_reviewed
        if ruling_url is not None:
            metadata["url"] = ruling_url
    elif doc_type == "rule":
        source = "rules"
        superseded_by = None
        ruling_date = None
        ruling_rules_version = None
        ruling_reviewed = None
        ruling_url = None
    else:
        source = "card"
        superseded_by = None
        ruling_date = None
        ruling_rules_version = None
        ruling_reviewed = None
        ruling_url = None

    return _Candidate(
        rowid=int(row["rowid"]),
        chunk_id=str(row["chunk_id"]),
        doc_type=doc_type,
        doc_id=str(row["doc_id"]),
        text=str(row["text"]),
        authority=int(row["authority"]),
        metadata=metadata,
        source=source,
        superseded_by=str(superseded_by) if superseded_by is not None else None,
        ruling_date=str(ruling_date) if ruling_date is not None else None,
        ruling_rules_version=(
            str(ruling_rules_version) if ruling_rules_version is not None else None
        ),
        ruling_reviewed=ruling_reviewed,
        ruling_url=str(ruling_url) if ruling_url is not None else None,
    )


_BASE_SELECT = """
    SELECT
        c.rowid AS rowid,
        c.chunk_id,
        c.doc_type,
        c.doc_id,
        c.text,
        c.authority,
        c.metadata,
        r.source AS ruling_source,
        r.date AS ruling_date,
        r.rules_version AS ruling_rules_version,
        r.url AS ruling_url,
        r.superseded_by AS superseded_by,
        r.reviewed AS ruling_reviewed
    FROM chunks c
    LEFT JOIN rulings r ON c.doc_type = 'ruling' AND r.ruling_id = c.doc_id
"""


def _register_candidate(
    candidates: dict[int, _Candidate],
    row: sqlite3.Row,
) -> _Candidate:
    rowid = int(row["rowid"])
    existing = candidates.get(rowid)
    if existing is not None:
        return existing
    candidate = _row_to_candidate(row)
    candidates[rowid] = candidate
    return candidate


def _collect_exact_matches(
    conn: sqlite3.Connection,
    candidates: dict[int, _Candidate],
    tokens: DetectedTokens,
    *,
    filter_sql: str,
    filter_params: list[Any],
    keyword_limit: int,
) -> None:
    if not tokens.has_any:
        return

    if tokens.card_ids:
        placeholders = ", ".join("?" for _ in tokens.card_ids)
        rows = conn.execute(
            f"""
            {_BASE_SELECT}
            WHERE c.doc_type = 'card'
              AND UPPER(c.doc_id) IN ({placeholders})
              AND ({filter_sql})
            """,
            [*tokens.card_ids, *filter_params],
        ).fetchall()
        for row in rows:
            cand = _register_candidate(candidates, row)
            cand.exact_boost = max(cand.exact_boost, EXACT_DOC_ID_BOOST)
            cand.add_signal("exact_card_id")

        ruling_rows = conn.execute(
            f"""
            {_BASE_SELECT}
            WHERE c.doc_type = 'ruling'
              AND c.doc_id IN (
                  SELECT ruling_id FROM ruling_cards WHERE UPPER(card_id) IN ({placeholders})
              )
              AND ({filter_sql})
            """,
            [*tokens.card_ids, *filter_params],
        ).fetchall()
        for row in ruling_rows:
            cand = _register_candidate(candidates, row)
            cand.exact_boost = max(cand.exact_boost, EXACT_LINKED_RULING_BOOST)
            cand.add_signal("exact_card_ruling")

    if tokens.rule_ids:
        placeholders = ", ".join("?" for _ in tokens.rule_ids)
        exact_rows = conn.execute(
            f"""
            {_BASE_SELECT}
            WHERE c.doc_type = 'rule'
              AND c.doc_id IN ({placeholders})
              AND ({filter_sql})
            """,
            [*tokens.rule_ids, *filter_params],
        ).fetchall()
        for row in exact_rows:
            cand = _register_candidate(candidates, row)
            cand.exact_boost = max(cand.exact_boost, EXACT_DOC_ID_BOOST)
            cand.add_signal("exact_rule_id")

        prefix_clauses = " OR ".join("c.doc_id LIKE ?" for _ in tokens.rule_ids)
        prefix_params = [f"{rid}-%" for rid in tokens.rule_ids]
        prefix_rows = conn.execute(
            f"""
            {_BASE_SELECT}
            WHERE c.doc_type = 'rule'
              AND ({prefix_clauses})
              AND ({filter_sql})
            ORDER BY c.doc_id
            LIMIT ?
            """,
            [*prefix_params, *filter_params, keyword_limit],
        ).fetchall()
        for row in prefix_rows:
            cand = _register_candidate(candidates, row)
            cand.exact_boost = max(cand.exact_boost, EXACT_RULE_PREFIX_BOOST)
            cand.add_signal("exact_rule_prefix")

        ruling_rows = conn.execute(
            f"""
            {_BASE_SELECT}
            WHERE c.doc_type = 'ruling'
              AND c.doc_id IN (
                  SELECT ruling_id FROM ruling_rules WHERE rule_id IN ({placeholders})
              )
              AND ({filter_sql})
            """,
            [*tokens.rule_ids, *filter_params],
        ).fetchall()
        for row in ruling_rows:
            cand = _register_candidate(candidates, row)
            cand.exact_boost = max(cand.exact_boost, EXACT_LINKED_RULING_BOOST)
            cand.add_signal("exact_rule_ruling")

    if tokens.keywords:
        keyword_hits: dict[int, int] = {}
        for keyword in tokens.keywords:
            rows = conn.execute(
                f"""
                {_BASE_SELECT}
                WHERE INSTR(LOWER(c.text), LOWER(?)) > 0
                  AND ({filter_sql})
                ORDER BY c.authority ASC, c.rowid ASC
                LIMIT ?
                """,
                [keyword, *filter_params, keyword_limit],
            ).fetchall()
            for row in rows:
                cand = _register_candidate(candidates, row)
                keyword_hits[cand.rowid] = keyword_hits.get(cand.rowid, 0) + 1
                cand.add_signal("exact_keyword")

        for rowid, hit_count in keyword_hits.items():
            cand = candidates[rowid]
            bonus = min(EXACT_KEYWORD_MAX_BOOST, hit_count * EXACT_KEYWORD_BOOST)
            cand.exact_boost += bonus


def _collect_bm25_matches(
    conn: sqlite3.Connection,
    candidates: dict[int, _Candidate],
    query: str,
    *,
    filter_sql: str,
    filter_params: list[Any],
    limit: int,
) -> None:
    strict_fts, relaxed_fts = build_fts_queries(query)
    if strict_fts is None:
        return

    seen_rowids: set[int] = set()
    ordered_rows: list[sqlite3.Row] = []

    def _run_fts(match_expr: str, fetch_limit: int) -> list[sqlite3.Row]:
        try:
            return conn.execute(
                f"""
                {_BASE_SELECT}
                JOIN chunks_fts ON chunks_fts.rowid = c.rowid
                WHERE chunks_fts MATCH ?
                  AND ({filter_sql})
                ORDER BY bm25(chunks_fts) ASC, c.authority ASC, c.rowid ASC
                LIMIT ?
                """,
                [match_expr, *filter_params, fetch_limit],
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 query failed cleanly for %r: %s", match_expr, exc)
            return []

    for row in _run_fts(strict_fts, limit):
        rowid = int(row["rowid"])
        if rowid not in seen_rowids:
            seen_rowids.add(rowid)
            ordered_rows.append(row)

    if len(ordered_rows) < limit and relaxed_fts is not None:
        for row in _run_fts(relaxed_fts, limit):
            rowid = int(row["rowid"])
            if rowid not in seen_rowids:
                seen_rowids.add(rowid)
                ordered_rows.append(row)
                if len(ordered_rows) >= limit:
                    break

    for rank, row in enumerate(ordered_rows, start=1):
        cand = _register_candidate(candidates, row)
        cand.bm25_rank = rank
        cand.add_signal("bm25")


def _can_run_vector_search(conn: sqlite3.Connection, embedder: Embedder | None) -> int:
    """Return count of indexed vectors when vector search is usable, else 0."""
    if embedder is None:
        return 0

    table_row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'"
    ).fetchone()
    if table_row is None:
        return 0

    recorded_dim = get_meta(conn, "embedding_dim")
    if recorded_dim is not None and int(recorded_dim) != embedder.dimension:
        logger.warning(
            "Skipping vector search: DB embedding_dim=%s != embedder.dimension=%d",
            recorded_dim,
            embedder.dimension,
        )
        return 0

    recorded_model = get_meta(conn, "embedding_model")
    if recorded_model is not None and recorded_model != embedder.model_name:
        logger.warning(
            "Skipping vector search: DB embedding_model=%s != embedder.model_name=%s",
            recorded_model,
            embedder.model_name,
        )
        return 0

    count_row = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()
    return int(count_row[0]) if count_row is not None else 0


def _collect_vector_matches(
    conn: sqlite3.Connection,
    candidates: dict[int, _Candidate],
    query: str,
    embedder: Embedder,
    *,
    total_vectors: int,
    filter_sql: str,
    filter_params: list[Any],
    limit: int,
) -> bool:
    if not query.strip() or total_vectors <= 0:
        return False

    query_vector = embedder.embed_query(query)
    if len(query_vector) != embedder.dimension:
        raise ValueError(
            f"Query vector dimension {len(query_vector)} != embedder.dimension {embedder.dimension}"
        )

    vec_k = min(total_vectors, max(limit * 2, 64))
    vec_rows = conn.execute(
        """
        SELECT rowid, distance
        FROM chunks_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance ASC
        """,
        (serialize_vector(query_vector), vec_k),
    ).fetchall()
    if not vec_rows:
        return True

    ordered_rowids = [int(r["rowid"]) for r in vec_rows]
    placeholders = ", ".join("?" for _ in ordered_rowids)
    chunk_rows = conn.execute(
        f"""
        {_BASE_SELECT}
        WHERE c.rowid IN ({placeholders})
          AND ({filter_sql})
        """,
        [*ordered_rowids, *filter_params],
    ).fetchall()

    rows_by_id = {int(row["rowid"]): row for row in chunk_rows}
    rank = 0
    for rowid in ordered_rowids:
        row = rows_by_id.get(rowid)
        if row is None:
            continue
        rank += 1
        cand = _register_candidate(candidates, row)
        cand.vec_rank = rank
        cand.add_signal("vector")
        if rank >= limit:
            break

    return True


def _compute_score(candidate: _Candidate) -> float:
    score = 0.0
    if candidate.bm25_rank is not None:
        score += 1.0 / (RRF_K + candidate.bm25_rank)
    if candidate.vec_rank is not None:
        score += 1.0 / (RRF_K + candidate.vec_rank)
    score += candidate.exact_boost
    # Gentle authority tie-breaker: 1=rules > 2=official_faq/errata > 3=judge > 4=community
    score += (5 - candidate.authority) * AUTHORITY_STEP_BOOST
    return score


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    embedder: Embedder | None = None,
    k: int = 8,
    doc_types: Sequence[str] | None = None,
    min_authority: int | None = None,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Run hybrid OPTCG retrieval (exact tokens + FTS5 BM25 + sqlite-vec + RRF).

    Args:
        conn: Open SQLite connection with schema initialized.
        query: Free-text or structured OPTCG query.
        embedder: Optional local embedder for vector retrieval (falls back to lexical if absent).
        k: Maximum number of results to return (default 8).
        doc_types: Optional subset of ``("rule", "ruling", "card")``.
        min_authority: Minimum authority strength (1=rules only, 2=rules+official_faq/errata,
            3=+judge, 4=all). Filters ``authority <= min_authority``.
        include_superseded: Whether to include rulings with ``superseded_by IS NOT NULL``.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    validated_doc_types = _validate_doc_types(doc_types)
    validated_min_auth = _validate_min_authority(min_authority)
    current_rules_version = get_meta(conn, "rules_version")
    tokens = detect_query_tokens(query)

    if not query or not query.strip():
        return {
            "query": query,
            "search_mode": "lexical_only",
            "rules_version": current_rules_version,
            "detected_tokens": tokens.to_dict(),
            "total": 0,
            "results": [],
        }

    filter_sql, filter_params = _build_filter_sql(
        doc_types=validated_doc_types,
        min_authority=validated_min_auth,
        include_superseded=include_superseded,
    )
    candidate_limit = max(k * 5, 40)
    candidates: dict[int, _Candidate] = {}

    _collect_exact_matches(
        conn,
        candidates,
        tokens,
        filter_sql=filter_sql,
        filter_params=filter_params,
        keyword_limit=candidate_limit,
    )
    _collect_bm25_matches(
        conn,
        candidates,
        query,
        filter_sql=filter_sql,
        filter_params=filter_params,
        limit=candidate_limit,
    )

    total_vectors = _can_run_vector_search(conn, embedder)
    used_vector = False
    if embedder is not None and total_vectors > 0:
        used_vector = _collect_vector_matches(
            conn,
            candidates,
            query,
            embedder,
            total_vectors=total_vectors,
            filter_sql=filter_sql,
            filter_params=filter_params,
            limit=candidate_limit,
        )

    # Also boost any candidate found by BM25/vector that contains queried bracketed keywords
    # even if it fell outside the keyword-only LIMIT cap.
    if tokens.keywords:
        lowered_keywords = [kw.lower() for kw in tokens.keywords]
        for cand in candidates.values():
            if "exact_keyword" in cand.signals:
                continue
            lowered_text = cand.text.lower()
            hits = sum(1 for kw in lowered_keywords if kw in lowered_text)
            if hits > 0:
                cand.exact_boost += min(EXACT_KEYWORD_MAX_BOOST, hits * EXACT_KEYWORD_BOOST)
                cand.add_signal("exact_keyword")

    scored = [(_compute_score(cand), cand) for cand in candidates.values()]
    scored.sort(key=lambda item: (-item[0], item[1].authority, item[1].chunk_id))
    top_scored = scored[:k]

    results: list[dict[str, Any]] = []
    for score, cand in top_scored:
        potentially_outdated = bool(
            cand.doc_type == "ruling"
            and cand.ruling_rules_version
            and current_rules_version
            and cand.ruling_rules_version != current_rules_version
        )
        results.append(
            {
                "chunk_id": cand.chunk_id,
                "doc_type": cand.doc_type,
                "doc_id": cand.doc_id,
                "authority": cand.authority,
                "source": cand.source,
                "score": round(score, 6),
                "match_signals": list(cand.signals),
                "text": cand.text,
                "metadata": cand.metadata,
                "superseded_by": cand.superseded_by,
                "potentially_outdated": potentially_outdated,
            }
        )

    return {
        "query": query,
        "search_mode": "hybrid" if used_vector else "lexical_only",
        "rules_version": current_rules_version,
        "detected_tokens": tokens.to_dict(),
        "total": len(results),
        "results": results,
    }
