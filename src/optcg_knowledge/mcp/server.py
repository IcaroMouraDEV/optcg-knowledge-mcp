"""MCP stdio server exposing OPTCG hybrid search, structured queries, resources, and prompts."""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from optcg_knowledge.config import Settings, get_settings
from optcg_knowledge.curation.rulings import (
    add_ruling as curate_add_ruling,
)
from optcg_knowledge.curation.rulings import (
    export_rulings as curate_export_rulings,
)
from optcg_knowledge.curation.rulings import (
    list_unreviewed_rulings as curate_list_unreviewed_rulings,
)
from optcg_knowledge.curation.rulings import (
    mark_ruling as curate_mark_ruling,
)
from optcg_knowledge.db.connection import open_db
from optcg_knowledge.db.schema import get_meta, init_schema, migrate_schema
from optcg_knowledge.ingest.embeddings import Embedder, SentenceTransformerEmbedder
from optcg_knowledge.mcp.prompts import build_curar_prompt, build_juiz_prompt
from optcg_knowledge.search.hybrid import hybrid_search
from optcg_knowledge.search.queries import (
    get_banlist as query_get_banlist,
)
from optcg_knowledge.search.queries import (
    get_card as query_get_card,
)
from optcg_knowledge.search.queries import (
    get_card_rulings as query_get_card_rulings,
)
from optcg_knowledge.search.queries import (
    get_cards as query_get_cards,
)
from optcg_knowledge.search.queries import (
    get_rule as query_get_rule,
)
from optcg_knowledge.search.queries import (
    get_rule_rulings as query_get_rule_rulings,
)
from optcg_knowledge.search.queries import (
    get_status as query_get_status,
)
from optcg_knowledge.search.queries import (
    get_trm_section as query_get_trm_section,
)
from optcg_knowledge.search.queries import (
    search_cards as query_search_cards,
)

logger = logging.getLogger(__name__)


def _configure_stderr_logging(verbose: bool = False) -> None:
    """Configure root logging exclusively to stderr to preserve MCP stdio frames."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


class _LazyEmbedderProvider:
    """Resolves and caches a local embedder only when vector search is actually needed."""

    def __init__(
        self,
        settings: Settings,
        *,
        injected_embedder: Embedder | None = None,
    ) -> None:
        self._settings = settings
        self._injected_embedder = injected_embedder
        self._cached_embedder: Embedder | None = None
        self._unavailable = False

    def resolve(self, conn: sqlite3.Connection) -> Embedder | None:
        if self._settings.search_mode == "lexical":
            return None

        if self._injected_embedder is not None:
            return self._injected_embedder

        if self._cached_embedder is not None:
            return self._cached_embedder

        if self._unavailable:
            return None

        table_row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'"
        ).fetchone()
        if table_row is None:
            return None

        count_row = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()
        vec_count = int(count_row[0]) if count_row is not None else 0
        if vec_count == 0:
            return None

        recorded_model = get_meta(conn, "embedding_model") or self._settings.embedding_model
        recorded_dim_raw = get_meta(conn, "embedding_dim")
        recorded_dim = int(recorded_dim_raw) if recorded_dim_raw else self._settings.embedding_dim

        candidate = SentenceTransformerEmbedder(
            model_name=recorded_model,
            device=self._settings.embedding_device,
            use_fp16=self._settings.embedding_fp16,
            batch_size=self._settings.embedding_batch_size,
            dimension=recorded_dim,
        )
        try:
            candidate.load()
        except Exception as exc:
            if self._settings.search_mode == "hybrid":
                raise
            logger.warning(
                "Falling back to lexical_only search because local embedder could not load: %s",
                exc,
            )
            self._unavailable = True
            return None

        self._cached_embedder = candidate
        return candidate


@contextmanager
def _open_query_db(settings: Settings) -> Iterator[sqlite3.Connection]:
    """Open SQLite DB without overwriting existing vec table configuration."""
    with open_db(settings.db_path, init=False) as conn:
        has_chunks = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks'"
        ).fetchone()
        if has_chunks is None:
            init_schema(
                conn,
                embedding_dim=settings.embedding_dim,
                embedding_model=settings.embedding_model,
            )
        else:
            with conn:
                migrate_schema(conn)
        yield conn


def create_server(
    *,
    db_path: Path | str | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    search_mode: str | None = None,
) -> MCPServer:
    """Create and configure the `optcg-knowledge` MCP server with tools, resources, and prompts."""
    resolved_settings = settings or get_settings(
        db_path=db_path,
        search_mode=search_mode,
    )
    if settings is not None and (db_path is not None or search_mode is not None):
        resolved_settings = get_settings(
            db_path=db_path or settings.db_path,
            raw_data_dir=settings.raw_data_dir,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
            embedding_device=settings.embedding_device,
            embedding_batch_size=settings.embedding_batch_size,
            embedding_fp16=settings.embedding_fp16,
            search_mode=search_mode or settings.search_mode,
            punk_records_url=settings.punk_records_url,
        )

    embedder_provider = _LazyEmbedderProvider(
        resolved_settings,
        injected_embedder=embedder,
    )

    server = MCPServer(
        name="optcg-knowledge",
        instructions=(
            "One Piece Card Game (OPTCG) knowledge base exposing Comprehensive Rules, "
            "Tournament Rules Manual (TRM), canonical cards with Block Icons (single get_card, "
            "batch paginated get_cards, and filtered search_cards including block_icon/min_block), "
            "official Bandai banlist / card restrictions "
            "(via get_banlist, get_card, get_cards, and search_cards), official FAQs/errata, "
            "judge rulings (via get_card_rulings, get_rule_rulings, and hybrid search), "
            "database health/freshness diagnostics (via get_status), and read-only MCP resources "
            "(optcg://status, optcg://banlist, optcg://rules/{rule_id}, optcg://trm/{section_id}, "
            "and optcg://cards/{card_id}). Authority hierarchy: "
            "1=rules/tournament_rules > 2=official_faq/errata > 3=judge > 4=community."
        ),
    )

    @server.tool(
        name="search",
        description=(
            "Hybrid search across OPTCG Comprehensive Rules, Tournament Rules Manual (TRM), "
            "cards, FAQs, and judge rulings using exact-match OPTCG token detection (card IDs, "
            "rule numbers, TRM section IDs like 'TRM-7.4.1', [Keywords]), FTS5 BM25, and local "
            "sqlite-vec embeddings fused with RRF and authority tie-breaking. Optional doc_types "
            "filters any subset of ('rule', 'tournament_rule', 'card', 'ruling'). min_authority "
            "filters authority <= N (1=rules/TRM only, 2=+official_faq/errata, 3=+judge, 4=all). "
            "Superseded rulings are excluded unless include_superseded=True."
        ),
    )
    def search(
        query: str,
        k: int = 8,
        doc_types: list[str] | None = None,
        min_authority: int | None = None,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """Run hybrid retrieval over rules, cards, and rulings."""
        with _open_query_db(resolved_settings) as conn:
            active_embedder = embedder_provider.resolve(conn)
            return hybrid_search(
                conn,
                query,
                embedder=active_embedder,
                k=k,
                doc_types=doc_types,
                min_authority=min_authority,
                include_superseded=include_superseded,
            )

    @server.tool(
        name="get_rule",
        description=(
            "Retrieve a specific OPTCG Comprehensive Rule by canonical rule_id (e.g. '7-1-4' "
            "or '7-1-4-1'), including its ancestor chain and, when include_children=True, all "
            "descendant sub-rules ordered numerically."
        ),
    )
    def get_rule(rule_id: str, include_children: bool = True) -> dict[str, Any]:
        """Look up a Comprehensive Rule and optionally its ordered descendants."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_rule(conn, rule_id, include_children=include_children)

    @server.tool(
        name="get_trm_section",
        description=(
            "Retrieve a specific official Bandai Tournament Rules Manual (TRM) section by "
            "section_id (e.g. 'TRM-7.4.1', 'TRM-7.4.1.2', or '7.4.1'), including its ancestor "
            "chain and, when include_children=True, all descendant subsections/infractions "
            "ordered numerically."
        ),
    )
    def get_trm_section(section_id: str, include_children: bool = True) -> dict[str, Any]:
        """Look up a Tournament Rules Manual section and optionally its ordered descendants."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_trm_section(conn, section_id, include_children=include_children)

    @server.tool(
        name="get_card",
        description=(
            "Look up an OPTCG card by canonical card_id (e.g. 'OP05-119', 'ST10-001', 'P-001') "
            "or by name (exact or fuzzy, ignoring punctuation differences like 'Monkey D Luffy'). "
            "Returns all matching canonical card printings/versions along with official Block Icon "
            "(block_icon, nullable int reflecting the highest known printing block for Standard "
            "rotation checks), active official banlist legality (ban_status, effective max_copies "
            "incorporating CR 5-1-2 Leader=1 and unlimited-copy=50 rules alongside banlist caps, "
            "paired_card_ids, restriction_note, effective_date), and any scheduled "
            "upcoming_restriction."
        ),
    )
    def get_card(id_or_name: str) -> dict[str, Any]:
        """Retrieve canonical card details by exact ID or fuzzy name."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_card(conn, id_or_name)

    @server.tool(
        name="get_cards",
        description=(
            "Batch look up multiple OPTCG cards by canonical card_ids (e.g. ['OP05-119', "
            "'OP04-083', 'ST10-001']) in a single paginated call (default page=1, page_size=25, "
            "max 50 per page; returns total_pages, has_more, and next_page). Preserves request "
            "order, reports missing IDs under not_found, includes official block_icon, active "
            "banlist legality, and upcoming_restriction, and optionally embeds each card's linked "
            "rulings ordered by authority ASC, judge_tier, date DESC when include_rulings=True."
        ),
    )
    def get_cards(
        card_ids: list[str],
        include_rulings: bool = False,
        include_superseded: bool = False,
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        """Batch look up canonical cards with pagination, banlist status, and optional rulings."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_cards(
                conn,
                card_ids,
                include_rulings=include_rulings,
                include_superseded=include_superseded,
                page=page,
                page_size=page_size,
            )

    @server.tool(
        name="search_cards",
        description=(
            "Filter canonical OPTCG cards by name substring, color, exact cost, trait substring, "
            "effect/trigger keyword (e.g. '[Blocker]' or 'Rush'), set_code (e.g. 'OP05'), "
            "active official ban_status ('legal', 'banned', 'restricted', or 'pair_banned'), "
            "exact block_icon (e.g. 1, 2, 3, 4), and/or min_block (block_icon >= N for Standard "
            "format rotation checks). Defaults to returning up to 25 matches (max 100), including "
            "block_icon and upcoming_restriction metadata on each card."
        ),
    )
    def search_cards(
        name: str | None = None,
        color: str | None = None,
        cost: int | None = None,
        trait: str | None = None,
        keyword: str | None = None,
        set_code: str | None = None,
        ban_status: str | None = None,
        block_icon: int | None = None,
        min_block: int | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        """Search canonical cards using structured attribute, block icon, and banlist filters."""
        with _open_query_db(resolved_settings) as conn:
            return query_search_cards(
                conn,
                name=name,
                color=color,
                cost=cost,
                trait=trait,
                keyword=keyword,
                set_code=set_code,
                ban_status=ban_status,
                block_icon=block_icon,
                min_block=min_block,
                limit=limit,
            )

    @server.tool(
        name="get_banlist",
        description=(
            "Retrieve the official Bandai OPTCG banlist / card restrictions in a single call, "
            "separating active restrictions (counts, deduplicated banned_pairs, and restrictions) "
            "from scheduled future announcements (under the `upcoming` section with its own "
            "effective_date, counts, banned_pairs, and restrictions). Optionally filter by "
            "ban_status ('banned', 'restricted', or 'pair_banned')."
        ),
    )
    def get_banlist(ban_status: str | None = None) -> dict[str, Any]:
        """List active official banlist and card restriction entries."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_banlist(conn, ban_status=ban_status)

    @server.tool(
        name="get_card_rulings",
        description=(
            "Retrieve official FAQs, errata, and judge rulings linked to a canonical card_id "
            "(e.g. 'OP05-119'), ordered by authority ASC (1 highest .. 4 lowest), judge_tier "
            "(tier_0 > tier_2 > tier_1), then date DESC. Excludes superseded rulings by default "
            "unless include_superseded=True."
        ),
    )
    def get_card_rulings(card_id: str, include_superseded: bool = False) -> dict[str, Any]:
        """List rulings associated with a specific card_id."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_card_rulings(
                conn,
                card_id,
                include_superseded=include_superseded,
            )

    @server.tool(
        name="get_rule_rulings",
        description=(
            "Retrieve official FAQs, errata, and judge rulings citing a specific Comprehensive "
            "Rule (e.g. '7-1-4-1' or '1-3-2'), ordered by authority ASC (1 highest .. 4 lowest), "
            "judge_tier (tier_0 > tier_2 > tier_1), then date DESC. Set include_children=True to "
            "also include rulings citing descendant sub-rules (e.g. '7-1-4-%'). Excludes "
            "superseded rulings by default unless include_superseded=True."
        ),
    )
    def get_rule_rulings(
        rule_id: str,
        include_superseded: bool = False,
        include_children: bool = False,
    ) -> dict[str, Any]:
        """List rulings citing a Comprehensive Rule (and optionally descendant sub-rules)."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_rule_rulings(
                conn,
                rule_id,
                include_superseded=include_superseded,
                include_children=include_children,
            )

    @server.tool(
        name="add_ruling",
        description=(
            "Add a curated OPTCG ruling (always enters with reviewed=False) and index its chunk "
            "immediately in FTS5. Authority is derived from source ('official_faq'/'errata'=2, "
            "'judge'=3, 'community'=4). Optional judge_tier ('tier_0' | 'tier_2' | 'tier_1', "
            "precedence tier_0 > tier_2 > tier_1) is valid for source='judge'. Never pass "
            "personal names, @handles, or Discord user IDs."
        ),
    )
    def add_ruling(
        question: str,
        answer: str,
        source: str,
        date: str | None = None,
        card_ids: list[str] | None = None,
        rule_refs: list[str] | None = None,
        url: str | None = None,
        raw_ref: str | None = None,
        rules_version: str | None = None,
        judge_tier: str | None = None,
    ) -> dict[str, Any]:
        """Persist a new curated ruling with reviewed=False and canonical card/rule links."""
        with _open_query_db(resolved_settings) as conn:
            return curate_add_ruling(
                conn,
                question=question,
                answer=answer,
                source=source,
                date=date,
                card_ids=card_ids,
                rule_refs=rule_refs,
                url=url,
                raw_ref=raw_ref,
                rules_version=rules_version,
                judge_tier=judge_tier,
            )

    @server.tool(
        name="mark_ruling",
        description=(
            "Update a ruling's reviewed status, link it as superseded_by a newer ruling_id, "
            "and/or set judge_tier ('tier_0' | 'tier_2' | 'tier_1' for source='judge'). "
            "Never deletes outdated rulings; rejects self-superseding and circular chains."
        ),
    )
    def mark_ruling(
        ruling_id: str,
        reviewed: bool | None = None,
        superseded_by: str | None = None,
        judge_tier: str | None = None,
    ) -> dict[str, Any]:
        """Mark a ruling as reviewed, superseded_by another ruling, or update its judge_tier."""
        with _open_query_db(resolved_settings) as conn:
            return curate_mark_ruling(
                conn,
                ruling_id,
                reviewed=reviewed,
                superseded_by=superseded_by,
                judge_tier=judge_tier,
            )

    @server.tool(
        name="list_unreviewed_rulings",
        description=(
            "List curated rulings pending human review (reviewed=False), ordered by authority, "
            "judge_tier (tier_0 > tier_2 > tier_1), and date DESC (default limit=20, max 100)."
        ),
    )
    def list_unreviewed_rulings(limit: int = 20) -> dict[str, Any]:
        """Return rulings awaiting curation review."""
        with _open_query_db(resolved_settings) as conn:
            return curate_list_unreviewed_rulings(conn, limit=limit)

    @server.tool(
        name="get_status",
        description=(
            "Return OPTCG knowledge base health and freshness diagnostics: schema_version, "
            "rules_version, trm_version, banlist_loaded/banlist_effective_date, embedding_model "
            "and embedding_dim, row counts by table/source/authority/judge_tier (including "
            "unreviewed rulings), vector embedding coverage (embedded/total/pending chunks), "
            "and ingestion/embed timestamps."
        ),
    )
    def get_status() -> dict[str, Any]:
        """Inspect database versions, counts, embedding coverage, and ingestion timestamps."""
        with _open_query_db(resolved_settings) as conn:
            return query_get_status(conn)

    @server.tool(
        name="export_rulings",
        description=(
            "Export rulings in a stable, versioned JSON envelope (shared with `optcg-ingest "
            "export-rulings`) for downstream consumers such as optcg-engine. Defaults to reviewed, "
            "non-superseded rulings; filter optionally by card_id, since (YYYY-MM-DD), "
            "min_authority (authority <= N), and source "
            "('official_faq'|'errata'|'judge'|'community')."
        ),
    )
    def export_rulings(
        card_id: str | None = None,
        since: str | None = None,
        min_authority: int | None = None,
        source: str | None = None,
        include_superseded: bool = False,
        include_unreviewed: bool = False,
    ) -> dict[str, Any]:
        """Export vetted OPTCG rulings as stable structured JSON."""
        with _open_query_db(resolved_settings) as conn:
            return curate_export_rulings(
                conn,
                card_id=card_id,
                since=since,
                min_authority=min_authority,
                source=source,
                include_superseded=include_superseded,
                include_unreviewed=include_unreviewed,
            )

    @server.resource(
        "optcg://status",
        name="status",
        description=(
            "Read-only OPTCG knowledge base health and freshness diagnostics (schema_version, "
            "rules_version, trm_version, banlist effective dates, embedding model/coverage, "
            "table/authority/tier counts, and ingestion timestamps)."
        ),
        mime_type="application/json",
    )
    def resource_status() -> dict[str, Any]:
        """Return database health and freshness diagnostics via `optcg://status`."""
        return get_status()

    @server.resource(
        "optcg://banlist",
        name="banlist",
        description=(
            "Read-only official Bandai OPTCG banlist / card restrictions separating active "
            "restrictions (counts, banned_pairs, restrictions) from scheduled upcoming "
            "announcements."
        ),
        mime_type="application/json",
    )
    def resource_banlist() -> dict[str, Any]:
        """Return the active and upcoming official Bandai banlist via `optcg://banlist`."""
        return get_banlist()

    @server.resource(
        "optcg://rules/{rule_id}",
        name="rule",
        description=(
            "Read-only OPTCG Comprehensive Rule lookup by canonical rule_id (e.g. '7-1-4' or "
            "'7-1-4-1'), including its ancestor chain and ordered descendant sub-rules."
        ),
        mime_type="application/json",
    )
    def resource_rule(rule_id: str) -> dict[str, Any]:
        """Return a Comprehensive Rule and its ordered descendants via `optcg://rules/{rule_id}`."""
        return get_rule(rule_id, include_children=True)

    @server.resource(
        "optcg://trm/{section_id}",
        name="trm_section",
        description=(
            "Read-only official Bandai Tournament Rules Manual (TRM) section lookup by "
            "section_id (e.g. 'TRM-7.4.1', 'TRM-7.4.1.2', or '7.4.1'), including its ancestor "
            "chain and ordered descendant subsections/infractions."
        ),
        mime_type="application/json",
    )
    def resource_trm_section(section_id: str) -> dict[str, Any]:
        """Return a TRM section and its ordered descendants via `optcg://trm/{section_id}`."""
        return get_trm_section(section_id, include_children=True)

    @server.resource(
        "optcg://cards/{card_id}",
        name="card",
        description=(
            "Read-only canonical OPTCG card lookup by card_id (e.g. 'OP05-119', 'ST10-001', "
            "'P-001'), including official block_icon, active official banlist legality (with "
            "effective max_copies respecting CR 5-1-2 deck rules), and any scheduled "
            "upcoming_restriction."
        ),
        mime_type="application/json",
    )
    def resource_card(card_id: str) -> dict[str, Any]:
        """Return canonical card details and banlist legality via `optcg://cards/{card_id}`."""
        return get_card(card_id)

    @server.prompt(
        name="juiz",
        description=(
            "OPTCG rules judge workflow: consult rules/cards/rulings, follow authority hierarchy "
            "(1=rules > 2=official_faq/errata > 3=judge [tier_0 > tier_2 > tier_1] > 4=community), "
            "cite exact IDs, surface conflicts and outdated rulings."
        ),
    )
    def juiz(question: str = "") -> str:
        """Return judge instructions optionally seeded with the user's rules question."""
        return build_juiz_prompt(question)

    @server.prompt(
        name="curar",
        description=(
            "Judge chat curation workflow: read a raw export under data/raw/judge_chats/, "
            "extract Q&A pairs, respect judge tiers (tier_0 > tier_2 > tier_1), strip personal "
            "identifiers, verify card/rule links, and call add_ruling."
        ),
    )
    def curar(arquivo: str) -> str:
        """Return curation instructions targeting a file under data/raw/judge_chats/."""
        return build_curar_prompt(arquivo)

    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optcg-mcp",
        description="Run the optcg-knowledge MCP server over stdio.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite database file (defaults to OPTCG_DB_PATH or data/optcg.db).",
    )
    parser.add_argument(
        "--search-mode",
        choices=("auto", "hybrid", "lexical"),
        default=None,
        help="Search mode: auto (default), hybrid (require local embedder), or lexical.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging on stderr.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint running the optcg-knowledge MCP server over stdio."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_stderr_logging(args.verbose)

    settings = get_settings(
        db_path=args.db,
        search_mode=args.search_mode,
    )
    logger.info(
        "Starting optcg-knowledge MCP server (db=%s, search_mode=%s)",
        settings.db_path,
        settings.search_mode,
    )
    server = create_server(settings=settings)
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
