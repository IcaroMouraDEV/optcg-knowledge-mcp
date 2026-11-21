"""CLI entrypoint for `optcg-ingest`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from optcg_knowledge.config import Settings, get_settings
from optcg_knowledge.curation.rulings import (
    RulingsValidationError,
    export_rulings,
    import_rulings,
    validate_rulings,
)
from optcg_knowledge.db import init_schema, migrate_schema, open_db
from optcg_knowledge.ingest.banlist_parser import ingest_banlist, parse_banlist_path
from optcg_knowledge.ingest.cards_importer import ingest_cards
from optcg_knowledge.ingest.embed_pipeline import embed_chunks
from optcg_knowledge.ingest.embeddings import Embedder, SentenceTransformerEmbedder
from optcg_knowledge.ingest.faq_parser import ingest_faqs, parse_faq_path
from optcg_knowledge.ingest.official_downloader import (
    DEFAULT_OFFICIAL_BASE_URL,
    DEFAULT_OFFICIAL_BASELINE_PATH,
    check_official_freshness,
    fetch_official_docs,
)
from optcg_knowledge.ingest.rules_parser import ingest_rules, parse_rules_pdf
from optcg_knowledge.ingest.sources import CardSource, LocalCardSource, PunkRecordsSource
from optcg_knowledge.ingest.trm_parser import ingest_trm, parse_trm_pdf
from optcg_knowledge.search.queries import get_status
from optcg_knowledge.sync import run_sync


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _create_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        use_fp16=settings.embedding_fp16,
        batch_size=settings.embedding_batch_size,
        dimension=settings.embedding_dim,
    )


def _format_status_report(db_path: Path, payload: dict[str, object]) -> str:
    counts = payload["counts"]
    assert isinstance(counts, dict)
    restrictions = counts["card_restrictions"]
    assert isinstance(restrictions, dict)
    rulings = counts["rulings"]
    assert isinstance(rulings, dict)
    by_source = rulings["by_source"]
    assert isinstance(by_source, dict)
    by_authority = rulings["by_authority"]
    assert isinstance(by_authority, dict)
    judge_by_tier = rulings["judge_by_tier"]
    assert isinstance(judge_by_tier, dict)
    chunks = counts["chunks"]
    assert isinstance(chunks, dict)
    by_doc_type = chunks["by_doc_type"]
    assert isinstance(by_doc_type, dict)
    embeddings = payload["embeddings"]
    assert isinstance(embeddings, dict)
    timestamps = payload["timestamps"]
    assert isinstance(timestamps, dict)

    upcoming_eff = payload.get("banlist_upcoming_effective_date")
    upcoming_suffix = f", upcoming={upcoming_eff}" if upcoming_eff else ""
    banlist_label = (
        f"loaded (effective={payload['banlist_effective_date'] or 'unknown'}"
        f"{upcoming_suffix}, source={payload['banlist_source'] or 'unknown'})"
        if payload["banlist_loaded"]
        else "not loaded"
    )
    upcoming_restrictions = restrictions.get("upcoming")
    upcoming_counts_suffix = ""
    if isinstance(upcoming_restrictions, dict) and upcoming_restrictions.get("total", 0):
        upcoming_counts_suffix = (
            f", upcoming={upcoming_restrictions['total']} "
            f"[banned={upcoming_restrictions['banned']}, "
            f"restricted={upcoming_restrictions['restricted']}, "
            f"pair_banned={upcoming_restrictions['pair_banned']}]"
        )
    lines = [
        f"OPTCG Knowledge Base Status ({db_path})",
        "Versions & Config:",
        f"  schema_version:    {payload['schema_version'] or 'unknown'}",
        (
            f"  rules_version:     {payload['rules_version'] or 'none'} "
            f"(effective={payload['rules_effective_date'] or 'unknown'})"
        ),
        (
            f"  trm_version:       {payload['trm_version'] or 'none'} "
            f"(effective={payload['trm_effective_date'] or 'unknown'})"
        ),
        f"  banlist:           {banlist_label}",
        f"  cards_source:      {payload['cards_source'] or 'none'}",
        (
            f"  embedding_model:   {payload['embedding_model'] or 'none'} "
            f"(dim={payload['embedding_dim'] if payload['embedding_dim'] is not None else 'none'})"
        ),
        "Counts:",
        f"  rules:             {counts['rules']}",
        f"  trm_sections:      {counts['trm_sections']}",
        f"  cards:             {counts['cards']}",
        (
            f"  card_restrictions: {restrictions['total']} "
            f"(banned={restrictions['banned']}, restricted={restrictions['restricted']}, "
            f"pair_banned={restrictions['pair_banned']}{upcoming_counts_suffix})"
        ),
        (
            f"  rulings:           {rulings['total']} "
            f"(active={rulings['active']}, superseded={rulings['superseded']}, "
            f"unreviewed={rulings['unreviewed']})"
        ),
        (
            f"    by_source:       official_faq={by_source['official_faq']}, "
            f"errata={by_source['errata']}, judge={by_source['judge']}, "
            f"community={by_source['community']}, rules={by_source['rules']}"
        ),
        (
            f"    by_authority:    auth1={by_authority['1']}, auth2={by_authority['2']}, "
            f"auth3={by_authority['3']}, auth4={by_authority['4']}"
        ),
        (
            f"    judge_by_tier:   tier_0={judge_by_tier['tier_0']}, "
            f"tier_2={judge_by_tier['tier_2']}, tier_1={judge_by_tier['tier_1']}, "
            f"unspecified={judge_by_tier['unspecified']}"
        ),
        (
            f"  chunks:            {chunks['total']} "
            f"(rule={by_doc_type['rule']}, tournament_rule={by_doc_type['tournament_rule']}, "
            f"card={by_doc_type['card']}, ruling={by_doc_type['ruling']})"
        ),
        "Embeddings:",
        (
            f"  coverage:          {embeddings['embedded_chunks']}/{embeddings['total_chunks']} "
            f"({embeddings['coverage_pct']}%, pending={embeddings['pending_chunks']}, "
            f"vec_rows={embeddings['vec_rows']}, ready={embeddings['ready']})"
        ),
        "Timestamps:",
        f"  last_ingested_at:      {timestamps['last_ingested_at'] or 'never'}",
        f"  rules_last_ingested:   {timestamps['rules_last_ingested_at'] or 'never'}",
        f"  trm_last_ingested:     {timestamps['trm_last_ingested_at'] or 'never'}",
        f"  cards_last_ingested:   {timestamps['cards_last_ingested_at'] or 'never'}",
        f"  banlist_last_ingested: {timestamps['banlist_last_ingested_at'] or 'never'}",
        f"  last_faq_ingested_at:  {timestamps['last_faq_ingested_at'] or 'never'}",
        f"  last_embed_at:         {timestamps['last_embed_at'] or 'never'}",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optcg-ingest",
        description="Ingest One Piece Card Game rules, cards, and embeddings into SQLite.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite database file (defaults to OPTCG_DB_PATH or data/optcg.db).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Configured embedding model recorded in meta (defaults to OPTCG_EMBEDDING_MODEL).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging on stderr.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "init-db",
        help="Initialize or verify the SQLite schema (FTS5 + sqlite-vec).",
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help=(
            "One-command end-to-end sync: fetch-official -> rules -> trm -> cards -> "
            "banlist -> faq -> import-rulings -> embed."
        ),
    )
    sync_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw data directory (defaults to OPTCG_RAW_DATA_DIR or data/raw).",
    )
    sync_parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip downloading official Bandai documents (use existing local files in data/raw).",
    )
    sync_parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Re-download official Bandai files even if unchanged in the manifest.",
    )
    sync_parser.add_argument(
        "--base-url",
        default=DEFAULT_OFFICIAL_BASE_URL,
        help=f"Official Bandai site base URL (default: {DEFAULT_OFFICIAL_BASE_URL}).",
    )
    sync_parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Skip Comprehensive Rules ingestion.",
    )
    sync_parser.add_argument(
        "--skip-trm",
        action="store_true",
        help="Skip Tournament Rules Manual (TRM) ingestion.",
    )
    sync_parser.add_argument(
        "--skip-cards",
        action="store_true",
        help="Skip canonical card ingestion.",
    )
    sync_parser.add_argument(
        "--cards-source",
        choices=("punk-records", "local"),
        default="punk-records",
        help="Card data source adapter for sync (default: punk-records).",
    )
    sync_parser.add_argument(
        "--cards-path",
        type=Path,
        default=None,
        help="Local JSON file or directory when --cards-source local is used.",
    )
    sync_parser.add_argument(
        "--cards-base-url",
        default=None,
        help="Override base URL when --cards-source punk-records is used.",
    )
    sync_parser.add_argument(
        "--packs",
        default=None,
        help="Comma-separated pack IDs for --cards-source punk-records (e.g., OP01,OP05).",
    )
    sync_parser.add_argument(
        "--skip-banlist",
        action="store_true",
        help="Skip official banlist / card restrictions ingestion.",
    )
    sync_parser.add_argument(
        "--include-upcoming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include scheduled future-effective announcements in the separate upcoming "
            "banlist scope (default: enabled; use --no-include-upcoming to skip)."
        ),
    )
    sync_parser.add_argument(
        "--skip-faq",
        action="store_true",
        help="Skip official FAQ / errata ingestion.",
    )
    sync_parser.add_argument(
        "--skip-rulings",
        action="store_true",
        help="Skip importing curated rulings JSON from data/rulings.",
    )
    sync_parser.add_argument(
        "--rulings-dir",
        type=Path,
        default=None,
        help="Curated rulings JSON file or directory (defaults to <db_dir>/rulings).",
    )
    sync_parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip local vector embedding generation (FTS5 lexical search still works).",
    )
    sync_parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Re-embed all chunks even if their content_hash is already indexed.",
    )
    sync_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Chunks per embedding batch (default: OPTCG_EMBEDDING_BATCH_SIZE or 8).",
    )
    sync_parser.add_argument(
        "--device",
        default=None,
        help="Device to run local embeddings on: auto, cuda, cpu (default: auto).",
    )
    sync_parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 weights on CUDA to fit 4GB VRAM GPUs (default: enabled).",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Show database health, versions, row counts, embedding coverage, and timestamps.",
    )
    status_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print machine-readable JSON matching the MCP get_status() contract.",
    )

    fetch_parser = subparsers.add_parser(
        "fetch-official",
        help=(
            "Download official Bandai Comprehensive Rules PDF, Tournament Rules Manual PDF, "
            "FAQ PDFs, errata HTML, and banlist/restriction HTML."
        ),
    )
    fetch_parser.add_argument(
        "--only",
        choices=("all", "rules", "trm", "faq", "errata", "banlist"),
        default="all",
        help="Download only a subset of official documents (default: all).",
    )
    fetch_parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Target raw data directory (defaults to OPTCG_RAW_DATA_DIR or data/raw).",
    )
    fetch_parser.add_argument(
        "--base-url",
        default=DEFAULT_OFFICIAL_BASE_URL,
        help=f"Official site base URL (default: {DEFAULT_OFFICIAL_BASE_URL}).",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already match the saved revision manifest.",
    )

    check_parser = subparsers.add_parser(
        "check-official",
        help=(
            "Inspect live Bandai official document revisions and SHA-256 semantic fingerprints "
            "without downloading PDFs or images, comparing against a tracked baseline JSON."
        ),
    )
    check_parser.add_argument(
        "--only",
        choices=("all", "rules", "trm", "faq", "errata", "banlist"),
        default="all",
        help="Inspect only a subset of official documents (default: all).",
    )
    check_parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_OFFICIAL_BASELINE_PATH,
        help=(
            "Path to tracked official revisions baseline JSON "
            f"(default: {DEFAULT_OFFICIAL_BASELINE_PATH})."
        ),
    )
    check_parser.add_argument(
        "--base-url",
        default=DEFAULT_OFFICIAL_BASE_URL,
        help=f"Official site base URL (default: {DEFAULT_OFFICIAL_BASE_URL}).",
    )
    check_parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write or update the baseline JSON file with the detected live revisions.",
    )
    check_parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Print machine-readable JSON freshness report to stdout.",
    )
    check_parser.add_argument(
        "--markdown",
        dest="as_markdown",
        action="store_true",
        help="Print GitHub-flavored Markdown freshness report to stdout.",
    )
    check_parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional file path to write the JSON freshness report.",
    )
    check_parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional file path to write the Markdown freshness report.",
    )
    check_parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Return exit code 1 if any document was added, updated, or removed.",
    )

    rules_parser = subparsers.add_parser(
        "rules",
        help="Parse and ingest an official Comprehensive Rules PDF.",
    )
    rules_parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the Comprehensive Rules PDF (e.g., data/raw/rules/rule_comprehensive.pdf).",
    )
    rules_parser.add_argument(
        "--version",
        dest="rules_version",
        default=None,
        help="Override detected Comprehensive Rules version (e.g., 1.24).",
    )
    rules_parser.add_argument(
        "--effective-date",
        default=None,
        help="Override detected effective date (e.g., 2025-09-01).",
    )

    trm_parser = subparsers.add_parser(
        "trm",
        help="Parse and ingest an official Bandai Tournament Rules Manual (TRM) PDF.",
    )
    trm_parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to the Tournament Rules Manual PDF "
            "(defaults to data/raw/rules/tournament_rules_manual.pdf)."
        ),
    )
    trm_parser.add_argument(
        "--version",
        dest="trm_version",
        default=None,
        help="Override detected Tournament Rules Manual version (optional).",
    )
    trm_parser.add_argument(
        "--effective-date",
        default=None,
        help="Override detected Last Updated / effective date (e.g., 2025-10-17).",
    )

    cards_parser = subparsers.add_parser(
        "cards",
        help="Import cards from local JSON files or punk-records.",
    )
    cards_parser.add_argument(
        "--source",
        choices=("local", "punk-records"),
        default="local",
        help="Card data source adapter (default: local).",
    )
    cards_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Local JSON file or directory for --source local (default: data/raw/cards).",
    )
    cards_parser.add_argument(
        "--base-url",
        default=None,
        help="Override base URL for --source punk-records.",
    )
    cards_parser.add_argument(
        "--packs",
        default=None,
        help="Comma-separated pack IDs to fetch for --source punk-records (e.g., OP01,OP05).",
    )

    faq_parser = subparsers.add_parser(
        "faq",
        help="Parse and ingest official Bandai FAQ/errata PDFs or text files (authority = 2).",
    )
    faq_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="FAQ/errata PDF, text file, or directory (defaults to data/raw/faq).",
    )
    faq_parser.add_argument(
        "--source",
        choices=("auto", "official_faq", "errata"),
        default="auto",
        help="Override ruling source classification (default: auto-detect per file).",
    )
    faq_parser.add_argument(
        "--date",
        default=None,
        help="Override detected publication/revision date (e.g., 2025-09-01).",
    )
    faq_parser.add_argument(
        "--rules-version",
        dest="rules_version",
        default=None,
        help="Override detected Comprehensive Rules version (defaults to NULL if absent).",
    )
    faq_parser.add_argument(
        "--url",
        default=None,
        help="Optional official reference URL for the ingested FAQ/errata document.",
    )

    banlist_parser = subparsers.add_parser(
        "banlist",
        help="Parse and ingest official Bandai banlist HTML or manual JSON restrictions.",
    )
    banlist_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Banlist HTML/JSON file or directory (defaults to data/raw/banlist).",
    )
    banlist_parser.add_argument(
        "--effective-date",
        default=None,
        help="Override detected effective date (e.g., 2026-10-12).",
    )
    banlist_parser.add_argument(
        "--url",
        default=None,
        help="Optional official reference URL for the ingested banlist.",
    )
    banlist_parser.add_argument(
        "--include-upcoming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include scheduled future-effective announcements in the separate upcoming "
            "banlist scope (default: enabled; use --no-include-upcoming to skip)."
        ),
    )

    embed_parser = subparsers.add_parser(
        "embed",
        help="Incrementally generate local embeddings for new or modified chunks.",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Chunks per embedding batch (default: OPTCG_EMBEDDING_BATCH_SIZE or 8).",
    )
    embed_parser.add_argument(
        "--device",
        default=None,
        help="Device to run local embeddings on: auto, cuda, cpu (default: auto).",
    )
    embed_parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 weights on CUDA to fit 4GB VRAM GPUs (default: enabled).",
    )
    embed_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all chunks even if their content_hash is already indexed.",
    )

    export_parser = subparsers.add_parser(
        "export-rulings",
        help="Export vetted rulings in a stable JSON envelope for downstream consumers.",
    )
    export_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON file path (writes to stdout when omitted).",
    )
    export_parser.add_argument(
        "--card-id",
        dest="card_id",
        default=None,
        help="Filter exported rulings by canonical card_id (e.g., OP05-119).",
    )
    export_parser.add_argument(
        "--since",
        default=None,
        help="Filter exported rulings with date >= YYYY-MM-DD.",
    )
    export_parser.add_argument(
        "--min-authority",
        dest="min_authority",
        type=int,
        choices=(1, 2, 3, 4),
        default=None,
        help="Filter exported rulings with authority <= N (1=rules .. 4=community).",
    )
    export_parser.add_argument(
        "--source",
        choices=("official_faq", "errata", "judge", "community"),
        default=None,
        help="Filter exported rulings by exact source type (e.g., judge).",
    )
    export_parser.add_argument(
        "--include-superseded",
        action="store_true",
        help="Include rulings marked with superseded_by.",
    )
    export_parser.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="Include rulings that have not been reviewed yet (reviewed=False).",
    )

    import_parser = subparsers.add_parser(
        "import-rulings",
        help="Idempotently import curated rulings JSON (default: data/rulings).",
    )
    import_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Rulings JSON file or directory (defaults to data/rulings).",
    )

    validate_parser = subparsers.add_parser(
        "validate-rulings",
        help=(
            "Validate tracked rulings JSON schema, zero PII, and superseded_by "
            "integrity via an in-memory SQLite dry-run."
        ),
    )
    validate_parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Rulings JSON file or directory (defaults to data/rulings).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optcg-ingest CLI and return an exit status code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    settings = get_settings(
        db_path=args.db,
        raw_data_dir=getattr(args, "raw_dir", None),
        embedding_model=args.embedding_model,
        embedding_device=getattr(args, "device", None),
        embedding_batch_size=getattr(args, "batch_size", None),
        embedding_fp16=getattr(args, "fp16", None),
    )

    if args.command == "init-db":
        with open_db(
            settings.db_path,
            init=True,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ):
            pass
        print(
            f"Initialized database at {settings.db_path} "
            f"(model={settings.embedding_model}, dim={settings.embedding_dim})"
        )
        return 0

    if args.command == "sync":
        pack_ids = (
            [p.strip() for p in args.packs.split(",") if p.strip()]
            if args.packs
            else None
        )
        sync_summary = run_sync(
            settings,
            skip_fetch=args.skip_fetch,
            force_fetch=args.force_fetch,
            official_base_url=args.base_url,
            skip_rules=args.skip_rules,
            skip_trm=args.skip_trm,
            skip_cards=args.skip_cards,
            cards_source_mode=args.cards_source,
            cards_path=args.cards_path,
            cards_base_url=args.cards_base_url,
            cards_packs=pack_ids,
            skip_banlist=args.skip_banlist,
            include_upcoming_banlist=args.include_upcoming,
            skip_faq=args.skip_faq,
            skip_rulings=args.skip_rulings,
            rulings_path=args.rulings_dir,
            no_embed=args.no_embed,
            force_embed=args.force_embed,
            embedder_factory=_create_embedder,
        )
        print(f"Sync completed for {sync_summary.db_path}:")
        for step in sync_summary.steps:
            tag = "OK" if step.status == "completed" else "SKIP"
            print(f"  [{tag}] {step.name}: {step.detail}")
        emb = sync_summary.status_snapshot["embeddings"]
        print(
            f"Embeddings coverage: {emb['embedded_chunks']}/{emb['total_chunks']} "
            f"({emb['coverage_pct']}%, pending={emb['pending_chunks']})"
        )
        return 0

    if args.command == "status":
        is_memory = str(settings.db_path) == ":memory:"
        if not is_memory and not settings.db_path.exists():
            if args.as_json:
                print(
                    json.dumps(
                        {"exists": False, "db_path": str(settings.db_path)},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print(
                    f"Database not found at {settings.db_path}. "
                    "Run `optcg-ingest sync` to initialize and populate it."
                )
            return 0

        with open_db(settings.db_path, init=False) as conn:
            payload = get_status(conn)

        if args.as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(_format_status_report(settings.db_path, payload))
        return 0

    if args.command == "fetch-official":
        out_dir = args.out_dir or settings.raw_data_dir
        summary = fetch_official_docs(
            out_dir,
            only=args.only,
            base_url=args.base_url,
            force=args.force,
        )
        print(
            f"Fetched official Bandai docs into {summary.out_dir} "
            f"(downloaded={summary.downloaded}, skipped={summary.skipped}, "
            f"rules={summary.rules_files}, trm={summary.trm_files}, "
            f"faq={summary.faq_files}, errata={summary.errata_files}, "
            f"banlist={summary.banlist_files})"
        )
        return 0

    if args.command == "check-official":
        report = check_official_freshness(
            args.baseline,
            only=args.only,
            base_url=args.base_url,
            write_baseline=args.write_baseline,
        )
        json_text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        markdown_text = report.to_markdown()

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(f"{json_text}\n", encoding="utf-8")
        if args.markdown_out is not None:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(f"{markdown_text}\n", encoding="utf-8")

        if args.as_json:
            print(json_text)
        elif args.as_markdown:
            print(markdown_text)
        else:
            status_label = "up-to-date" if report.up_to_date else "updates-detected"
            baseline_note = (
                f", baseline_written={report.baseline_path}" if report.baseline_updated else ""
            )
            print(
                f"Official Bandai docs check: {status_label} "
                f"(checked={report.checked_count}, unchanged={report.unchanged_count}, "
                f"added={len(report.added)}, updated={len(report.updated)}, "
                f"removed={len(report.removed)}{baseline_note})"
            )
            for diff in (*report.added, *report.updated, *report.removed):
                prev_rev = diff.previous.revision if diff.previous is not None else "-"
                curr_rev = diff.current.revision if diff.current is not None else "-"
                print(f"  [{diff.change_type.upper()}] {diff.key}: {prev_rev} -> {curr_rev}")

        if args.fail_on_diff and not report.up_to_date:
            return 1
        return 0

    if args.command == "rules":
        doc = parse_rules_pdf(
            args.pdf,
            rules_version=args.rules_version,
            effective_date=args.effective_date,
        )
        with open_db(
            settings.db_path,
            init=True,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ) as conn:
            summary = ingest_rules(conn, doc)
        print(
            f"Ingested {summary.rules_upserted} rules "
            f"(removed={summary.rules_removed}, chunks={summary.chunks_upserted}, "
            f"version={summary.rules_version or 'unknown'}, "
            f"effective_date={summary.effective_date or 'unknown'})"
        )
        return 0

    if args.command == "trm":
        trm_pdf = args.pdf or (settings.raw_data_dir / "rules" / "tournament_rules_manual.pdf")
        trm_doc = parse_trm_pdf(
            trm_pdf,
            trm_version=args.trm_version,
            effective_date=args.effective_date,
        )
        with open_db(
            settings.db_path,
            init=True,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ) as conn:
            trm_summary = ingest_trm(conn, trm_doc)
        print(
            f"Ingested {trm_summary.sections_upserted} TRM sections "
            f"(removed={trm_summary.sections_removed}, chunks={trm_summary.chunks_upserted}, "
            f"version={trm_summary.trm_version or 'unknown'}, "
            f"effective_date={trm_summary.effective_date or 'unknown'})"
        )
        return 0

    if args.command == "cards":
        source: CardSource
        if args.source == "local":
            local_path = args.path or (settings.raw_data_dir / "cards")
            source = LocalCardSource(local_path)
        else:
            pack_ids = (
                [p.strip() for p in args.packs.split(",") if p.strip()]
                if args.packs
                else None
            )
            source = PunkRecordsSource(
                base_url=args.base_url or settings.punk_records_url,
                pack_ids=pack_ids,
            )

        with open_db(
            settings.db_path,
            init=True,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ) as conn:
            summary = ingest_cards(conn, source)
        print(
            f"Ingested {summary.cards_upserted} canonical cards from {summary.source_name} "
            f"(raw={summary.raw_records_seen}, variants_merged={summary.variants_merged}, "
            f"chunks={summary.chunks_upserted})"
        )
        return 0

    if args.command == "faq":
        faq_path = args.path or (settings.raw_data_dir / "faq")
        docs = parse_faq_path(
            faq_path,
            source_mode=args.source,
            date=args.date,
            rules_version=args.rules_version,
            url=args.url,
        )
        with open_db(
            settings.db_path,
            init=True,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ) as conn:
            summary = ingest_faqs(conn, docs)
        print(
            f"Ingested {summary.rulings_upserted} official rulings from "
            f"{summary.files_processed} file(s) "
            f"(official_faq={summary.official_faq_count}, errata={summary.errata_count}, "
            f"card_links={summary.card_links_upserted}, rule_links={summary.rule_links_upserted}, "
            f"chunks={summary.chunks_upserted})"
        )
        return 0

    if args.command == "banlist":
        banlist_path = args.path or (settings.raw_data_dir / "banlist")
        banlist_doc = parse_banlist_path(
            banlist_path,
            effective_date=args.effective_date,
            source_url=args.url,
            include_upcoming=args.include_upcoming,
        )
        with open_db(
            settings.db_path,
            init=True,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        ) as conn:
            banlist_summary = ingest_banlist(conn, banlist_doc)
        print(
            f"Ingested {banlist_summary.restrictions_upserted} card restriction(s) "
            f"(removed={banlist_summary.restrictions_removed}, "
            f"banned={banlist_summary.banned_count}, "
            f"restricted={banlist_summary.restricted_count}, "
            f"pair_banned={banlist_summary.pair_banned_count}, "
            f"upcoming={banlist_summary.upcoming_restrictions_upserted}, "
            f"effective_date={banlist_summary.effective_date or 'unknown'}, "
            f"upcoming_effective_date={banlist_summary.upcoming_effective_date or 'none'}, "
            f"source={banlist_summary.source_name})"
        )
        return 0

    if args.command == "embed":
        embedder = _create_embedder(settings)
        with open_db(
            settings.db_path,
            init=True,
            embedding_model=embedder.model_name,
            embedding_dim=embedder.dimension,
        ) as conn:
            summary = embed_chunks(
                conn,
                embedder,
                batch_size=settings.embedding_batch_size,
                force=args.force,
            )
        print(
            f"Embedded {summary.embedded_chunks}/{summary.total_chunks} chunks "
            f"(skipped={summary.skipped_chunks}, orphans_removed={summary.orphans_removed}, "
            f"rebuilt={summary.rebuilt_vec_table}, model={summary.model_name}, "
            f"dim={summary.dimension})"
        )
        return 0

    if args.command == "export-rulings":
        with open_db(settings.db_path, init=False) as conn:
            has_rulings = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rulings'"
            ).fetchone()
            if has_rulings is None:
                init_schema(
                    conn,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                )
            else:
                with conn:
                    migrate_schema(conn)

            payload = export_rulings(
                conn,
                card_id=args.card_id,
                since=args.since,
                min_authority=args.min_authority,
                source=args.source,
                include_superseded=args.include_superseded,
                include_unreviewed=args.include_unreviewed,
            )

        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(f"{serialized}\n", encoding="utf-8")
            print(f"Exported {payload['total']} rulings to {args.out}")
        else:
            print(serialized)
        return 0

    if args.command == "import-rulings":
        import_path = args.path or (settings.db_path.parent / "rulings")
        with open_db(settings.db_path, init=False) as conn:
            has_rulings = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rulings'"
            ).fetchone()
            if has_rulings is None:
                init_schema(
                    conn,
                    embedding_model=settings.embedding_model,
                    embedding_dim=settings.embedding_dim,
                )
            else:
                with conn:
                    migrate_schema(conn)

            summary = import_rulings(conn, import_path)
        print(
            f"Imported {summary.rulings_upserted} rulings from {summary.files_processed} file(s) "
            f"(card_links={summary.card_links_upserted}, rule_links={summary.rule_links_upserted}, "
            f"superseded={summary.superseded_links_applied}, chunks={summary.chunks_upserted})"
        )
        return 0

    if args.command == "validate-rulings":
        validate_path = args.path or (settings.db_path.parent / "rulings")
        try:
            summary = validate_rulings(validate_path)
        except (RulingsValidationError, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for warning in summary.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        print(
            f"Validated {summary.rulings_checked} ruling(s) across "
            f"{summary.json_files_checked} JSON file(s) and "
            f"{summary.thread_files_checked} thread file(s) "
            f"(superseded_links={summary.superseded_links_checked}, "
            f"warnings={len(summary.warnings)})"
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
