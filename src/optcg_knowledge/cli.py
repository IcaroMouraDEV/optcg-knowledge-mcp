"""CLI entrypoint for `optcg-ingest`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from optcg_knowledge.config import get_settings
from optcg_knowledge.db import open_db
from optcg_knowledge.ingest.cards_importer import ingest_cards
from optcg_knowledge.ingest.rules_parser import ingest_rules, parse_rules_pdf
from optcg_knowledge.ingest.sources import CardSource, LocalCardSource, PunkRecordsSource


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optcg-ingest",
        description="Ingest One Piece Card Game rules and cards into the SQLite knowledge base.",
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optcg-ingest CLI and return an exit status code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    settings = get_settings(
        db_path=args.db,
        embedding_model=args.embedding_model,
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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
