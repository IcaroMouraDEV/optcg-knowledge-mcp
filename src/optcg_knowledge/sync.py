"""One-command end-to-end sync orchestrator for `optcg-ingest sync`."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from optcg_knowledge.config import Settings
from optcg_knowledge.curation.rulings import import_rulings
from optcg_knowledge.db.connection import open_db
from optcg_knowledge.ingest.banlist_parser import ingest_banlist, parse_banlist_path
from optcg_knowledge.ingest.cards_importer import ingest_cards
from optcg_knowledge.ingest.embed_pipeline import embed_chunks
from optcg_knowledge.ingest.embeddings import Embedder, SentenceTransformerEmbedder
from optcg_knowledge.ingest.faq_parser import ingest_faqs, parse_faq_path
from optcg_knowledge.ingest.official_downloader import (
    DEFAULT_OFFICIAL_BASE_URL,
    fetch_official_docs,
)
from optcg_knowledge.ingest.rules_parser import ingest_rules, parse_rules_pdf
from optcg_knowledge.ingest.sources import CardSource, LocalCardSource, PunkRecordsSource
from optcg_knowledge.ingest.trm_parser import ingest_trm, parse_trm_pdf
from optcg_knowledge.search.queries import get_status

logger = logging.getLogger(__name__)

SyncStepStatus = Literal["completed", "skipped"]
CardsSourceMode = Literal["punk-records", "local"]

_BANLIST_SUFFIXES = (".html", ".htm", ".json")
_FAQ_SUFFIXES = (".pdf", ".html", ".htm", ".txt")
_RULINGS_SUFFIXES = (".json",)


@dataclass(frozen=True, slots=True)
class SyncStepResult:
    """Outcome of a single step within `optcg-ingest sync`."""

    name: str
    status: SyncStepStatus
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncSummary:
    """Aggregate report produced by `run_sync`."""

    db_path: Path
    raw_data_dir: Path
    steps: tuple[SyncStepResult, ...]
    status_snapshot: dict[str, Any]

    def step_by_name(self, name: str) -> SyncStepResult | None:
        """Look up a step result by canonical step name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None


def is_embeddings_extra_installed() -> bool:
    """Return True when the optional `[embeddings]` dependencies are importable."""
    return (
        importlib.util.find_spec("sentence_transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )


def _has_supported_files(path: Path, suffixes: tuple[str, ...]) -> bool:
    if path.is_file():
        return path.suffix.lower() in suffixes
    if not path.is_dir():
        return False
    return any(
        child.is_file()
        and not child.name.startswith(".")
        and child.suffix.lower() in suffixes
        for child in path.iterdir()
    )


def run_sync(
    settings: Settings,
    *,
    skip_fetch: bool = False,
    force_fetch: bool = False,
    official_base_url: str = DEFAULT_OFFICIAL_BASE_URL,
    skip_rules: bool = False,
    skip_trm: bool = False,
    skip_cards: bool = False,
    cards_source_mode: CardsSourceMode = "punk-records",
    cards_path: Path | None = None,
    cards_base_url: str | None = None,
    cards_packs: Sequence[str] | None = None,
    skip_banlist: bool = False,
    include_upcoming_banlist: bool = False,
    skip_faq: bool = False,
    skip_rulings: bool = False,
    rulings_path: Path | None = None,
    no_embed: bool = False,
    force_embed: bool = False,
    fetch_client: httpx.Client | None = None,
    card_source: CardSource | None = None,
    embedder: Embedder | None = None,
    embedder_factory: Callable[[Settings], Embedder] | None = None,
    embeddings_available: Callable[[], bool] | None = None,
) -> SyncSummary:
    """Run the idempotent end-to-end OPTCG knowledge base sync pipeline."""
    steps: list[SyncStepResult] = []
    raw_dir = settings.raw_data_dir

    # 1. fetch-official
    if skip_fetch:
        steps.append(
            SyncStepResult(
                name="fetch-official",
                status="skipped",
                detail="Skipped via --skip-fetch",
            )
        )
    else:
        fetch_summary = fetch_official_docs(
            raw_dir,
            only="all",
            base_url=official_base_url,
            force=force_fetch,
            client=fetch_client,
        )
        steps.append(
            SyncStepResult(
                name="fetch-official",
                status="completed",
                detail=(
                    f"downloaded={fetch_summary.downloaded}, skipped={fetch_summary.skipped}, "
                    f"rules={fetch_summary.rules_files}, trm={fetch_summary.trm_files}, "
                    f"faq={fetch_summary.faq_files}, errata={fetch_summary.errata_files}, "
                    f"banlist={fetch_summary.banlist_files}"
                ),
                metrics={
                    "downloaded": fetch_summary.downloaded,
                    "skipped": fetch_summary.skipped,
                    "rules_files": fetch_summary.rules_files,
                    "trm_files": fetch_summary.trm_files,
                    "faq_files": fetch_summary.faq_files,
                    "errata_files": fetch_summary.errata_files,
                    "banlist_files": fetch_summary.banlist_files,
                },
            )
        )

    init_model = embedder.model_name if embedder is not None else settings.embedding_model
    init_dim = embedder.dimension if embedder is not None else settings.embedding_dim

    with open_db(
        settings.db_path,
        init=True,
        embedding_model=init_model,
        embedding_dim=init_dim,
    ) as conn:
        # 2. rules
        rules_pdf = raw_dir / "rules" / "rule_comprehensive.pdf"
        if skip_rules:
            steps.append(
                SyncStepResult(name="rules", status="skipped", detail="Skipped via --skip-rules")
            )
        elif skip_fetch and not rules_pdf.is_file():
            steps.append(
                SyncStepResult(
                    name="rules",
                    status="skipped",
                    detail=f"Local file not found under --skip-fetch: {rules_pdf}",
                )
            )
        else:
            rules_doc = parse_rules_pdf(rules_pdf)
            rules_summary = ingest_rules(conn, rules_doc)
            steps.append(
                SyncStepResult(
                    name="rules",
                    status="completed",
                    detail=(
                        f"upserted={rules_summary.rules_upserted}, "
                        f"removed={rules_summary.rules_removed}, "
                        f"version={rules_summary.rules_version or 'unknown'}"
                    ),
                    metrics={
                        "rules_upserted": rules_summary.rules_upserted,
                        "rules_removed": rules_summary.rules_removed,
                        "chunks_upserted": rules_summary.chunks_upserted,
                        "rules_version": rules_summary.rules_version,
                    },
                )
            )

        # 3. trm
        trm_pdf = raw_dir / "rules" / "tournament_rules_manual.pdf"
        if skip_trm:
            steps.append(
                SyncStepResult(name="trm", status="skipped", detail="Skipped via --skip-trm")
            )
        elif skip_fetch and not trm_pdf.is_file():
            steps.append(
                SyncStepResult(
                    name="trm",
                    status="skipped",
                    detail=f"Local file not found under --skip-fetch: {trm_pdf}",
                )
            )
        else:
            trm_doc = parse_trm_pdf(trm_pdf)
            trm_summary = ingest_trm(conn, trm_doc)
            steps.append(
                SyncStepResult(
                    name="trm",
                    status="completed",
                    detail=(
                        f"upserted={trm_summary.sections_upserted}, "
                        f"removed={trm_summary.sections_removed}, "
                        f"effective_date={trm_summary.effective_date or 'unknown'}"
                    ),
                    metrics={
                        "sections_upserted": trm_summary.sections_upserted,
                        "sections_removed": trm_summary.sections_removed,
                        "chunks_upserted": trm_summary.chunks_upserted,
                        "trm_version": trm_summary.trm_version,
                        "effective_date": trm_summary.effective_date,
                    },
                )
            )

        # 4. cards
        if skip_cards:
            steps.append(
                SyncStepResult(name="cards", status="skipped", detail="Skipped via --skip-cards")
            )
        else:
            resolved_card_source: CardSource
            if card_source is not None:
                resolved_card_source = card_source
            elif cards_source_mode == "local":
                resolved_card_source = LocalCardSource(cards_path or (raw_dir / "cards"))
            else:
                resolved_card_source = PunkRecordsSource(
                    base_url=cards_base_url or settings.punk_records_url,
                    pack_ids=list(cards_packs) if cards_packs else None,
                )
            cards_summary = ingest_cards(conn, resolved_card_source)
            steps.append(
                SyncStepResult(
                    name="cards",
                    status="completed",
                    detail=(
                        f"upserted={cards_summary.cards_upserted}, "
                        f"variants_merged={cards_summary.variants_merged}, "
                        f"source={cards_summary.source_name}"
                    ),
                    metrics={
                        "cards_upserted": cards_summary.cards_upserted,
                        "raw_records_seen": cards_summary.raw_records_seen,
                        "variants_merged": cards_summary.variants_merged,
                        "chunks_upserted": cards_summary.chunks_upserted,
                        "source_name": cards_summary.source_name,
                    },
                )
            )

        # 5. banlist
        banlist_dir = raw_dir / "banlist"
        if skip_banlist:
            steps.append(
                SyncStepResult(
                    name="banlist",
                    status="skipped",
                    detail="Skipped via --skip-banlist",
                )
            )
        elif skip_fetch and not _has_supported_files(banlist_dir, _BANLIST_SUFFIXES):
            steps.append(
                SyncStepResult(
                    name="banlist",
                    status="skipped",
                    detail=(
                        f"No local banlist HTML/JSON files found under --skip-fetch: {banlist_dir}"
                    ),
                )
            )
        else:
            banlist_doc = parse_banlist_path(
                banlist_dir,
                include_upcoming=include_upcoming_banlist,
            )
            banlist_summary = ingest_banlist(conn, banlist_doc)
            steps.append(
                SyncStepResult(
                    name="banlist",
                    status="completed",
                    detail=(
                        f"upserted={banlist_summary.restrictions_upserted} "
                        f"(banned={banlist_summary.banned_count}, "
                        f"restricted={banlist_summary.restricted_count}, "
                        f"pair_banned={banlist_summary.pair_banned_count}), "
                        f"effective_date={banlist_summary.effective_date or 'unknown'}"
                    ),
                    metrics={
                        "restrictions_upserted": banlist_summary.restrictions_upserted,
                        "restrictions_removed": banlist_summary.restrictions_removed,
                        "banned_count": banlist_summary.banned_count,
                        "restricted_count": banlist_summary.restricted_count,
                        "pair_banned_count": banlist_summary.pair_banned_count,
                        "effective_date": banlist_summary.effective_date,
                    },
                )
            )

        # 6. faq
        faq_dir = raw_dir / "faq"
        if skip_faq:
            steps.append(
                SyncStepResult(name="faq", status="skipped", detail="Skipped via --skip-faq")
            )
        elif skip_fetch and not _has_supported_files(faq_dir, _FAQ_SUFFIXES):
            steps.append(
                SyncStepResult(
                    name="faq",
                    status="skipped",
                    detail=f"No local FAQ/errata files found under --skip-fetch: {faq_dir}",
                )
            )
        else:
            faq_docs = parse_faq_path(faq_dir)
            faq_summary = ingest_faqs(conn, faq_docs)
            steps.append(
                SyncStepResult(
                    name="faq",
                    status="completed",
                    detail=(
                        f"upserted={faq_summary.rulings_upserted} from "
                        f"{faq_summary.files_processed} file(s) "
                        f"(official_faq={faq_summary.official_faq_count}, "
                        f"errata={faq_summary.errata_count})"
                    ),
                    metrics={
                        "files_processed": faq_summary.files_processed,
                        "rulings_upserted": faq_summary.rulings_upserted,
                        "official_faq_count": faq_summary.official_faq_count,
                        "errata_count": faq_summary.errata_count,
                        "chunks_upserted": faq_summary.chunks_upserted,
                    },
                )
            )

        # 7. import-rulings
        explicit_rulings_path = rulings_path is not None
        resolved_rulings_path = rulings_path or (settings.db_path.parent / "rulings")
        if skip_rulings:
            steps.append(
                SyncStepResult(
                    name="import-rulings",
                    status="skipped",
                    detail="Skipped via --skip-rulings",
                )
            )
        elif not explicit_rulings_path and not _has_supported_files(
            resolved_rulings_path, _RULINGS_SUFFIXES
        ):
            steps.append(
                SyncStepResult(
                    name="import-rulings",
                    status="skipped",
                    detail=f"No curated rulings JSON files found at {resolved_rulings_path}",
                )
            )
        else:
            import_summary = import_rulings(conn, resolved_rulings_path)
            steps.append(
                SyncStepResult(
                    name="import-rulings",
                    status="completed",
                    detail=(
                        f"upserted={import_summary.rulings_upserted} from "
                        f"{import_summary.files_processed} file(s)"
                    ),
                    metrics={
                        "files_processed": import_summary.files_processed,
                        "rulings_upserted": import_summary.rulings_upserted,
                        "card_links_upserted": import_summary.card_links_upserted,
                        "rule_links_upserted": import_summary.rule_links_upserted,
                        "chunks_upserted": import_summary.chunks_upserted,
                    },
                )
            )

        # 8. embed
        check_embeddings = embeddings_available or is_embeddings_extra_installed
        if no_embed:
            steps.append(
                SyncStepResult(
                    name="embed",
                    status="skipped",
                    detail="Skipped via --no-embed (lexical search FTS5 BM25 is ready)",
                )
            )
        elif embedder is None and not check_embeddings():
            note = (
                "Optional [embeddings] extra is not installed; skipping local vector embeddings "
                "(install with: uv sync --extra embeddings). Lexical search (FTS5 BM25) is ready."
            )
            logger.info("%s", note)
            steps.append(SyncStepResult(name="embed", status="skipped", detail=note))
        else:
            if embedder is not None:
                active_embedder = embedder
            elif embedder_factory is not None:
                active_embedder = embedder_factory(settings)
            else:
                active_embedder = SentenceTransformerEmbedder(
                    model_name=settings.embedding_model,
                    device=settings.embedding_device,
                    use_fp16=settings.embedding_fp16,
                    batch_size=settings.embedding_batch_size,
                    dimension=settings.embedding_dim,
                )
            embed_summary = embed_chunks(
                conn,
                active_embedder,
                batch_size=settings.embedding_batch_size,
                force=force_embed,
            )
            steps.append(
                SyncStepResult(
                    name="embed",
                    status="completed",
                    detail=(
                        f"embedded={embed_summary.embedded_chunks}/{embed_summary.total_chunks}, "
                        f"skipped={embed_summary.skipped_chunks}, "
                        f"model={embed_summary.model_name}, dim={embed_summary.dimension}"
                    ),
                    metrics={
                        "total_chunks": embed_summary.total_chunks,
                        "embedded_chunks": embed_summary.embedded_chunks,
                        "skipped_chunks": embed_summary.skipped_chunks,
                        "orphans_removed": embed_summary.orphans_removed,
                        "rebuilt_vec_table": embed_summary.rebuilt_vec_table,
                        "model_name": embed_summary.model_name,
                        "dimension": embed_summary.dimension,
                    },
                )
            )

        snapshot = get_status(conn)

    return SyncSummary(
        db_path=settings.db_path,
        raw_data_dir=raw_dir,
        steps=tuple(steps),
        status_snapshot=snapshot,
    )
