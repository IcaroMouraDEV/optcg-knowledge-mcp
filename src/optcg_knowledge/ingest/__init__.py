"""Ingestion pipelines for OPTCG rules, cards, and chunks."""

from optcg_knowledge.ingest.cards_importer import (
    CardsIngestSummary,
    ingest_cards,
    normalize_cards,
    parse_card_id,
)
from optcg_knowledge.ingest.chunking import (
    build_card_chunk,
    build_rule_chunk,
    compute_content_hash,
    upsert_chunks,
)
from optcg_knowledge.ingest.embeddings import Embedder, HashEmbedder, serialize_vector
from optcg_knowledge.ingest.rules_parser import (
    ParsedRulesDocument,
    RulesIngestSummary,
    extract_pdf_text,
    ingest_rules,
    parse_rules_pdf,
    parse_rules_text,
)

__all__ = [
    "CardsIngestSummary",
    "Embedder",
    "HashEmbedder",
    "ParsedRulesDocument",
    "RulesIngestSummary",
    "build_card_chunk",
    "build_rule_chunk",
    "compute_content_hash",
    "extract_pdf_text",
    "ingest_cards",
    "ingest_rules",
    "normalize_cards",
    "parse_card_id",
    "parse_rules_pdf",
    "parse_rules_text",
    "serialize_vector",
    "upsert_chunks",
]
