"""Ingestion pipelines for OPTCG rules, cards, chunks, and local embeddings."""

from optcg_knowledge.ingest.cards_importer import CardsIngestSummary, ingest_cards, normalize_cards
from optcg_knowledge.ingest.chunking import (
    build_card_chunk,
    build_rule_chunk,
    compute_content_hash,
    upsert_chunks,
)
from optcg_knowledge.ingest.embed_pipeline import EmbedSummary, embed_chunks
from optcg_knowledge.ingest.embeddings import (
    Embedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    serialize_vector,
)
from optcg_knowledge.ingest.rules_parser import (
    ParsedRulesDocument,
    RulesIngestSummary,
    ingest_rules,
    parse_rules_pdf,
    parse_rules_text,
)

__all__ = [
    "CardsIngestSummary",
    "EmbedSummary",
    "Embedder",
    "HashEmbedder",
    "ParsedRulesDocument",
    "RulesIngestSummary",
    "SentenceTransformerEmbedder",
    "build_card_chunk",
    "build_rule_chunk",
    "compute_content_hash",
    "embed_chunks",
    "ingest_cards",
    "ingest_rules",
    "normalize_cards",
    "parse_rules_pdf",
    "parse_rules_text",
    "serialize_vector",
    "upsert_chunks",
]
