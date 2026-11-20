"""Ingestion pipelines for OPTCG rules, cards, chunks, and local embeddings."""

from optcg_knowledge.ingest.banlist_parser import (
    OFFICIAL_BANLIST_URL,
    BanlistIngestSummary,
    ParsedBanlistDocument,
    ingest_banlist,
    parse_banlist_html,
    parse_banlist_json,
    parse_banlist_path,
)
from optcg_knowledge.ingest.cards_importer import CardsIngestSummary, ingest_cards, normalize_cards
from optcg_knowledge.ingest.chunking import (
    build_card_chunk,
    build_rule_chunk,
    build_ruling_chunk,
    build_trm_chunk,
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
from optcg_knowledge.ingest.faq_parser import (
    FaqIngestSummary,
    ParsedFaqDocument,
    ParsedFaqEntry,
    ingest_faqs,
    parse_faq_file,
    parse_faq_html,
    parse_faq_path,
    parse_faq_pdf,
    parse_faq_text,
)
from optcg_knowledge.ingest.official_downloader import (
    DEFAULT_OFFICIAL_BASE_URL,
    OfficialFetchSummary,
    fetch_official_docs,
)
from optcg_knowledge.ingest.rules_parser import (
    ParsedRulesDocument,
    RulesIngestSummary,
    ingest_rules,
    parse_rules_pdf,
    parse_rules_text,
)
from optcg_knowledge.ingest.trm_parser import (
    ParsedTrmDocument,
    TrmIngestSummary,
    ingest_trm,
    parse_trm_pdf,
    parse_trm_text,
)

__all__ = [
    "BanlistIngestSummary",
    "CardsIngestSummary",
    "DEFAULT_OFFICIAL_BASE_URL",
    "EmbedSummary",
    "Embedder",
    "FaqIngestSummary",
    "HashEmbedder",
    "OFFICIAL_BANLIST_URL",
    "OfficialFetchSummary",
    "ParsedBanlistDocument",
    "ParsedFaqDocument",
    "ParsedFaqEntry",
    "ParsedRulesDocument",
    "ParsedTrmDocument",
    "RulesIngestSummary",
    "SentenceTransformerEmbedder",
    "TrmIngestSummary",
    "build_card_chunk",
    "build_rule_chunk",
    "build_ruling_chunk",
    "build_trm_chunk",
    "compute_content_hash",
    "embed_chunks",
    "fetch_official_docs",
    "ingest_banlist",
    "ingest_cards",
    "ingest_faqs",
    "ingest_rules",
    "ingest_trm",
    "normalize_cards",
    "parse_banlist_html",
    "parse_banlist_json",
    "parse_banlist_path",
    "parse_faq_file",
    "parse_faq_html",
    "parse_faq_path",
    "parse_faq_pdf",
    "parse_faq_text",
    "parse_rules_pdf",
    "parse_rules_text",
    "parse_trm_pdf",
    "parse_trm_text",
    "serialize_vector",
    "upsert_chunks",
]
