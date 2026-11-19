---
name: ingestion-pipeline
description: >-
  Use this skill when building, updating, or debugging the optcg-ingest CLI
  pipelines (rules PDF parser, cards JSON importer, official FAQ/errata parser,
  local embedding generator, or export-rulings) and SQLite schema migrations.
---

# Ingestion Pipeline (`optcg-ingest`)

Implement deterministic, idempotent ingestion into SQLite (FTS5 + `sqlite-vec`) with pluggable sources.

## Steps

1. **Enforce source & legal boundaries**
   - Never fabricate or hardcode unverified external URLs. Before adding an online source (e.g., for `cards --source`), present candidate sources and ask the user for approval.
   - Always support manual local ingestion from `data/raw/` (`data/raw/rules/`, `data/raw/cards/`, `data/raw/faq/`, `data/raw/judge_chats/`) behind a pluggable source interface.
   - Never download, cache, or embed Bandai card images; persist only `image_url` alongside text and stats.

2. **Guarantee idempotency**
   - Every ingestion command (`rules`, `cards`, `faq`, `embed`) must be safe to re-run repeatedly on the same or updated inputs without creating duplicate rows or duplicate chunks.
   - Use deterministic primary keys (`rule_id`, `card_id`, stable hash/key for `ruling_id` and `chunk_id`) and SQLite `INSERT ... ON CONFLICT(...) DO UPDATE` inside explicit transactions.
   - Preserve existing `reviewed` and `superseded_by` states on rulings during re-ingestion unless explicitly overwritten.
   - Update the `meta` table (`rules_version`, `embedding_model`, `last_ingested_at`) atomically with each run.

3. **Build chunks according to the domain contract**
   - **Rule chunk:** one chunk per rule (`doc_type = "rule"`), prepending the full ancestor heading/rule path in `chunks.text` for hierarchical context.
   - **Ruling chunk:** one chunk per ruling (`doc_type = "ruling"`), combining `question` and `answer` in a single chunk with `authority` and linked card/rule IDs in `metadata`.
   - **Card chunk:** one chunk per card (`doc_type = "card"`), combining `card_id`, `name`, `category`, `colors`, `cost`, `power`, `counter`, `traits`, `effect_text`, and `trigger_text`.
   - Keep FTS5 (`chunks_fts`) synchronized with `chunks` (via triggers or transactional upsert/delete) so BM25 never goes stale.

4. **Handle local embeddings incrementally (`optcg-ingest embed`)**
   - Use local `sentence-transformers` with the model configurable via env var / CLI flag (default `BAAI/bge-m3` or `intfloat/multilingual-e5-base`).
   - Track a content hash (and model name) per chunk so `embed` only computes vectors for new or changed chunks—or rebuilds cleanly if the configured embedding model changed.
   - Abstract the embedder behind a small protocol/interface so tests inject a lightweight deterministic fake embedder without network access or multi-GB downloads.

5. **Validate**
   - Write `pytest` tests covering:
     - Parsing of representative fixtures (hierarchical rule numbering `7-1-4-1`, multi-line card effects/triggers, official FAQ Q&A entries).
     - **Idempotency check:** run the same ingestor twice against a temporary SQLite DB and assert identical row counts in `rules`/`cards`/`rulings`/`chunks`/FTS5/vec tables.
     - Incremental `embed` skips unchanged chunks and re-embeds modified ones.
   - Run `uv run pytest` and confirm all tests pass offline.
