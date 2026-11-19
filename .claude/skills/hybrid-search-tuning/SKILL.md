---
name: hybrid-search-tuning
description: >-
  Use this skill when implementing, debugging, or tuning hybrid search in
  optcg-knowledge (SQLite FTS5 BM25, sqlite-vec vector search, Reciprocal Rank
  Fusion, authority boosting, and OPTCG exact-match detection for card IDs,
  rule numbers, and bracketed keywords).
---

# Hybrid Search & Ranking (`FTS5` + `sqlite-vec` + `RRF`)

Design and verify accurate, crash-free hybrid retrieval tailored to One Piece Card Game queries.

## Steps

1. **Detect OPTCG exact-match tokens in the query**
   - Parse the incoming query before running BM25/vector retrieval to extract structured tokens:
     - **Card IDs:** e.g., `\b(?:OP|ST|EB|PRB)\d{2}-\d{3}\b` and promo IDs `\bP-\d{3}\b` (case-insensitive, normalized uppercase).
     - **Rule numbers:** e.g., `\b\d+(?:-\d+){1,4}\b` (such as `7-1-4` or `7-1-4-1`).
     - **Bracketed keywords:** e.g., `\[[^\]]+\]` (such as `[Blocker]`, `[On K.O.]`, `[Rush]`, `[DON!! x1]`, `[Trigger]`).
   - Prioritize exact matches (e.g., direct lookup on `doc_id` / metadata / exact substring match on `chunks.text`) ahead of purely fuzzy hits.

2. **Sanitize FTS5 queries safely**
   - Never pass raw user input directly into `FTS5 MATCH`—hyphens (`OP05-119`, `7-1-4`), brackets (`[On K.O.]`), periods, exclamation marks (`[DON!! x1]`), and colons trigger FTS5 syntax errors or `NOT` operator misinterpretation.
   - Quote phrase/token terms safely (escaping embedded `"` as `""`) and ensure empty or punctuation-only queries degrade gracefully without raising `sqlite3.OperationalError`.

3. **Combine signals with RRF and authority tie-breaking**
   - Run candidate retrieval from both FTS5 (BM25) and `sqlite-vec` (cosine/L2 distance) with filters applied (`doc_types`, `min_authority`, and `superseded_by IS NULL` unless `include_superseded=True`).
   - Fuse rankings using Reciprocal Rank Fusion:
     `rrf_score(d) = sum(1 / (K + rank_i(d)))` (e.g., `K = 60`), plus a clear boost for exact OPTCG token matches.
   - Apply a gentle authority boost (`1=rules > 2=official_faq/errata > 3=judge > 4=community`) so higher-authority sources win ties without burying highly relevant card FAQs or rulings.

4. **Test deterministically and offline**
   - Never require downloading `BAAI/bge-m3` or network access during `uv run pytest`.
   - Use an injected deterministic fake embedder (e.g., fixed-dimension hashed bag-of-words / n-gram vector) for search tests.
   - Maintain golden search regression tests asserting top results for:
     - Exact card ID (`OP05-119`, `ST10-001`)
     - Exact rule number (`7-1-4-1` and prefix `7-1-4`)
     - Bracketed keyword (`[Blocker]`, `[On K.O.]`, `[DON!! x1]`)
     - Authority ordering on tied/conflicting hits
     - Superseded rulings excluded by default and included when `include_superseded=True`
     - Special characters in queries never causing FTS5 syntax errors.

5. **Validate**
   - Run `uv run pytest` and verify all search ranking and edge-case tests pass.
