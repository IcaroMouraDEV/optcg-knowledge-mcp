# optcg-knowledge — Project Rules

MCP server in Python providing One Piece Card Game (OPTCG) rules, cards, official FAQs, and judge rulings with hybrid search. Full specification lives in [PROMPT.md](PROMPT.md).

## Hard Boundaries

- **Data-only MCP server:** never call paid or external LLM/embedding APIs from server or ingestion code. Embeddings are strictly local via `sentence-transformers` (configurable, e.g., `BAAI/bge-m3` or `intfloat/multilingual-e5-base`).
- **No game engine:** never add effect parsers, DSLs, board state, rule validation, or simulation logic here. That belongs to a separate `optcg-engine` repo.
- **Interoperability first:** keep card IDs (`OP05-119`), rule IDs (`7-1-4-1`), and `export_rulings` JSON schemas stable.
- **Copyright & privacy:**
  - Never download or host Bandai card images—store text/stats and `image_url` only.
  - Never store person names, handles, or user IDs in judge rulings—store only the content and role `"judge"`.
- **No invented sources:** never fabricate source URLs. Ask the user before adopting any external data source, and always support local files under `data/raw/`.

## Authority Hierarchy (Central Domain Invariant)

| `authority` | `source` | Meaning |
| :--- | :--- | :--- |
| `1` | `rules` | Comprehensive Rules (highest) |
| `2` | `official_faq` / `errata` | Official card FAQs and errata |
| `3` | `judge` | Official Bandai judge group rulings |
| `4` | `community` | Community rulings (optional, lowest) |

- Never delete outdated rulings: retain `date` and `rules_version`, and link `superseded_by`.
- Exclude superseded rulings by default in search and lookup tools unless `include_superseded=True`.

## Stack & Commands

- **Runtime:** Python 3.12 managed with `uv` (`pyproject.toml`).
- **MCP:** official `mcp` SDK (`FastMCP`) over `stdio` transport—**never write logs or debug output to `stdout`** (use `stderr` / Python `logging` to `stderr`).
- **Database:** single SQLite file with FTS5 (BM25) + `sqlite-vec` (vectors).
- **CLI:** `optcg-ingest` (`rules`, `cards`, `faq`, `embed`, `export-rulings`).
- **Lint & Format:** `uv run ruff check` (enable rule `T201` in `pyproject.toml` to forbid `print()` outside CLI modules and protect MCP `stdio` frames).
- **Tests:** `uv run pytest` (must run fast and offline without downloading large embedding models).

## Workflow

Work strictly in phases and **stop at the end of each phase for user review**:
1. Project structure, SQLite schema, rules & cards ingestion, basic tests.
2. Hybrid search + query MCP tools + `juiz` prompt.
3. Official FAQs / errata ingestion (`authority = 2`).
4. Judge ruling curation + `curar` prompt + `export_rulings`.

If a requirement, external data source, format, or naming choice is ambiguous, ask before assuming.
