# optcg-knowledge

Local, data-only MCP server (Python 3.12 + `stdio`) providing a searchable knowledge base for the **One Piece Card Game (OPTCG)**: Comprehensive Rules, Tournament Rules Manual (TRM), canonical cards with official banlist/restriction status, official Bandai FAQs & errata, and curated judge rulings with hybrid search (SQLite FTS5 BM25 + `sqlite-vec` local embeddings + Reciprocal Rank Fusion).

`optcg-knowledge` never calls external/paid LLM or embedding APIs and contains no game-engine simulation logic. The connected agent (Claude Code, Antigravity, Gemini CLI) performs reasoning using structured tools and prompts exposed by this server.

---

## Authority Hierarchy

Every rule and ruling follows a strict authority order:

| `authority` | `source` | Description |
| :--- | :--- | :--- |
| `1` | `rules` / `tournament_rules` | Official Comprehensive Rules and Tournament Rules Manual (highest) |
| `2` | `official_faq` / `errata` | Official Bandai card FAQs and errata |
| `3` | `judge` | Certified Bandai judge rulings (`tier_0 > tier_2 > tier_1`) |
| `4` | `community` | Community rulings (optional, lowest) |

- **Judge tiers (within `authority = 3`):** `tier_0` (highest) > `tier_2` (senior) > `tier_1` (basic). Only the role (`"judge"`) and optional `judge_tier` are persisted—personal names, Discord handles, and user IDs are never stored.
- **No silent deletion:** outdated rulings retain `date` and `rules_version` and are linked via `superseded_by` (excluded by default unless `include_superseded=True`).
- **Interoperability:** canonical `card_id` (`OP05-119`, `ST10-001`, `P-001`), canonical `rule_id` (`7-1-4-1`), and versioned `export_rulings` JSON (`schema_version = "1.0"`) are designed for downstream projects such as an external `optcg-engine`.

---

## Installation

Requires **Python 3.12+** and [`uv`](https://docs.astral.sh/uv/).

```bash
# Core installation (CLI, SQLite FTS5 + sqlite-vec, lexical search, MCP server)
uv sync

# Optional: install local sentence-transformers + PyTorch for hybrid vector search
uv sync --extra embeddings
```

Copy `.env.example` to `.env` if you want to customize paths or local embedding settings:

```bash
cp .env.example .env
```

Default local embedding model is `BAAI/bge-m3` (`1024` dims, FP16 on CUDA with `batch_size=8` to fit 4 GB VRAM GPUs, automatic CPU fallback). You can switch to `intfloat/multilingual-e5-base` (`768` dims) via `OPTCG_EMBEDDING_MODEL`.

---

## Quick Start (One-Command Sync)

After installing dependencies with `uv sync` (or `uv sync --extra embeddings` for local hybrid vector search), populate the entire local SQLite knowledge base in a single idempotent command:

```bash
uv run optcg-ingest sync
```

`optcg-ingest sync` orchestrates every stage in order:
1. `fetch-official` — downloads official Bandai Comprehensive Rules PDF, TRM PDF, FAQ PDFs, errata HTML, and banlist HTML (skips unchanged files; pass `--skip-fetch` to reuse local `data/raw/` files)
2. `rules` & `trm` — ingests Comprehensive Rules and Tournament Rules Manual (`authority = 1`)
3. `cards` — imports canonical cards from `punk-records` (pass `--skip-cards` or `--cards-source local`)
4. `banlist` — ingests official Bandai banlist / card restrictions
5. `faq` — ingests official Bandai FAQs & errata (`authority = 2`)
6. `import-rulings` — imports tracked curated judge rulings from `data/rulings/`
7. `embed` — incrementally generates local embeddings (automatically skipped with a friendly note if the optional `[embeddings]` extra is not installed, or pass `--no-embed` for lexical-only FTS5 BM25)

Inspect database health, versions, row counts, embedding coverage, and timestamps at any time:

```bash
uv run optcg-ingest status
# Or emit machine-readable JSON matching the MCP get_status() contract:
uv run optcg-ingest status --json
```

---

## Individual Ingestion Commands (`optcg-ingest`)

All ingestion commands are **idempotent** and safe to re-run without duplicating rows or re-embedding unchanged chunks. Raw Bandai PDFs/HTML and chat dumps live under `data/raw/` (gitignored).

### 1. Download official Bandai rules, TRM, FAQ PDFs, errata HTML, and banlist HTML
Downloads the official Comprehensive Rules PDF (`data/raw/rules/rule_comprehensive.pdf`), Tournament Rules Manual PDF (`data/raw/rules/tournament_rules_manual.pdf`), all official FAQ PDFs (`data/raw/faq/*.pdf`), the text-only errata HTML page (`data/raw/faq/errata_card.html`), and the text-only restriction/banlist HTML page (`data/raw/banlist/restriction.html`) directly from Bandai's English website (skips unchanged files on re-run, never downloads card images):
```bash
uv run optcg-ingest fetch-official
# Or fetch only one category: --only rules | trm | faq | errata | banlist
```

### 2. Ingest Comprehensive Rules & Tournament Rules Manual (`authority = 1`)
```bash
uv run optcg-ingest rules data/raw/rules/rule_comprehensive.pdf
uv run optcg-ingest trm data/raw/rules/tournament_rules_manual.pdf
```

### 3. Ingest canonical cards (text/stats + `image_url` only)
From `punk-records` JSON packs:
```bash
uv run optcg-ingest cards --source punk-records
# Optional subset of packs:
uv run optcg-ingest cards --source punk-records --packs OP01,OP05,ST10
```
Or from local JSON files in `data/raw/cards/`:
```bash
uv run optcg-ingest cards --source local --path data/raw/cards
```

### 4. Ingest official banlist / card restrictions
Parses `data/raw/banlist/` (official Bandai HTML `restriction.html` and/or manual local `.json` files) into `card_restrictions` (separating active restrictions from scheduled future announcements) without touching `chunks_vec`:
```bash
uv run optcg-ingest banlist
# Optional: skip scheduled future-effective announcements from the official HTML
uv run optcg-ingest banlist --no-include-upcoming
```

### 5. Ingest official FAQs & errata (`authority = 2`)
```bash
uv run optcg-ingest faq data/raw/faq/
```
Supports single files or directories containing `.pdf` (Bandai tabular Excel-exported PDFs and `Q:`/`A:`/`Before:`/`After:` PDFs), `.html`/`.htm` (official `errata_card` page), and `.txt`.

Optional flags: `--source auto|official_faq|errata`, `--date YYYY-MM-DD`, `--rules-version 1.24`, `--url https://...`.

### 6. Generate local embeddings (incremental)
```bash
uv run optcg-ingest embed
```
Useful flags: `--batch-size 8`, `--device auto|cuda|cpu`, `--fp16/--no-fp16`, `--force`. Only new or modified chunks (tracked via `content_hash`) are embedded.

### 7. Import & export curated rulings (collaborative JSON workflow)
Tracked, anonymized judge rulings live in `data/rulings/` (`TEMPLATE.md`, `threads/`, and `judge_rulings.json`) so collaborators can contribute new rulings via Pull Requests without committing raw Bandai PDFs or un-anonymized chat logs:

```bash
# Import curated judge rulings from the repository into your local DB
uv run optcg-ingest import-rulings data/rulings/judge_rulings.json
uv run optcg-ingest embed

# Export only curated judge rulings back to the repository after curating & approving
uv run optcg-ingest export-rulings --source judge --out data/rulings/judge_rulings.json

# Export vetted rulings for downstream consumers (e.g., optcg-engine)
uv run optcg-ingest export-rulings --card-id OP05-119 --min-authority 3 --since 2025-01-01 --out rulings.json
```
By default `export-rulings` exports only `reviewed = True`, non-superseded rulings. Pass `--include-superseded` and/or `--include-unreviewed` to include them.

---

## Judge Ruling Curation Workflow

1. Save raw Discord forum threads under `data/raw/judge_chats/` (gitignored) using the lightweight thread template:

```markdown
---
Thread: OP05-119 extra turn vs OP04-083 Sabo
Date: 2026-11-18

[PLAYER]
<Question copied from forum post>

[JUDGE T2]
<Judge answer>

[JUDGE T0]
<Optional higher-tier clarification>
---
```

2. Trigger the MCP prompt `curar(arquivo="2026-11-judge-forum.md")` in your agent. It extracts Q&A pairs, strips all personal names/`@handles`/Discord IDs, verifies canonical `card_ids` and `rule_refs`, applies judge tier precedence (`tier_0 > tier_2 > tier_1`), and calls `add_ruling(...)` (entering with `reviewed = False`).
3. Review pending entries via `list_unreviewed_rulings`, approve via `mark_ruling(ruling_id, reviewed=True)`, link outdated rulings via `mark_ruling(old_id, superseded_by=new_id)`, and run `uv run optcg-ingest embed`.

---

## MCP Tools & Prompts

| Category | Name | Purpose |
| :--- | :--- | :--- |
| **Query** | `search` | Hybrid search (exact OPTCG tokens + FTS5 BM25 + `sqlite-vec` + RRF + authority/tier tie-breaking) |
| **Query** | `get_rule` | Look up a Comprehensive Rule by `rule_id` with ancestors and ordered children |
| **Query** | `get_trm_section` | Look up a Tournament Rules Manual section by `section_id` (`TRM-7.4.1`) with ancestors and children |
| **Query** | `get_card` | Look up canonical card(s) by exact `card_id` or fuzzy name, including active banlist legality and `upcoming_restriction` |
| **Query** | `get_cards` | Batch look up multiple `card_ids` with pagination (`page`, `page_size`, `has_more`, `next_page`), banlist legality, and optional `include_rulings=True` |
| **Query** | `search_cards` | Filter cards by `name`, `color`, `cost`, `trait`, `keyword`, `set_code`, and active `ban_status` |
| **Query** | `get_banlist` | Retrieve the official Bandai banlist (`banned`, `restricted`, `pair_banned`) and separate `upcoming` announcements in one call |
| **Query** | `get_card_rulings` | List rulings linked to a `card_id` ordered by `authority`, `judge_tier`, `date DESC` |
| **Query** | `get_rule_rulings` | List rulings citing a Comprehensive Rule `rule_id` (and optional sub-rules via `include_children=True`) ordered by `authority`, `judge_tier`, `date DESC` |
| **Curation** | `add_ruling` | Insert anonymized curated ruling (`reviewed=False`), indexed immediately in FTS5 |
| **Curation** | `mark_ruling` | Update `reviewed`, `superseded_by` (cycle-checked), and/or `judge_tier` |
| **Curation** | `list_unreviewed_rulings` | Inspect curated rulings awaiting manual approval |
| **Diagnostics** | `get_status` | Report DB schema/rules/TRM/banlist versions, table/authority/tier counts, embedding coverage, and timestamps |
| **Interop** | `export_rulings` | Export vetted rulings using the exact same versioned JSON schema as the CLI |
| **Prompt** | `juiz` | OPTCG rules judge workflow citing authority, exact IDs, banlist legality, conflicts, and staleness |
| **Prompt** | `curar(arquivo)` | Anonymized judge chat curation workflow targeting `data/raw/judge_chats/<arquivo>` |

---

## Connecting the MCP Server

### Claude Code

Register the local stdio server with `claude mcp add`:

```bash
claude mcp add optcg-knowledge -- \
  /absolute/path/to/optcg-knowledge/.venv/bin/optcg-mcp \
  --db /absolute/path/to/optcg-knowledge/data/optcg.db
```

Or via `uv`:

```bash
claude mcp add optcg-knowledge -- \
  uv --directory /absolute/path/to/optcg-knowledge run optcg-mcp \
  --db /absolute/path/to/optcg-knowledge/data/optcg.db
```

### Antigravity

#### Option A — Workspace-scoped plugin (recommended for project-only use)

Create `.agents/plugins/optcg-knowledge/plugin.json`:
```json
{
  "name": "optcg-knowledge",
  "description": "Local One Piece Card Game rules, cards, FAQs, and judge rulings MCP server."
}
```

And `.agents/plugins/optcg-knowledge/mcp_config.json`:
```json
{
  "mcpServers": {
    "optcg-knowledge": {
      "command": "/absolute/path/to/optcg-knowledge/.venv/bin/optcg-mcp",
      "args": [
        "--db",
        "/absolute/path/to/optcg-knowledge/data/optcg.db"
      ]
    }
  }
}
```

#### Option B — Global configuration

Add the `optcg-knowledge` entry above inside `mcpServers` in `~/.gemini/config/mcp_config.json`.

---

## Development & Tests

Tests run fast and **100% offline** (no large embedding model downloads):

```bash
uv run ruff check
uv run pytest -v
```

---

## License & Disclaimer

- **License:** This project's source code and community-curated rulings under `data/rulings/` are licensed under the [MIT License](LICENSE).
- **Third-party dependency note:** PDF parsing uses `PyMuPDF` (`pymupdf`), which is dual-licensed under **AGPL-3.0** / commercial license. Review AGPL-3.0 obligations if you distribute bundled binaries or operate a hosted network service using `PyMuPDF`.
- **Disclaimer:** `optcg-knowledge` is an unofficial, non-commercial community project and is not affiliated with, endorsed by, or sponsored by Bandai Co., Ltd., Eiichiro Oda, Shueisha, or Toei Animation. One Piece and the One Piece Card Game are trademarks and copyrights of their respective owners. This repository does not host or redistribute official Bandai PDFs or card images.
