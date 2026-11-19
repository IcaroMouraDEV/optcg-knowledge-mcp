---
name: mcp-tool-authoring
description: >-
  Use this skill when creating, modifying, or testing FastMCP tools, resources,
  or MCP prompts (search, get_rule, get_card, search_cards, get_card_rulings,
  add_ruling, mark_ruling, list_unreviewed_rulings, export_rulings, juiz, curar)
  in optcg-knowledge.
---

# MCP Tool & Prompt Authoring

Build and maintain clean, interoperable `FastMCP` (`stdio`) tools and prompts for `optcg-knowledge`.

## Steps

1. **Preserve stdio protocol integrity**
   - Never use `print()` or write anything to `stdout` inside server or DB code—`stdout` carries MCP JSON-RPC frames.
   - Route all diagnostics and library warnings to `sys.stderr` (or configure `logging` with `stream=sys.stderr`).

2. **Enforce domain & interoperability contracts**
   - Keep identifiers canonical and stable: `card_id` (e.g., `OP05-119`, `ST10-001`, `P-001`), `rule_id` (e.g., `7-1-4-1`), and `ruling_id`.
   - Return structured, predictable dicts/lists (JSON-serializable) rather than opaque prose blobs so downstream consumers like `optcg-engine` can parse them deterministically.
   - Respect default safety flags:
     - `search(...)`: `k=8`, `include_superseded=False` by default.
     - `get_rule(rule_id, include_children=True)`: return the rule and ordered descendants when requested.
     - `get_card(id_or_name)`: exact `card_id` match first, then fuzzy name match returning all printings/versions.
     - `get_card_rulings(card_id)`: sort by `authority ASC, date DESC`; exclude superseded unless explicitly requested.
     - `add_ruling(...)`: always insert with `reviewed=False`, validate `source` -> `authority` mapping, and never accept or persist judge personal names.
     - `mark_ruling(ruling_id, reviewed=None, superseded_by=None)`: reject self-superseding (`ruling_id == superseded_by`) and verify target IDs exist.
     - `export_rulings(card_id=None, since=None, min_authority=None)`: emit a stable, versioned JSON envelope shared with `optcg-ingest export-rulings` (no duplicate serialization logic).

3. **Write high-signal MCP tool/prompt metadata**
   - Give every tool concise type annotations and a clear docstring describing parameters, authority semantics, and edge cases.
   - For the `juiz` prompt: instruct the consuming agent to query `search` / `get_card` / `get_card_rulings`, follow `authority` order (`1 > 2 > 3 > 4`), cite exact rule/ruling IDs and sources, flag conflicts explicitly, warn if a ruling predates the current `rules_version` in `meta`, and optionally use an engine simulation tool if one is connected without depending on it.
   - For the `curar(arquivo)` prompt: read from `data/raw/judge_chats/`, extract judge Q&A pairs, strip personal names, link `card_ids` and `rule_refs`, and call `add_ruling`.

4. **Validate**
   - Add or update `pytest` tests exercising the tool directly against a temporary SQLite DB (happy path, missing ID, superseded filtering, and JSON contract keys).
   - Verify no output leaks to `stdout` during server initialization or tool calls.
   - Run `uv run pytest` and confirm all tests pass.
