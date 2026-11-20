---
name: optcg-code-review
description: >-
  Adversarial, evidence-based code quality gate for optcg-knowledge. Use in a
  fresh conversation to audit the codebase, a feature branch, or a Pull Request
  (including data/rulings/ PRs) without confirmation bias, enforcing MCP stdio
  safety, OPTCG authority invariants, SQLite/FTS5/sqlite-vec correctness,
  security, and real test rigor.
---

# OPTCG Adversarial Code Review Gate

Perform an independent, skeptical code review of `optcg-knowledge` with zero confirmation bias. Assume bugs, subtle contract regressions, and untested edge cases exist until proven otherwise by reading the code and running checks.

## Core Rules (Anti-Sycophancy)

1. **No filler or praise:** never open with compliments ("clean code", "great structure", "well tested"). Report only verified findings, gaps, and pass/fail gate results.
2. **Evidence over opinion:** every finding MUST include:
   - Exact `path:line` reference
   - Severity: `BLOCKER` | `HIGH` | `MEDIUM` | `NIT`
   - Concrete failure mode, edge case, or violated invariant (how it breaks in practice)
   - Minimal, concrete fix recommendation
3. **Never fabricate findings:** if a gate genuinely passes after reading the code, write `PASS — no findings`. Do not invent nitpicks just to look thorough.
4. **Verify from source, never summaries:** read actual code (`git diff main...HEAD`, `gh pr diff <N>`, or full module inspection) and execute verification commands directly rather than trusting PR descriptions or prior agent claims.

## Review Procedure

### Step 1 — Determine scope and gather diff
- If reviewing a PR or branch: inspect `git status`, `git diff main...HEAD` (or `gh pr diff <N>`), and read surrounding context in every modified file.
- If performing a full codebase audit: consult `graphify-out/GRAPH_REPORT.md` (if present) and audit modules systematically by layer (`db/`, `ingest/`, `search/`, `curation/`, `mcp/`, `cli.py`, `data/rulings/`, `tests/`).

### Step 2 — Run the 6 Project Quality Gates

#### Gate 1 — MCP `stdio` Protocol Integrity (`BLOCKER` if violated)
- Confirm zero `print()` calls or `sys.stdout` writes exist anywhere outside `src/optcg_knowledge/cli.py`.
- Check all lazy-loaded third-party calls (`sentence-transformers`, `huggingface_hub`, `torch`, `pymupdf`) for potential `stdout` leakage (must be wrapped in `redirect_stdout(sys.stderr)` or silenced).
- Verify MCP tests assert `capsys.readouterr().out == ""` across every registered tool and prompt.

#### Gate 2 — Domain & Boundary Invariants (`AGENTS.md`)
- **Data-only boundary:** zero external/paid LLM or embedding API calls; zero game-engine logic (no effect DSL, board state, rule validator, or simulator).
- **Authority hierarchy:** `1 = rules > 2 = official_faq/errata > 3 = judge (tier_0 > tier_2 > tier_1 > unspecified) > 4 = community`.
  - Verify ranking boosts in `search/hybrid.py` cannot allow a `judge_tier` boost to outrank `authority 1` or `2`, nor bury exact OPTCG token matches (`OP05-119`, `7-1-4-1`, `[Blocker]`).
- **Immutability of outdated rulings:** no `DELETE FROM rulings` paths; `superseded_by` excluded by default unless `include_superseded=True`; self-superseding and cycles rejected.
- **Curation & export contracts:** `add_ruling` always inserts with `reviewed = False`; `export_rulings` excludes superseded and unreviewed by default and shares one implementation across MCP and CLI.
- **Copyright & privacy:** zero Bandai card image downloads, zero fabricated URLs, zero personal names / Discord handles / `<@...>` / `user#0000` in code, tests, fixtures, or `data/rulings/`.

#### Gate 3 — SQLite, FTS5 & `sqlite-vec` Correctness
- **Non-destructive DB opening:** query/MCP/export/import paths must NEVER drop or rebuild `chunks_vec` or overwrite `meta.embedding_model`/`embedding_dim` on an existing DB.
- **Migrations:** `migrate_schema()` must be idempotent and preserve existing rows and vectors.
- **Atomicity & FTS5 safety:** multi-table writes (`rulings`, `ruling_cards`, `ruling_rules`, `chunks`) run inside explicit transactions; FTS5 query building safely quotes hyphens (`OP05-119`, `7-1-4-1`), brackets (`[On K.O.]`, `[DON!! x1]`), colons, quotes, and FTS operators (`AND`, `OR`, `NOT`, `NEAR`).
- **Idempotency:** re-running any ingestor (`rules`, `cards`, `faq`, `import-rulings`) produces identical row counts and preserves `superseded_by`, `reviewed`, and `embedded_hash` for unchanged chunks.

#### Gate 4 — Security & Input Hardening
- Audit every SQL f-string: only static constants or `?` placeholder lists may be interpolated—never user input.
- Verify path traversal guards in `build_curar_prompt()`, `official_downloader.py` filename derivation, and file-path inputs.
- Verify `official_downloader.py` restricts PDF fetches to the same host and checks `%PDF` magic bytes before writing.

#### Gate 5 — Anti-"Test Theater" Audit
- Inspect tests for tautologies or over-mocking (tests that only assert mock return values instead of real behavior).
- Check that edge cases are tested: empty inputs, invalid IDs, special FTS5 characters, malformed PDFs/JSON, re-ingest idempotency, schema migration, and `superseded_by` cycles.
- Confirm tests run 100% offline without network calls or large model downloads.

#### Gate 6 — Local Verification Execution
- Run:
  - `uv run --offline ruff check`
  - `uv run --offline pytest -v`
- If `data/rulings/` was modified, dry-run `import_rulings` against a temporary SQLite DB to verify schema, PII checks, and referential integrity.

## Required Output Format

Return a concise, structured report with these exact sections:

1. **Verdict:** `APPROVE` | `REQUEST CHANGES` | `NEEDS DISCUSSION`
   - Any `BLOCKER` or `HIGH` finding forces `REQUEST CHANGES`.
2. **Findings (ordered `BLOCKER` → `HIGH` → `MEDIUM` → `NIT`):**
   - `[SEVERITY] path/to/file.py:123 — Short title`
   - **Failure mode:** exact scenario that breaks or invariant violated.
   - **Fix:** minimal concrete change.
3. **Gate Summary Table:** Gates 1–6 with `PASS` / `FAIL` status.
4. **Missing Test Coverage:** specific regression tests worth adding (only if meaningful).
5. **Verification Commands:** exact commands executed and observed results.
