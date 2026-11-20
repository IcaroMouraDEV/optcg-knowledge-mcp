## Summary
<!-- Briefly describe what this PR changes and why. -->

Closes #

## Type of Change
- [ ] Curated judge ruling(s) (`data/rulings/`)
- [ ] Feature / Enhancement
- [ ] Bug fix
- [ ] Documentation / Chore

## Checklist
- [ ] **Privacy (PII):** No personal names, Discord usernames, `@handles`, `user#0000` tags, or raw user IDs (`<@...>`) in rulings, threads, or commit history.
- [ ] **Copyright:** No official Bandai PDFs, card images, or `data/*.db` files committed.
- [ ] **Domain invariants:** Authority hierarchy (`1=rules > 2=official_faq/errata > 3=judge > 4=community`) preserved; outdated rulings linked via `superseded_by` instead of deleted.
- [ ] **Checks (when touching Python code):** `uv run ruff check` and `uv run pytest -v` pass offline (no `print()` outside `src/optcg_knowledge/cli.py`).
