---
name: optcg-ruling-curation
description: >-
  Use this skill when curating judge chat exports in data/raw/judge_chats/,
  structuring Q&A pairs into rulings, reviewing unreviewed rulings, or marking
  outdated rulings as superseded_by in optcg-knowledge.
---

# OPTCG Ruling Curation

Convert raw judge chat exports and FAQ updates into clean, anonymized, traceable rulings.

## Steps

1. **Read raw source safely**
   - Read the target export from `data/raw/judge_chats/` (or inspect pending items via `list_unreviewed_rulings`).
   - Keep a minimal `raw_ref` (file name + line/message range or timestamp) for traceability without embedding raw chat logs or user identities.

2. **Extract and anonymize judge Q&A pairs**
   - Identify genuine ruling exchanges (player question + judge answer). Skip off-topic chatter, unconfirmed speculation, or unanswered questions.
   - **Strict anonymization:** strip all real names, nicknames, `@handles`, Discord tags, phone numbers, and user IDs from both `question` and `answer`. Record only `source = "judge"` (`authority = 3`).
   - Preserve exact OPTCG game terminology, phase names, and bracketed keywords (`[On Play]`, `[Blocker]`, `[Once Per Turn]`, `[DON!! x1]`) faithfully—never invent or extrapolate mechanics not stated in the answer.

3. **Resolve and validate entity links**
   - Normalize referenced cards to canonical `card_id`s (e.g., `OP05-119`) and verify them against the `cards` table (`get_card` / `search_cards`) before linking in `card_ids`.
   - Normalize referenced rules to canonical `rule_id`s (e.g., `7-1-4-1`) and verify them against `rules` (`get_rule`) before linking in `rule_refs`.
   - Record the ruling's `date` (`YYYY-MM-DD`) and current `rules_version` from `meta` so future queries can flag potentially stale rulings.

4. **Store, review, and handle superseding**
   - Insert newly curated rulings via `add_ruling(...)` so they enter with `reviewed = False`.
   - When reviewing (`list_unreviewed_rulings`), verify accuracy, anonymization, and links before calling `mark_ruling(ruling_id, reviewed=True)`.
   - If a newer Comprehensive Rules update, official FAQ/errata, or newer judge ruling overrides an older ruling, **never delete** the old ruling—call `mark_ruling(old_id, superseded_by=new_id)`.

5. **Validate**
   - Spot-check inserted rulings to confirm zero personal names/handles appear in `question`, `answer`, or `raw_ref`.
   - Confirm `export_rulings` produces valid, deterministic JSON containing the curated rulings and linked `card_ids` / `rule_refs`.
   - Run `uv run pytest` if curation helper code or prompt templates were modified.
