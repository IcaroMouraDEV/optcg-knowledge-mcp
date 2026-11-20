---
name: optcg-ruling-curation
description: >-
  Use this skill when curating Discord judge forum threads or chat exports in
  data/raw/judge_chats/, formatting pasted Discord threads into anonymized
  markdown templates, structuring Q&A pairs into rulings with judge_tier
  precedence, reviewing unreviewed rulings, or marking outdated rulings as
  superseded_by in optcg-knowledge.
---

# OPTCG Ruling Curation

Convert raw Discord judge forum threads and chat exports into clean, anonymized, traceable rulings with high signal and zero PII.

## Steps

1. **Read or normalize the raw source safely**
   - Accept either an existing file under `data/raw/judge_chats/` or a Discord forum thread pasted directly by the user.
   - When the user pastes a raw Discord thread, optionally save a clean anonymized copy under `data/raw/judge_chats/YYYY-MM-DD-<slug>.md` using the project thread format (`---` separators, `Thread:`, `Date:`, `Cards:`, `Rules:`, and role markers `[PLAYER]`, `[JUDGE T0]`, `[JUDGE T2]`, `[JUDGE T1]`, `[JUDGE]`).
   - Keep a minimal `raw_ref` (e.g., `2026-11-18-captain-john-rested.md#thread-1`) without embedding raw chat logs or user identities.

2. **Extract Q&A pairs, classify roles, and enforce judge tier precedence**
   - Identify genuine ruling exchanges (player question + certified OPTCG judge answer). Skip off-topic chatter, thank-yous, unconfirmed player speculation, and non-OPTCG game roles (e.g., `DCG Lvl 1` is Digimon, not an OPTCG judge ruling—treat follow-up questions from other games as `[PLAYER]` context only).
   - Map official OPTCG judge tiers within `authority = 3` (`source = "judge"`):
     - `Tier 0` / `[JUDGE T0]` -> `judge_tier = "tier_0"` (highest judge authority)
     - `Tier 2` / `[JUDGE T2]` -> `judge_tier = "tier_2"` (highest normally attainable judge tier)
     - `Tier 1` / `OPCG Lvl 1` / `[JUDGE T1]` -> `judge_tier = "tier_1"` (basic certified judge)
     - Unknown or unconfirmed roles (e.g., `[JUDGE]` without confirmed tier) -> `judge_tier = None` (never guess a tier).
   - **Precedence (`tier_0 > tier_2 > tier_1`):** when multiple judges reply in a thread and a higher-tier judge or consensus refines/corrects an earlier quick reply, consolidate the final authoritative conclusion rather than storing contradictory partial replies.
   - **Mechanical vs. tournament policy rulings:** preserve exact OPTCG terminology, phase names, and bracketed keywords (`[On Play]`, `[Blocker]`, `[Once Per Turn]`, `[DON!! x1]`). When a ruling depends on judge investigation or tournament policy (e.g., Scenario A vs. Scenario B for an illegal action), state the conditions explicitly—never flatten contextual policy guidance into an unconditional mechanical rule.

3. **Strict anonymization (privacy invariant)**
   - Strip ALL real names, nicknames, Discord usernames, `@handles`, `username#0000` discriminators, raw mentions (`<@...>`), phone numbers, and user IDs from `question`, `answer`, and `raw_ref`.
   - Record only the ruling content, `source = "judge"` (`authority = 3`), optional `judge_tier`, and anonymized `raw_ref`.

4. **Disambiguate and verify canonical entity links**
   - **Disambiguate homonymous cards:** always call `get_card` (or `search_cards`) and match the effect text to the exact printing before linking `card_ids` (e.g., `Captain John` has both `OP07-082` and `OP17-044`—verify which effect the thread discusses).
   - Verify every referenced rule with `get_rule` before linking canonical `rule_refs` (e.g., `7-1-1-1`, `7-1-4-1`).
   - Call `get_card_rulings` on linked cards to check whether an existing ruling is duplicated or superseded by the new thread.
   - Record the ruling's ISO `date` (`YYYY-MM-DD`); `add_ruling` records the active `meta.rules_version` automatically when omitted.

5. **Persist, review, and handle superseding (never delete)**
   - Call `add_ruling(question=..., answer=..., source="judge", date="YYYY-MM-DD", card_ids=[...], rule_refs=[...], judge_tier=..., raw_ref=...)` so every newly curated ruling enters with `reviewed = False` and is indexed immediately in FTS5.
   - During review (`list_unreviewed_rulings`), verify accuracy, anonymization, tier, and links before calling `mark_ruling(ruling_id, reviewed=True)`.
   - If a newer Comprehensive Rule, official FAQ/errata, or higher-tier judge ruling overrides an older ruling, **never delete** the old ruling—call `mark_ruling(old_id, superseded_by=new_id)`.

6. **Validate & finish**
   - Confirm zero personal names/handles appear in `question`, `answer`, or `raw_ref`.
   - Remind the user to run `uv run optcg-ingest embed` to index vectors for newly added rulings, and confirm `export_rulings` emits deterministic JSON once approved.
   - Run `uv run pytest` and `uv run ruff check` if curation code or prompt templates were modified.
