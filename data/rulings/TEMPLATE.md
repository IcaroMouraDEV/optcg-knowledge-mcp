# Discord Judge Forum Export Template (`data/raw/judge_chats/`)

Copy this file (e.g. `2026-11-judge-forum.md`) and paste one or more Discord forum threads separated by `---`.

## Role & Judge Tier Tags

Within `authority = 3` (`source = "judge"`), judge tiers follow the official precedence:

- `[JUDGE T0]` — **Tier 0** (highest judge authority)
- `[JUDGE T2]` — **Tier 2** (highest normally attainable judge tier)
- `[JUDGE T1]` — **Tier 1** (basic certified judge)
- `[JUDGE]` — Judge with unspecified tier
- `[PLAYER]` — Player asking the question (or follow-up)

> **Privacy note:** Never keep personal names, Discord handles, or user IDs in curated rulings. If raw Discord copy/paste includes usernames, the `curar` prompt will strip them automatically and store only the ruling content, role `"judge"`, and optional `judge_tier`.

---

Thread: <Forum post title (e.g., OP05-119 extra turn vs OP04-083 Sabo)>
Date: YYYY-MM-DD
Cards: OP05-119, OP04-083  <!-- optional hint -->
Rules: 7-1-4-1             <!-- optional hint -->

[PLAYER]
<Paste the initial question from the forum post here.>

[JUDGE T2]
<Paste the judge's answer here.>

[JUDGE T0]
<Optional: if a higher-tier judge clarified or overruled an earlier reply in the same thread, paste it here so curation uses the highest-tier ruling.>

---

Thread: <Second forum post title>
Date: YYYY-MM-DD

[PLAYER]
<Question>

[JUDGE T1]
<Answer>

---
