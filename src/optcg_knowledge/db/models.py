"""Domain data models for OPTCG rules, cards, rulings, and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AuthorityLevel = Literal[1, 2, 3, 4]
SourceType = Literal["rules", "official_faq", "errata", "judge", "community"]
DocType = Literal["rule", "ruling", "card"]
JudgeTier = Literal["tier_0", "tier_2", "tier_1"]

AUTHORITY_BY_SOURCE: dict[SourceType, AuthorityLevel] = {
    "rules": 1,
    "official_faq": 2,
    "errata": 2,
    "judge": 3,
    "community": 4,
}

# Precedence within authority = 3 (source = "judge"): Tier 0 > Tier 2 > Tier 1.
JUDGE_TIER_RANK: dict[JudgeTier, int] = {
    "tier_0": 1,
    "tier_2": 2,
    "tier_1": 3,
}

_JUDGE_TIER_ALIASES: dict[str, JudgeTier] = {
    "tier_0": "tier_0",
    "tier0": "tier_0",
    "tier 0": "tier_0",
    "t0": "tier_0",
    "0": "tier_0",
    "tier_2": "tier_2",
    "tier2": "tier_2",
    "tier 2": "tier_2",
    "t2": "tier_2",
    "2": "tier_2",
    "tier_1": "tier_1",
    "tier1": "tier_1",
    "tier 1": "tier_1",
    "t1": "tier_1",
    "1": "tier_1",
}


def authority_for_source(source: str) -> AuthorityLevel:
    """Return the canonical authority level (1..4) for a ruling/document source."""
    if source not in AUTHORITY_BY_SOURCE:
        valid = ", ".join(sorted(AUTHORITY_BY_SOURCE))
        raise ValueError(f"Unknown source {source!r}; expected one of: {valid}")
    return AUTHORITY_BY_SOURCE[source]  # type: ignore[index]


def normalize_judge_tier(tier: str | None) -> JudgeTier | None:
    """Normalize a judge tier label to canonical ``'tier_0' | 'tier_2' | 'tier_1'``."""
    if tier is None:
        return None
    cleaned = " ".join(tier.strip().lower().split())
    if not cleaned:
        return None
    normalized = _JUDGE_TIER_ALIASES.get(cleaned)
    if normalized is None:
        raise ValueError(
            f"Unknown judge_tier {tier!r}; expected one of: 'tier_0', 'tier_2', 'tier_1'"
        )
    return normalized


def judge_tier_rank(tier: str | None) -> int:
    """Return numeric sort rank for judge tier (1=tier_0, 2=tier_2, 3=tier_1, 4=unspecified)."""
    if tier is None:
        return 4
    return JUDGE_TIER_RANK.get(tier, 4)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RuleRecord:
    """A single Comprehensive Rules entry."""

    rule_id: str
    parent_id: str | None
    section: str
    text: str
    rules_version: str | None = None
    effective_date: str | None = None


@dataclass(frozen=True, slots=True)
class CardRecord:
    """Canonical mechanical representation of a One Piece TCG card."""

    card_id: str
    name: str
    set_code: str
    category: str
    colors: tuple[str, ...] = ()
    cost: int | None = None
    life: int | None = None
    power: int | None = None
    counter: int | None = None
    attributes: tuple[str, ...] = ()
    traits: tuple[str, ...] = ()
    effect_text: str | None = None
    trigger_text: str | None = None
    rarity: str | None = None
    image_url: str | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RulingRecord:
    """Structured OPTCG ruling (official FAQ, errata, judge, or community)."""

    ruling_id: str
    question: str
    answer: str
    source: SourceType
    authority: AuthorityLevel
    date: str | None = None
    rules_version: str | None = None
    url: str | None = None
    superseded_by: str | None = None
    reviewed: bool = False
    raw_ref: str | None = None
    judge_tier: JudgeTier | None = None
    card_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Searchable unit indexed by FTS5 (BM25) and sqlite-vec (embeddings)."""

    chunk_id: str
    doc_type: DocType
    doc_id: str
    text: str
    authority: AuthorityLevel
    metadata: dict[str, Any]
    content_hash: str
