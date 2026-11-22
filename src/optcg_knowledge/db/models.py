"""Domain data models for OPTCG rules, cards, rulings, and chunks."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

AuthorityLevel = Literal[1, 2, 3, 4]
SourceType = Literal["rules", "official_faq", "errata", "judge", "community"]
DocType = Literal["rule", "ruling", "card", "tournament_rule"]
JudgeTier = Literal["tier_0", "tier_2", "tier_1"]
BanStatus = Literal["legal", "banned", "restricted", "pair_banned"]
ActiveBanStatus = Literal["banned", "restricted", "pair_banned"]
RestrictionScope = Literal["active", "upcoming"]

VALID_BAN_STATUSES: tuple[BanStatus, ...] = (
    "legal",
    "banned",
    "restricted",
    "pair_banned",
)

MAX_COPIES_BY_BAN_STATUS: dict[BanStatus, int] = {
    "legal": 4,
    "banned": 0,
    "restricted": 1,
    "pair_banned": 4,
}

# Comprehensive Rules 5-1-2: 1 Leader card + 50-card main deck.
LEADER_MAX_COPIES = 1
MAX_MAIN_DECK_COPIES = 50

# Comprehensive Rules 5-1-2-4 canonical unlimited-copy deck construction clause.
_UNLIMITED_DECK_COPIES_RE = re.compile(
    r"\bunder\s+the\s+rules\s+of\s+this\s+game\s*,\s*"
    r"you\s+may\s+have\s+any\s+number\s+of\s+this\s+card\s+in\s+your\s+deck\b",
    re.IGNORECASE,
)

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


_BAN_STATUS_ALIASES: dict[str, BanStatus] = {
    "legal": "legal",
    "unrestricted": "legal",
    "banned": "banned",
    "ban": "banned",
    "restricted": "restricted",
    "limited": "restricted",
    "pair_banned": "pair_banned",
    "pair-banned": "pair_banned",
    "pair banned": "pair_banned",
    "banned_pair": "pair_banned",
    "banned-pair": "pair_banned",
    "banned pair": "pair_banned",
}


def normalize_ban_status(status: str) -> BanStatus:
    """Normalize a banlist status string to canonical BanStatus."""
    cleaned = " ".join(status.strip().lower().split())
    normalized = _BAN_STATUS_ALIASES.get(cleaned)
    if normalized is None:
        valid = ", ".join(VALID_BAN_STATUSES)
        raise ValueError(f"Unknown ban_status {status!r}; expected one of: {valid}")
    return normalized


def has_unlimited_copies_rule(effect_text: str | None) -> bool:
    """Return True when card text carries the canonical CR 5-1-2-4 unlimited-copy clause."""
    if not effect_text:
        return False
    return _UNLIMITED_DECK_COPIES_RE.search(effect_text) is not None


def resolve_card_max_copies(
    ban_status: str,
    *,
    category: str | None = None,
    effect_text: str | None = None,
    restricted_max_copies: int | None = None,
) -> int:
    """Compute effective max_copies combining banlist status and CR 5-1-2 deck rules.

    Precedence:
    1. ``banned`` -> ``0`` and ``restricted`` -> ``1`` (or explicit cap) override card text.
    2. ``Leader`` cards -> ``1`` (CR 5-1-2) when ``legal`` or ``pair_banned``.
    3. Cards with the canonical unlimited-copy rule (CR 5-1-2-4) -> ``50`` when ``legal``
       or ``pair_banned``.
    4. Otherwise ``restricted_max_copies`` (when provided) or ``4``.
    """
    normalized_status = normalize_ban_status(ban_status)
    if normalized_status == "banned":
        return 0
    if normalized_status == "restricted":
        if restricted_max_copies is not None:
            return restricted_max_copies
        return MAX_COPIES_BY_BAN_STATUS["restricted"]
    if category is not None and category.strip().lower() == "leader":
        return LEADER_MAX_COPIES
    if has_unlimited_copies_rule(effect_text):
        return MAX_MAIN_DECK_COPIES
    if restricted_max_copies is not None:
        return restricted_max_copies
    return MAX_COPIES_BY_BAN_STATUS[normalized_status]


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
class TrmSectionRecord:
    """A single official Tournament Rules Manual (TRM) section or infraction entry."""

    section_id: str
    parent_id: str | None
    section: str
    title: str
    text: str
    trm_version: str | None = None
    effective_date: str | None = None


def normalize_block_icon(value: Any) -> int | None:
    """Normalize an OPTCG Block Icon value to a positive integer (>= 1) or None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned == "-":
            return None
        try:
            parsed = int(cleaned)
        except ValueError:
            return None
        return parsed if parsed >= 1 else None
    return None


def extract_block_icon_from_raw(raw: Mapping[str, Any]) -> int | None:
    """Extract the highest known Block Icon across a card's base printing and variants.

    Supports `punk-records` (`block_number`) and local JSON aliases (`block_icon`, `block`).
    Under official Bandai Standard rotation rules, a reprint with a newer Block Icon keeps
    all printings of that canonical card_id legal, so the maximum valid block is returned.
    """
    candidates: list[int] = []

    for key in ("block_icon", "block_number", "block"):
        parsed = normalize_block_icon(raw.get(key))
        if parsed is not None:
            candidates.append(parsed)
            break

    variants = raw.get("variants")
    if isinstance(variants, list):
        for variant in variants:
            if not isinstance(variant, Mapping):
                continue
            for key in ("block_icon", "block_number", "block"):
                parsed = normalize_block_icon(variant.get(key))
                if parsed is not None:
                    candidates.append(parsed)
                    break

    return max(candidates) if candidates else None


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
    block_icon: int | None = None
    image_url: str | None = None
    raw_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CardRestrictionRecord:
    """Official banlist / restriction status for a canonical OPTCG card."""

    card_id: str
    ban_status: ActiveBanStatus
    max_copies: int
    paired_card_ids: tuple[str, ...] = ()
    restriction_note: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    scope: RestrictionScope = "active"


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
