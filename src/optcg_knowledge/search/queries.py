"""Structured SQLite query functions backing the OPTCG MCP query tools."""

from __future__ import annotations

import json
import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any

from optcg_knowledge.db.schema import get_meta

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_CARD_LIKE_RE = re.compile(r"^(?:(?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})$", re.IGNORECASE)

MAX_SEARCH_CARDS_LIMIT = 100
MAX_FUZZY_CARD_RESULTS = 50


def _rule_sort_key(rule_id: str) -> tuple[tuple[int, int | str], ...]:
    """Natural numeric sort key so '7-1-4-2' sorts before '7-1-4-10'."""
    parts: list[tuple[int, int | str]] = []
    for segment in rule_id.split("-"):
        if segment.isdigit():
            parts.append((0, int(segment)))
        else:
            parts.append((1, segment.lower()))
    return tuple(parts)


def _normalize_text(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", value.lower()).strip()


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _extract_variant_count(raw_json_str: str | None) -> int:
    if not raw_json_str:
        return 1
    try:
        parsed = json.loads(raw_json_str)
    except json.JSONDecodeError:
        return 1
    if isinstance(parsed, dict):
        variants = parsed.get("variants")
        if isinstance(variants, list):
            return 1 + len(variants)
    return 1


def _serialize_rule_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "rule_id": str(row["rule_id"]),
        "parent_id": str(row["parent_id"]) if row["parent_id"] is not None else None,
        "section": str(row["section"]),
        "text": str(row["text"]),
        "rules_version": str(row["rules_version"]) if row["rules_version"] is not None else None,
        "effective_date": (
            str(row["effective_date"]) if row["effective_date"] is not None else None
        ),
    }


def _serialize_card_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "card_id": str(row["card_id"]),
        "name": str(row["name"]),
        "set_code": str(row["set_code"]),
        "category": str(row["category"]),
        "colors": _parse_json_list(row["colors"]),
        "cost": int(row["cost"]) if row["cost"] is not None else None,
        "life": int(row["life"]) if row["life"] is not None else None,
        "power": int(row["power"]) if row["power"] is not None else None,
        "counter": int(row["counter"]) if row["counter"] is not None else None,
        "attributes": _parse_json_list(row["attributes"]),
        "traits": _parse_json_list(row["traits"]),
        "effect_text": str(row["effect_text"]) if row["effect_text"] is not None else None,
        "trigger_text": str(row["trigger_text"]) if row["trigger_text"] is not None else None,
        "rarity": str(row["rarity"]) if row["rarity"] is not None else None,
        "image_url": str(row["image_url"]) if row["image_url"] is not None else None,
        "variant_count": _extract_variant_count(row["raw_json"]),
    }


def get_rule(
    conn: sqlite3.Connection,
    rule_id: str,
    *,
    include_children: bool = True,
) -> dict[str, Any]:
    """Fetch a Comprehensive Rule by ID, including ancestor chain and optional descendants."""
    normalized_id = rule_id.strip().rstrip(".")
    current_rules_version = get_meta(conn, "rules_version")

    if not normalized_id:
        return {
            "found": False,
            "rule_id": normalized_id,
            "rules_version": current_rules_version,
            "rule": None,
            "ancestors": [],
            "children": [],
        }

    row = conn.execute(
        """
        SELECT rule_id, parent_id, section, text, rules_version, effective_date
        FROM rules
        WHERE rule_id = ?
        """,
        (normalized_id,),
    ).fetchone()

    if row is None:
        return {
            "found": False,
            "rule_id": normalized_id,
            "rules_version": current_rules_version,
            "rule": None,
            "ancestors": [],
            "children": [],
        }

    ancestors: list[dict[str, Any]] = []
    seen_ids: set[str] = {normalized_id}
    cursor = row["parent_id"]
    while cursor is not None and str(cursor) not in seen_ids:
        seen_ids.add(str(cursor))
        parent_row = conn.execute(
            """
            SELECT rule_id, parent_id, section, text, rules_version, effective_date
            FROM rules
            WHERE rule_id = ?
            """,
            (str(cursor),),
        ).fetchone()
        if parent_row is None:
            break
        ancestors.append(_serialize_rule_row(parent_row))
        cursor = parent_row["parent_id"]
    ancestors.reverse()

    children: list[dict[str, Any]] = []
    if include_children:
        child_rows = conn.execute(
            """
            SELECT rule_id, parent_id, section, text, rules_version, effective_date
            FROM rules
            WHERE rule_id LIKE ? OR parent_id = ?
            """,
            (f"{normalized_id}-%", normalized_id),
        ).fetchall()
        deduped = {str(r["rule_id"]): r for r in child_rows if str(r["rule_id"]) != normalized_id}
        ordered = sorted(deduped.values(), key=lambda r: _rule_sort_key(str(r["rule_id"])))
        children = [_serialize_rule_row(r) for r in ordered]

    return {
        "found": True,
        "rule_id": normalized_id,
        "rules_version": current_rules_version,
        "rule": _serialize_rule_row(row),
        "ancestors": ancestors,
        "children": children,
    }


def _fuzzy_name_score(norm_query: str, query_tokens: list[str], card_name: str) -> float:
    norm_name = _normalize_text(card_name)
    if not norm_name:
        return 0.0
    if norm_query == norm_name:
        return 1.0
    if norm_query in norm_name:
        coverage = len(norm_query) / max(len(norm_name), 1)
        return 0.90 + 0.09 * coverage
    if query_tokens and all(tok in norm_name for tok in query_tokens):
        return 0.85

    full_ratio = SequenceMatcher(None, norm_query, norm_name).ratio()
    name_tokens = norm_name.split()
    token_ratio = 0.0
    if len(query_tokens) == 1 and name_tokens:
        token_ratio = max(SequenceMatcher(None, norm_query, tok).ratio() for tok in name_tokens)

    best = max(full_ratio, token_ratio)
    return best if best >= 0.65 else 0.0


def get_card(conn: sqlite3.Connection, id_or_name: str) -> dict[str, Any]:
    """Look up a card by exact canonical ID or fuzzy name, returning all matching versions."""
    cleaned = id_or_name.strip()
    if not cleaned:
        return {
            "query": id_or_name,
            "match_type": "not_found",
            "total": 0,
            "cards": [],
        }

    # 1. Exact canonical card_id match (case-insensitive)
    exact_id_row = conn.execute(
        "SELECT * FROM cards WHERE UPPER(card_id) = UPPER(?)",
        (cleaned,),
    ).fetchone()
    if exact_id_row is not None:
        return {
            "query": cleaned,
            "match_type": "exact_id",
            "total": 1,
            "cards": [_serialize_card_row(exact_id_row)],
        }

    # If the user entered a valid-looking card ID that is not in the DB, avoid noisy fuzzy hits.
    if _CARD_LIKE_RE.match(cleaned):
        return {
            "query": cleaned,
            "match_type": "not_found",
            "total": 0,
            "cards": [],
        }

    # 2. Exact name match (case-insensitive or punctuation-normalized, e.g. Monkey D Luffy)
    norm_query = _normalize_text(cleaned)
    all_rows = conn.execute("SELECT * FROM cards ORDER BY card_id ASC").fetchall()

    exact_name_rows = [
        row
        for row in all_rows
        if str(row["name"]).lower() == cleaned.lower()
        or (norm_query and _normalize_text(str(row["name"])) == norm_query)
    ]
    if exact_name_rows:
        return {
            "query": cleaned,
            "match_type": "exact_name",
            "total": len(exact_name_rows),
            "cards": [_serialize_card_row(row) for row in exact_name_rows],
        }

    # 3. Fuzzy name match
    query_tokens = norm_query.split()
    scored_rows: list[tuple[float, str, sqlite3.Row]] = []
    for row in all_rows:
        score = _fuzzy_name_score(norm_query, query_tokens, str(row["name"]))
        if score > 0.0:
            scored_rows.append((score, str(row["card_id"]), row))

    if not scored_rows:
        return {
            "query": cleaned,
            "match_type": "not_found",
            "total": 0,
            "cards": [],
        }

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    top_rows = [row for _, _, row in scored_rows[:MAX_FUZZY_CARD_RESULTS]]
    return {
        "query": cleaned,
        "match_type": "fuzzy_name",
        "total": len(top_rows),
        "cards": [_serialize_card_row(row) for row in top_rows],
    }


def _matches_keyword(effect_text: str | None, trigger_text: str | None, keyword: str) -> bool:
    haystack = f"{effect_text or ''}\n{trigger_text or ''}".lower()
    needle = keyword.strip().lower()
    if not needle:
        return True
    if needle in haystack:
        return True
    stripped = needle.strip("[]")
    return bool(stripped and f"[{stripped}]" in haystack)


def search_cards(
    conn: sqlite3.Connection,
    *,
    name: str | None = None,
    color: str | None = None,
    cost: int | None = None,
    trait: str | None = None,
    keyword: str | None = None,
    set_code: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Filter canonical OPTCG cards by structured attributes and text keywords."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    effective_limit = min(limit, MAX_SEARCH_CARDS_LIMIT)

    clauses: list[str] = []
    params: list[Any] = []

    if set_code is not None and set_code.strip():
        clauses.append("UPPER(set_code) = UPPER(?)")
        params.append(set_code.strip())
    if cost is not None:
        clauses.append("cost = ?")
        params.append(cost)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM cards {where_sql} ORDER BY card_id ASC",
        params,
    ).fetchall()

    norm_name_filter = _normalize_text(name) if name and name.strip() else None
    raw_name_filter = name.strip().lower() if name and name.strip() else None
    color_filter = color.strip().lower() if color and color.strip() else None
    trait_filter = trait.strip().lower() if trait and trait.strip() else None
    keyword_filter = keyword.strip() if keyword and keyword.strip() else None

    matched: list[dict[str, Any]] = []
    for row in rows:
        if raw_name_filter is not None:
            card_name = str(row["name"])
            if raw_name_filter not in card_name.lower() and (
                not norm_name_filter or norm_name_filter not in _normalize_text(card_name)
            ):
                continue

        if color_filter is not None:
            colors = [c.lower() for c in _parse_json_list(row["colors"])]
            if color_filter not in colors:
                continue

        if trait_filter is not None:
            traits = _parse_json_list(row["traits"])
            if not any(trait_filter in t.lower() for t in traits):
                continue

        if keyword_filter is not None and not _matches_keyword(
            row["effect_text"], row["trigger_text"], keyword_filter
        ):
            continue

        matched.append(_serialize_card_row(row))

    returned_cards = matched[:effective_limit]
    return {
        "filters": {
            "name": name,
            "color": color,
            "cost": cost,
            "trait": trait,
            "keyword": keyword,
            "set_code": set_code,
        },
        "total_matched": len(matched),
        "returned": len(returned_cards),
        "cards": returned_cards,
    }


def get_card_rulings(
    conn: sqlite3.Connection,
    card_id: str,
    *,
    include_superseded: bool = False,
) -> dict[str, Any]:
    """Return rulings linked to a canonical card_id, ordered by authority ASC, date DESC."""
    normalized_id = card_id.strip().upper()
    current_rules_version = get_meta(conn, "rules_version")

    card_row = conn.execute(
        "SELECT 1 FROM cards WHERE UPPER(card_id) = ?",
        (normalized_id,),
    ).fetchone()
    card_found = card_row is not None

    superseded_clause = "" if include_superseded else "AND r.superseded_by IS NULL"
    rows = conn.execute(
        f"""
        SELECT
            r.ruling_id,
            r.question,
            r.answer,
            r.source,
            r.authority,
            r.judge_tier,
            r.date,
            r.rules_version,
            r.url,
            r.superseded_by,
            r.reviewed
        FROM rulings r
        JOIN ruling_cards rc ON rc.ruling_id = r.ruling_id
        WHERE UPPER(rc.card_id) = ?
          {superseded_clause}
        ORDER BY
            r.authority ASC,
            CASE r.judge_tier
                WHEN 'tier_0' THEN 1
                WHEN 'tier_2' THEN 2
                WHEN 'tier_1' THEN 3
                ELSE 4
            END ASC,
            COALESCE(r.date, '') DESC,
            r.ruling_id ASC
        """,
        (normalized_id,),
    ).fetchall()

    ruling_ids = [str(row["ruling_id"]) for row in rows]
    cards_by_ruling: dict[str, list[str]] = {rid: [] for rid in ruling_ids}
    rules_by_ruling: dict[str, list[str]] = {rid: [] for rid in ruling_ids}

    if ruling_ids:
        placeholders = ", ".join("?" for _ in ruling_ids)
        for link in conn.execute(
            f"""
            SELECT ruling_id, card_id
            FROM ruling_cards
            WHERE ruling_id IN ({placeholders})
            ORDER BY card_id ASC
            """,
            ruling_ids,
        ).fetchall():
            cards_by_ruling[str(link["ruling_id"])].append(str(link["card_id"]))

        for link in conn.execute(
            f"""
            SELECT ruling_id, rule_id
            FROM ruling_rules
            WHERE ruling_id IN ({placeholders})
            """,
            ruling_ids,
        ).fetchall():
            rules_by_ruling[str(link["ruling_id"])].append(str(link["rule_id"]))

        for rid in rules_by_ruling:
            rules_by_ruling[rid].sort(key=_rule_sort_key)

    rulings: list[dict[str, Any]] = []
    for row in rows:
        rid = str(row["ruling_id"])
        ruling_version = str(row["rules_version"]) if row["rules_version"] is not None else None
        potentially_outdated = bool(
            ruling_version and current_rules_version and ruling_version != current_rules_version
        )
        rulings.append(
            {
                "ruling_id": rid,
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "source": str(row["source"]),
                "authority": int(row["authority"]),
                "judge_tier": (
                    str(row["judge_tier"]) if row["judge_tier"] is not None else None
                ),
                "date": str(row["date"]) if row["date"] is not None else None,
                "rules_version": ruling_version,
                "url": str(row["url"]) if row["url"] is not None else None,
                "superseded_by": (
                    str(row["superseded_by"]) if row["superseded_by"] is not None else None
                ),
                "reviewed": bool(row["reviewed"]),
                "potentially_outdated": potentially_outdated,
                "card_ids": cards_by_ruling[rid],
                "rule_ids": rules_by_ruling[rid],
            }
        )

    return {
        "card_id": normalized_id,
        "card_found": card_found,
        "current_rules_version": current_rules_version,
        "include_superseded": include_superseded,
        "total": len(rulings),
        "rulings": rulings,
    }
