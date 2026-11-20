"""Structured SQLite query functions backing the OPTCG MCP query tools."""

from __future__ import annotations

import json
import re
import sqlite3
from difflib import SequenceMatcher
from typing import Any

from optcg_knowledge.db.models import MAX_COPIES_BY_BAN_STATUS, normalize_ban_status
from optcg_knowledge.db.schema import get_meta
from optcg_knowledge.ingest.trm_parser import TRM_ID_PREFIX, normalize_trm_section_id

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


def _trm_sort_key(section_id: str) -> tuple[tuple[int, int | str], ...]:
    """Natural numeric sort key so 'TRM-7.4.2' sorts before 'TRM-7.4.10'."""
    raw = (
        section_id[len(TRM_ID_PREFIX) :]
        if section_id.upper().startswith(TRM_ID_PREFIX)
        else section_id
    )
    parts: list[tuple[int, int | str]] = []
    for segment in raw.split("."):
        if segment.isdigit():
            parts.append((0, int(segment)))
        else:
            parts.append((1, segment.lower()))
    return tuple(parts)


def _serialize_trm_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "section_id": str(row["section_id"]),
        "parent_id": str(row["parent_id"]) if row["parent_id"] is not None else None,
        "section": str(row["section"]),
        "title": str(row["title"]),
        "text": str(row["text"]),
        "trm_version": str(row["trm_version"]) if row["trm_version"] is not None else None,
        "effective_date": (
            str(row["effective_date"]) if row["effective_date"] is not None else None
        ),
    }


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


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _has_card_restrictions_table(conn: sqlite3.Connection) -> bool:
    return _has_table(conn, "card_restrictions")


def _load_restrictions_by_card_id(
    conn: sqlite3.Connection,
    card_ids: list[str] | None = None,
) -> dict[str, sqlite3.Row]:
    if not _has_card_restrictions_table(conn):
        return {}
    if card_ids is not None:
        if not card_ids:
            return {}
        placeholders = ", ".join("?" for _ in card_ids)
        rows = conn.execute(
            f"""
            SELECT card_id, ban_status, max_copies, paired_card_ids,
                   restriction_note, effective_date, source_url
            FROM card_restrictions
            WHERE UPPER(card_id) IN ({placeholders})
            """,
            [cid.upper() for cid in card_ids],
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT card_id, ban_status, max_copies, paired_card_ids,
                   restriction_note, effective_date, source_url
            FROM card_restrictions
            """
        ).fetchall()
    return {str(row["card_id"]).upper(): row for row in rows}


def _banlist_meta(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "banlist_loaded": get_meta(conn, "banlist_last_ingested_at") is not None,
        "banlist_effective_date": get_meta(conn, "banlist_effective_date"),
    }


def _serialize_card_row(
    row: sqlite3.Row,
    restriction: sqlite3.Row | None = None,
) -> dict[str, Any]:
    if restriction is not None:
        ban_status = str(restriction["ban_status"])
        max_copies = int(restriction["max_copies"])
        paired_card_ids = _parse_json_list(restriction["paired_card_ids"])
        restriction_note = (
            str(restriction["restriction_note"])
            if restriction["restriction_note"] is not None
            else None
        )
        effective_date = (
            str(restriction["effective_date"])
            if restriction["effective_date"] is not None
            else None
        )
    else:
        ban_status = "legal"
        max_copies = MAX_COPIES_BY_BAN_STATUS["legal"]
        paired_card_ids = []
        restriction_note = None
        effective_date = None

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
        "ban_status": ban_status,
        "max_copies": max_copies,
        "paired_card_ids": paired_card_ids,
        "restriction_note": restriction_note,
        "effective_date": effective_date,
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


def get_trm_section(
    conn: sqlite3.Connection,
    section_id: str,
    *,
    include_children: bool = True,
) -> dict[str, Any]:
    """Fetch a Tournament Rules Manual (TRM) section, ancestors, and optional descendants."""
    normalized_id = normalize_trm_section_id(section_id)
    current_trm_version = get_meta(conn, "trm_version")
    current_trm_effective_date = get_meta(conn, "trm_effective_date")

    if not normalized_id:
        return {
            "found": False,
            "section_id": normalized_id,
            "source": "tournament_rules",
            "trm_version": current_trm_version,
            "trm_effective_date": current_trm_effective_date,
            "section": None,
            "ancestors": [],
            "children": [],
        }

    row = conn.execute(
        """
        SELECT section_id, parent_id, section, title, text, trm_version, effective_date
        FROM trm_sections
        WHERE UPPER(section_id) = UPPER(?)
        """,
        (normalized_id,),
    ).fetchone()

    if row is None:
        return {
            "found": False,
            "section_id": normalized_id,
            "source": "tournament_rules",
            "trm_version": current_trm_version,
            "trm_effective_date": current_trm_effective_date,
            "section": None,
            "ancestors": [],
            "children": [],
        }

    canonical_id = str(row["section_id"])
    ancestors: list[dict[str, Any]] = []
    seen_ids: set[str] = {canonical_id}
    cursor = row["parent_id"]
    while cursor is not None and str(cursor) not in seen_ids:
        seen_ids.add(str(cursor))
        parent_row = conn.execute(
            """
            SELECT section_id, parent_id, section, title, text, trm_version, effective_date
            FROM trm_sections
            WHERE section_id = ?
            """,
            (str(cursor),),
        ).fetchone()
        if parent_row is None:
            break
        ancestors.append(_serialize_trm_row(parent_row))
        cursor = parent_row["parent_id"]
    ancestors.reverse()

    children: list[dict[str, Any]] = []
    if include_children:
        raw_num = (
            canonical_id[len(TRM_ID_PREFIX) :]
            if canonical_id.upper().startswith(TRM_ID_PREFIX)
            else canonical_id
        )
        parts = raw_num.split(".")
        if len(parts) == 2 and parts[1] == "0":
            prefix_pattern = f"{TRM_ID_PREFIX}{parts[0]}.%"
        else:
            prefix_pattern = f"{canonical_id}.%"

        child_rows = conn.execute(
            """
            SELECT section_id, parent_id, section, title, text, trm_version, effective_date
            FROM trm_sections
            WHERE UPPER(section_id) LIKE UPPER(?) OR parent_id = ?
            """,
            (prefix_pattern, canonical_id),
        ).fetchall()
        deduped = {
            str(r["section_id"]): r
            for r in child_rows
            if str(r["section_id"]) != canonical_id
        }
        ordered = sorted(
            deduped.values(),
            key=lambda r: _trm_sort_key(str(r["section_id"])),
        )
        children = [_serialize_trm_row(r) for r in ordered]

    return {
        "found": True,
        "section_id": canonical_id,
        "source": "tournament_rules",
        "trm_version": current_trm_version,
        "trm_effective_date": current_trm_effective_date,
        "section": _serialize_trm_row(row),
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
    meta = _banlist_meta(conn)
    if not cleaned:
        return {
            "query": id_or_name,
            "match_type": "not_found",
            "total": 0,
            **meta,
            "cards": [],
        }

    # 1. Exact canonical card_id match (case-insensitive)
    exact_id_row = conn.execute(
        "SELECT * FROM cards WHERE UPPER(card_id) = UPPER(?)",
        (cleaned,),
    ).fetchone()
    if exact_id_row is not None:
        cid = str(exact_id_row["card_id"]).upper()
        restrictions = _load_restrictions_by_card_id(conn, [cid])
        return {
            "query": cleaned,
            "match_type": "exact_id",
            "total": 1,
            **meta,
            "cards": [_serialize_card_row(exact_id_row, restrictions.get(cid))],
        }

    # If the user entered a valid-looking card ID that is not in the DB, avoid noisy fuzzy hits.
    if _CARD_LIKE_RE.match(cleaned):
        return {
            "query": cleaned,
            "match_type": "not_found",
            "total": 0,
            **meta,
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
        matched_ids = [str(row["card_id"]).upper() for row in exact_name_rows]
        restrictions = _load_restrictions_by_card_id(conn, matched_ids)
        return {
            "query": cleaned,
            "match_type": "exact_name",
            "total": len(exact_name_rows),
            **meta,
            "cards": [
                _serialize_card_row(row, restrictions.get(str(row["card_id"]).upper()))
                for row in exact_name_rows
            ],
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
            **meta,
            "cards": [],
        }

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    top_rows = [row for _, _, row in scored_rows[:MAX_FUZZY_CARD_RESULTS]]
    matched_ids = [str(row["card_id"]).upper() for row in top_rows]
    restrictions = _load_restrictions_by_card_id(conn, matched_ids)
    return {
        "query": cleaned,
        "match_type": "fuzzy_name",
        "total": len(top_rows),
        **meta,
        "cards": [
            _serialize_card_row(row, restrictions.get(str(row["card_id"]).upper()))
            for row in top_rows
        ],
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
    ban_status: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Filter canonical OPTCG cards by structured attributes, text keywords, and ban_status."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    effective_limit = min(limit, MAX_SEARCH_CARDS_LIMIT)

    normalized_ban_status = (
        normalize_ban_status(ban_status)
        if ban_status is not None and ban_status.strip()
        else None
    )

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
    restrictions = _load_restrictions_by_card_id(conn)

    norm_name_filter = _normalize_text(name) if name and name.strip() else None
    raw_name_filter = name.strip().lower() if name and name.strip() else None
    color_filter = color.strip().lower() if color and color.strip() else None
    trait_filter = trait.strip().lower() if trait and trait.strip() else None
    keyword_filter = keyword.strip() if keyword and keyword.strip() else None

    matched: list[dict[str, Any]] = []
    for row in rows:
        card_id_upper = str(row["card_id"]).upper()
        restriction = restrictions.get(card_id_upper)
        effective_ban_status = (
            str(restriction["ban_status"]) if restriction is not None else "legal"
        )
        if normalized_ban_status is not None and effective_ban_status != normalized_ban_status:
            continue

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

        matched.append(_serialize_card_row(row, restriction))

    returned_cards = matched[:effective_limit]
    return {
        "filters": {
            "name": name,
            "color": color,
            "cost": cost,
            "trait": trait,
            "keyword": keyword,
            "set_code": set_code,
            "ban_status": normalized_ban_status,
        },
        **_banlist_meta(conn),
        "total_matched": len(matched),
        "returned": len(returned_cards),
        "cards": returned_cards,
    }


def get_banlist(
    conn: sqlite3.Connection,
    *,
    ban_status: str | None = None,
) -> dict[str, Any]:
    """Return all active official banlist/restriction entries in a single structured response."""
    normalized_filter: str | None = None
    if ban_status is not None and ban_status.strip():
        normalized_filter = normalize_ban_status(ban_status)
        if normalized_filter == "legal":
            raise ValueError(
                "get_banlist only returns restricted cards ('banned', 'restricted', or "
                "'pair_banned'); use search_cards(ban_status='legal') to list legal cards"
            )

    meta = _banlist_meta(conn)
    banlist_source = get_meta(conn, "banlist_source")

    if not _has_card_restrictions_table(conn):
        return {
            **meta,
            "banlist_source": banlist_source,
            "filter_ban_status": normalized_filter,
            "counts": {"banned": 0, "restricted": 0, "pair_banned": 0, "total": 0},
            "banned_pairs": [],
            "restrictions": [],
        }

    where_sql = ""
    params: list[Any] = []
    if normalized_filter is not None:
        where_sql = "WHERE cr.ban_status = ?"
        params.append(normalized_filter)

    rows = conn.execute(
        f"""
        SELECT
            cr.card_id,
            cr.ban_status,
            cr.max_copies,
            cr.paired_card_ids,
            cr.restriction_note,
            cr.effective_date,
            cr.source_url,
            c.name AS card_name,
            c.set_code,
            c.category,
            c.colors
        FROM card_restrictions cr
        LEFT JOIN cards c ON c.card_id = cr.card_id
        {where_sql}
        ORDER BY
            CASE cr.ban_status
                WHEN 'banned' THEN 1
                WHEN 'restricted' THEN 2
                WHEN 'pair_banned' THEN 3
                ELSE 4
            END ASC,
            cr.card_id ASC
        """,
        params,
    ).fetchall()

    restrictions: list[dict[str, Any]] = []
    counts = {"banned": 0, "restricted": 0, "pair_banned": 0}
    pair_set: set[tuple[str, str]] = set()

    for row in rows:
        cid = str(row["card_id"])
        status = str(row["ban_status"])
        if status in counts:
            counts[status] += 1
        paired = _parse_json_list(row["paired_card_ids"])
        for partner in paired:
            a, b = sorted((cid, partner))
            if a != b:
                pair_set.add((a, b))

        restrictions.append(
            {
                "card_id": cid,
                "card_found": row["card_name"] is not None,
                "name": str(row["card_name"]) if row["card_name"] is not None else None,
                "set_code": str(row["set_code"]) if row["set_code"] is not None else None,
                "category": str(row["category"]) if row["category"] is not None else None,
                "colors": _parse_json_list(row["colors"]),
                "ban_status": status,
                "max_copies": int(row["max_copies"]),
                "paired_card_ids": paired,
                "restriction_note": (
                    str(row["restriction_note"]) if row["restriction_note"] is not None else None
                ),
                "effective_date": (
                    str(row["effective_date"]) if row["effective_date"] is not None else None
                ),
                "source_url": str(row["source_url"]) if row["source_url"] is not None else None,
            }
        )

    banned_pairs = [[a, b] for a, b in sorted(pair_set)]
    return {
        **meta,
        "banlist_source": banlist_source,
        "filter_ban_status": normalized_filter,
        "counts": {**counts, "total": len(restrictions)},
        "banned_pairs": banned_pairs,
        "restrictions": restrictions,
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


def _count_table(conn: sqlite3.Connection, table_name: str) -> int:
    if not _has_table(conn, table_name):
        return 0
    row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    return int(row[0]) if row is not None else 0


def get_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return database health, versions, row counts, embedding coverage, and timestamps."""
    has_meta = _has_table(conn, "meta")

    def _meta(key: str) -> str | None:
        return get_meta(conn, key) if has_meta else None

    schema_version = _meta("schema_version")
    rules_version = _meta("rules_version")
    rules_effective_date = _meta("rules_effective_date")
    trm_version = _meta("trm_version")
    trm_effective_date = _meta("trm_effective_date")
    banlist_effective_date = _meta("banlist_effective_date")
    banlist_source = _meta("banlist_source")
    banlist_last_ingested_at = _meta("banlist_last_ingested_at")
    cards_source = _meta("cards_source")
    embedding_model = _meta("embedding_model")
    embedding_dim_raw = _meta("embedding_dim")
    embedding_dim: int | None = None
    if embedding_dim_raw is not None:
        try:
            embedding_dim = int(embedding_dim_raw)
        except ValueError:
            embedding_dim = None

    rules_count = _count_table(conn, "rules")
    trm_sections_count = _count_table(conn, "trm_sections")
    cards_count = _count_table(conn, "cards")

    restriction_counts = {"total": 0, "banned": 0, "restricted": 0, "pair_banned": 0}
    if _has_table(conn, "card_restrictions"):
        for row in conn.execute(
            "SELECT ban_status, COUNT(*) AS cnt FROM card_restrictions GROUP BY ban_status"
        ).fetchall():
            status_key = str(row["ban_status"])
            cnt = int(row["cnt"])
            restriction_counts["total"] += cnt
            if status_key in restriction_counts:
                restriction_counts[status_key] = cnt

    rulings_by_source = {
        "rules": 0,
        "official_faq": 0,
        "errata": 0,
        "judge": 0,
        "community": 0,
    }
    rulings_by_authority = {"1": 0, "2": 0, "3": 0, "4": 0}
    judge_by_tier = {"tier_0": 0, "tier_2": 0, "tier_1": 0, "unspecified": 0}
    rulings_total = 0
    rulings_active = 0
    rulings_superseded = 0
    rulings_unreviewed = 0

    if _has_table(conn, "rulings"):
        agg_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(CASE WHEN superseded_by IS NULL THEN 1 ELSE 0 END), 0) AS active,
                COALESCE(
                    SUM(CASE WHEN superseded_by IS NOT NULL THEN 1 ELSE 0 END), 0
                ) AS superseded,
                COALESCE(SUM(CASE WHEN reviewed = 0 THEN 1 ELSE 0 END), 0) AS unreviewed
            FROM rulings
            """
        ).fetchone()
        if agg_row is not None:
            rulings_total = int(agg_row["total"])
            rulings_active = int(agg_row["active"])
            rulings_superseded = int(agg_row["superseded"])
            rulings_unreviewed = int(agg_row["unreviewed"])

        for row in conn.execute(
            "SELECT source, COUNT(*) AS cnt FROM rulings GROUP BY source"
        ).fetchall():
            src_key = str(row["source"])
            if src_key in rulings_by_source:
                rulings_by_source[src_key] = int(row["cnt"])

        for row in conn.execute(
            "SELECT authority, COUNT(*) AS cnt FROM rulings GROUP BY authority"
        ).fetchall():
            auth_key = str(row["authority"])
            if auth_key in rulings_by_authority:
                rulings_by_authority[auth_key] = int(row["cnt"])

        ruling_columns = {
            str(r[1]) for r in conn.execute("PRAGMA table_info(rulings)").fetchall()
        }
        if "judge_tier" in ruling_columns:
            for row in conn.execute(
                """
                SELECT COALESCE(judge_tier, 'unspecified') AS tier, COUNT(*) AS cnt
                FROM rulings
                WHERE source = 'judge'
                GROUP BY COALESCE(judge_tier, 'unspecified')
                """
            ).fetchall():
                tier_key = str(row["tier"])
                if tier_key in judge_by_tier:
                    judge_by_tier[tier_key] = int(row["cnt"])
        else:
            judge_by_tier["unspecified"] = rulings_by_source["judge"]

    chunks_by_doc_type = {"rule": 0, "tournament_rule": 0, "card": 0, "ruling": 0}
    total_chunks = 0
    embedded_chunks = 0
    if _has_table(conn, "chunks"):
        for row in conn.execute(
            "SELECT doc_type, COUNT(*) AS cnt FROM chunks GROUP BY doc_type"
        ).fetchall():
            dtype = str(row["doc_type"])
            cnt = int(row["cnt"])
            total_chunks += cnt
            if dtype in chunks_by_doc_type:
                chunks_by_doc_type[dtype] = cnt

        if embedding_model is not None:
            emb_row = conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE embedded_hash IS NOT NULL
                  AND embedded_hash = content_hash
                  AND embedded_model = ?
                """,
                (embedding_model,),
            ).fetchone()
        else:
            emb_row = conn.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE embedded_hash IS NOT NULL
                  AND embedded_hash = content_hash
                  AND embedded_model IS NOT NULL
                """
            ).fetchone()
        embedded_chunks = int(emb_row[0]) if emb_row is not None else 0

    pending_chunks = max(total_chunks - embedded_chunks, 0)
    vec_rows = _count_table(conn, "chunks_vec")
    coverage_pct = round((embedded_chunks / total_chunks) * 100.0, 1) if total_chunks > 0 else 0.0
    embeddings_ready = total_chunks > 0 and pending_chunks == 0 and vec_rows >= embedded_chunks

    return {
        "schema_version": schema_version,
        "rules_version": rules_version,
        "rules_effective_date": rules_effective_date,
        "trm_version": trm_version,
        "trm_effective_date": trm_effective_date,
        "banlist_loaded": banlist_last_ingested_at is not None,
        "banlist_effective_date": banlist_effective_date,
        "banlist_source": banlist_source,
        "cards_source": cards_source,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "counts": {
            "rules": rules_count,
            "trm_sections": trm_sections_count,
            "cards": cards_count,
            "card_restrictions": restriction_counts,
            "rulings": {
                "total": rulings_total,
                "active": rulings_active,
                "superseded": rulings_superseded,
                "unreviewed": rulings_unreviewed,
                "by_source": rulings_by_source,
                "by_authority": rulings_by_authority,
                "judge_by_tier": judge_by_tier,
            },
            "chunks": {
                "total": total_chunks,
                "by_doc_type": chunks_by_doc_type,
            },
        },
        "embeddings": {
            "model": embedding_model,
            "dim": embedding_dim,
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "pending_chunks": pending_chunks,
            "vec_rows": vec_rows,
            "coverage_pct": coverage_pct,
            "ready": embeddings_ready,
        },
        "timestamps": {
            "last_ingested_at": _meta("last_ingested_at"),
            "rules_last_ingested_at": _meta("rules_last_ingested_at"),
            "trm_last_ingested_at": _meta("trm_last_ingested_at"),
            "cards_last_ingested_at": _meta("cards_last_ingested_at"),
            "banlist_last_ingested_at": banlist_last_ingested_at,
            "last_faq_ingested_at": _meta("last_faq_ingested_at"),
            "last_embed_at": _meta("last_embed_at"),
        },
    }
