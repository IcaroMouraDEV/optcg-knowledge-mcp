"""Official Bandai OPTCG banlist / restriction HTML & local JSON parser and ingestor.

Parses text-only official Bandai restriction HTML (``https://en.onepiece-cardgame.com/rules/restriction/``)
and local manual JSON files under ``data/raw/banlist/`` without downloading card images.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from optcg_knowledge.db.models import (
    MAX_COPIES_BY_BAN_STATUS,
    ActiveBanStatus,
    CardRestrictionRecord,
    normalize_ban_status,
)
from optcg_knowledge.db.schema import set_meta, utc_now_iso
from optcg_knowledge.ingest.cards_importer import parse_card_id

logger = logging.getLogger(__name__)

OFFICIAL_BANLIST_URL = "https://en.onepiece-cardgame.com/rules/restriction/"

_CARD_ID_RE = re.compile(
    r"\b((?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})\b",
    re.IGNORECASE,
)
_EFFECTIVE_FROM_RE = re.compile(
    r"effective\s+from\s+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{2}[-/]\d{2})",
    re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLASH_DATE_RE = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")

_STATUS_PRECEDENCE: dict[ActiveBanStatus, int] = {
    "banned": 1,
    "restricted": 2,
    "pair_banned": 3,
}


@dataclass(frozen=True, slots=True)
class ParsedBanlistDocument:
    """Normalized banlist restrictions parsed from official HTML or local JSON."""

    restrictions: tuple[CardRestrictionRecord, ...]
    effective_date: str | None = None
    source_url: str | None = None
    source_name: str = "local"
    upcoming_effective_date: str | None = None
    upcoming_included: bool = False


@dataclass(frozen=True, slots=True)
class BanlistIngestSummary:
    """Counters returned after idempotently ingesting card restrictions."""

    restrictions_upserted: int
    restrictions_removed: int
    banned_count: int
    restricted_count: int
    pair_banned_count: int
    effective_date: str | None
    source_name: str


def _normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return None
    if _ISO_DATE_RE.match(cleaned):
        return cleaned
    slash_match = _SLASH_DATE_RE.match(cleaned)
    if slash_match:
        year, month, day = (int(slash_match.group(i)) for i in (1, 2, 3))
        return f"{year:04d}-{month:02d}-{day:02d}"
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return cleaned


def _extract_card_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CARD_ID_RE.findall(text):
        canonical = match.upper()
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)
    return ordered


def _default_restriction_note(
    status: ActiveBanStatus,
    paired_card_ids: tuple[str, ...],
) -> str:
    if status == "banned":
        return "Cannot be included in any deck."
    if status == "restricted":
        return "Limited to 1 copy per deck."
    if paired_card_ids:
        return f"Cannot be included in the same deck as {', '.join(paired_card_ids)}."
    return "Cannot be included in the same deck as its paired card(s)."


class _RawSectionBucket:
    def __init__(self, effective_date: str | None = None) -> None:
        self.effective_date = effective_date
        self.banned: list[str] = []
        self.restricted: list[str] = []
        self.pairs: list[tuple[str, str]] = []


class _BandaiBanlistHTMLParser(HTMLParser):
    """Streaming stdlib HTML parser for Bandai's official restriction page."""

    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "header", "footer", "nav"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0

        self.page_date: str | None = None
        self.has_active_section = False
        self.active_bucket = _RawSectionBucket()
        self.upcoming_buckets: list[_RawSectionBucket] = []
        self.archive_effective_dates: list[str] = []

        self._section_kind: str = "other"  # "active" | "upcoming" | "other"
        self._current_bucket: _RawSectionBucket | None = None
        self._subsection: ActiveBanStatus | None = None

        self._in_page_date = False
        self._page_date_buf: list[str] = []

        self._heading_tag: str | None = None
        self._heading_buf: list[str] = []

        self._in_ul = False
        self._in_li = False
        self._li_buf: list[str] = []
        self._ul_cards: list[str] = []

        self._pair_side: str | None = None  # "a" | "b"
        self._pair_side_a: list[str] = []
        self._pair_side_b: list[str] = []
        self._in_p = False
        self._p_buf: list[str] = []

        self._in_anchor = False
        self._anchor_buf: list[str] = []

    def _flush_pair_sides(self) -> None:
        if self._current_bucket is not None and self._pair_side_a and self._pair_side_b:
            for card_a in self._pair_side_a:
                for card_b in self._pair_side_b:
                    if card_a != card_b:
                        self._current_bucket.pairs.append((card_a, card_b))
        self._pair_side = None
        self._pair_side_a = []
        self._pair_side_b = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.lower()
        if lower in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth > 0:
            return
        # Never inspect <img> tags or alt/src attributes.
        if lower == "img":
            return

        attr_map = {k.lower(): (v or "") for k, v in attrs}
        classes = set(attr_map.get("class", "").split())

        if lower == "span" and "pageTitDate" in classes:
            self._in_page_date = True
            self._page_date_buf = []
            return

        if lower == "br":
            if self._heading_tag is not None:
                self._heading_buf.append(" ")
            if self._in_li:
                self._li_buf.append("\n")
            if self._in_p:
                self._p_buf.append("\n")
            if self._in_anchor:
                self._anchor_buf.append(" ")
            return

        if lower in ("h3", "h4", "h5"):
            self._heading_tag = lower
            self._heading_buf = []
            return

        if lower == "ul" and self._current_bucket is not None and self._subsection is not None:
            self._in_ul = True
            self._ul_cards = []
            return

        if lower == "li" and self._in_ul:
            self._in_li = True
            self._li_buf = []
            return

        if lower == "p" and self._current_bucket is not None and self._subsection is not None:
            self._in_p = True
            self._p_buf = []
            return

        if lower == "a":
            self._in_anchor = True
            self._anchor_buf = []

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in self._IGNORED_TAGS:
            if self._ignored_depth > 0:
                self._ignored_depth -= 1
            return
        if self._ignored_depth > 0:
            return

        if lower == "span" and self._in_page_date:
            self._in_page_date = False
            date_text = " ".join("".join(self._page_date_buf).split())
            if date_text and self.page_date is None:
                self.page_date = _normalize_date(date_text)
            return

        if lower == self._heading_tag:
            heading_text = " ".join("".join(self._heading_buf).split())
            self._heading_tag = None
            self._heading_buf = []
            self._handle_heading(lower, heading_text)
            return

        if lower == "li" and self._in_li:
            self._in_li = False
            li_text = "".join(self._li_buf)
            self._li_buf = []
            self._ul_cards.extend(_extract_card_ids(li_text))
            return

        if lower == "ul" and self._in_ul:
            self._in_ul = False
            self._handle_ul_cards()
            self._ul_cards = []
            return

        if lower == "p" and self._in_p:
            self._in_p = False
            p_text = "".join(self._p_buf)
            self._p_buf = []
            if self._subsection == "pair_banned" and self._pair_side is not None:
                cards = _extract_card_ids(p_text)
                if self._pair_side == "a":
                    self._pair_side_a.extend(cards)
                elif self._pair_side == "b":
                    self._pair_side_b.extend(cards)
            return

        if lower == "a" and self._in_anchor:
            self._in_anchor = False
            anchor_text = " ".join("".join(self._anchor_buf).split())
            self._anchor_buf = []
            if self.has_active_section:
                match = _EFFECTIVE_FROM_RE.search(anchor_text)
                if match:
                    parsed_date = _normalize_date(match.group(1))
                    if parsed_date and parsed_date not in self.archive_effective_dates:
                        self.archive_effective_dates.append(parsed_date)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        if self._in_page_date:
            self._page_date_buf.append(data)
        if self._heading_tag is not None:
            self._heading_buf.append(data)
        if self._in_li:
            self._li_buf.append(data)
        if self._in_p:
            self._p_buf.append(data)
        if self._in_anchor:
            self._anchor_buf.append(data)

    def _handle_heading(self, tag: str, text: str) -> None:
        lower = text.lower()
        if tag == "h3":
            self._flush_pair_sides()
            self._subsection = None
            if "active restrictions" in lower or "current" in lower and "restriction" in lower:
                self.has_active_section = True
                self._section_kind = "active"
                self._current_bucket = self.active_bucket
                match = _EFFECTIVE_FROM_RE.search(text)
                if match:
                    self.active_bucket.effective_date = _normalize_date(match.group(1))
                return

            eff_match = _EFFECTIVE_FROM_RE.search(text)
            if eff_match and ("banned" in lower or "restricted" in lower):
                self._section_kind = "upcoming"
                bucket = _RawSectionBucket(effective_date=_normalize_date(eff_match.group(1)))
                self.upcoming_buckets.append(bucket)
                self._current_bucket = bucket
                return

            self._section_kind = "other"
            self._current_bucket = None
            return

        if tag == "h4":
            if self._current_bucket is None:
                return
            if "announced" in lower:
                return
            self._flush_pair_sides()
            if "banned pair" in lower or "pair ban" in lower:
                self._subsection = "pair_banned"
            elif "banned" in lower:
                self._subsection = "banned"
            elif "restricted" in lower or "limited" in lower:
                self._subsection = "restricted"
            else:
                self._subsection = None
            return

        if tag == "h5":
            if self._subsection != "pair_banned" or self._current_bucket is None:
                return
            if "card a" in lower:
                if self._pair_side_b:
                    self._flush_pair_sides()
                self._pair_side = "a"
            elif "card b" in lower:
                self._pair_side = "b"

    def _handle_ul_cards(self) -> None:
        if self._current_bucket is None or self._subsection is None:
            return
        deduped = list(dict.fromkeys(self._ul_cards))
        if not deduped:
            return
        if self._subsection == "banned":
            self._current_bucket.banned.extend(deduped)
        elif self._subsection == "restricted":
            self._current_bucket.restricted.extend(deduped)
        elif self._subsection == "pair_banned":
            if len(deduped) == 2:
                self._current_bucket.pairs.append((deduped[0], deduped[1]))
            else:
                logger.warning(
                    "Skipping unexpected banned-pair list with %d card IDs: %s",
                    len(deduped),
                    deduped,
                )

    def finalize(self) -> None:
        self._flush_pair_sides()


@dataclass
class _MutableRestriction:
    card_id: str
    ban_status: ActiveBanStatus
    max_copies: int
    paired_card_ids: set[str]
    restriction_note: str | None
    effective_date: str | None
    source_url: str | None


def _merge_entry(
    entries: dict[str, _MutableRestriction],
    *,
    card_id: str,
    ban_status: ActiveBanStatus,
    max_copies: int | None = None,
    paired_card_ids: tuple[str, ...] = (),
    restriction_note: str | None = None,
    effective_date: str | None = None,
    source_url: str | None = None,
) -> None:
    canonical_id = parse_card_id(card_id)[0]
    cleaned_pairs = {
        parse_card_id(pid)[0]
        for pid in paired_card_ids
        if pid.strip() and parse_card_id(pid)[0] != canonical_id
    }
    resolved_max = (
        max_copies if max_copies is not None else MAX_COPIES_BY_BAN_STATUS[ban_status]
    )
    existing = entries.get(canonical_id)
    if existing is None:
        entries[canonical_id] = _MutableRestriction(
            card_id=canonical_id,
            ban_status=ban_status,
            max_copies=resolved_max,
            paired_card_ids=set(cleaned_pairs),
            restriction_note=restriction_note,
            effective_date=effective_date,
            source_url=source_url,
        )
        return

    existing.paired_card_ids.update(cleaned_pairs)
    # Stricter status takes precedence: banned > restricted > pair_banned.
    if _STATUS_PRECEDENCE[ban_status] < _STATUS_PRECEDENCE[existing.ban_status]:
        existing.ban_status = ban_status
        existing.max_copies = resolved_max
        if restriction_note:
            existing.restriction_note = restriction_note
    elif ban_status == existing.ban_status and restriction_note:
        existing.restriction_note = restriction_note

    if effective_date and (
        existing.effective_date is None or effective_date > existing.effective_date
    ):
        existing.effective_date = effective_date
    if source_url and not existing.source_url:
        existing.source_url = source_url


def _finalize_entries(
    entries: dict[str, _MutableRestriction],
    *,
    default_effective_date: str | None,
    default_source_url: str | None,
) -> tuple[CardRestrictionRecord, ...]:
    records: list[CardRestrictionRecord] = []
    for card_id in sorted(entries):
        item = entries[card_id]
        paired = tuple(sorted(item.paired_card_ids))
        note = item.restriction_note or _default_restriction_note(item.ban_status, paired)
        records.append(
            CardRestrictionRecord(
                card_id=card_id,
                ban_status=item.ban_status,
                max_copies=item.max_copies,
                paired_card_ids=paired,
                restriction_note=note,
                effective_date=item.effective_date or default_effective_date,
                source_url=item.source_url or default_source_url,
            )
        )
    return tuple(records)


def _apply_bucket(
    entries: dict[str, _MutableRestriction],
    bucket: _RawSectionBucket,
    *,
    effective_date: str | None,
    source_url: str | None,
) -> None:
    eff_date = bucket.effective_date or effective_date
    for cid in bucket.banned:
        _merge_entry(
            entries,
            card_id=cid,
            ban_status="banned",
            effective_date=eff_date,
            source_url=source_url,
        )
    for cid in bucket.restricted:
        _merge_entry(
            entries,
            card_id=cid,
            ban_status="restricted",
            effective_date=eff_date,
            source_url=source_url,
        )
    for card_a, card_b in bucket.pairs:
        _merge_entry(
            entries,
            card_id=card_a,
            ban_status="pair_banned",
            paired_card_ids=(card_b,),
            effective_date=eff_date,
            source_url=source_url,
        )
        _merge_entry(
            entries,
            card_id=card_b,
            ban_status="pair_banned",
            paired_card_ids=(card_a,),
            effective_date=eff_date,
            source_url=source_url,
        )


def parse_banlist_html(
    html: str,
    *,
    effective_date: str | None = None,
    source_url: str | None = OFFICIAL_BANLIST_URL,
    include_upcoming: bool = False,
    source_name: str = "bandai_official_html",
) -> ParsedBanlistDocument:
    """Parse Bandai's official restriction HTML page into canonical CardRestrictionRecords.

    By default, ingests only the ``Cards with Active Restrictions`` section so
    upcoming announcements do not prematurely mark currently legal cards.
    Set ``include_upcoming=True`` to also include scheduled future-effective changes.
    """
    parser = _BandaiBanlistHTMLParser()
    parser.feed(html)
    parser.close()
    parser.finalize()

    upcoming_date = (
        parser.upcoming_buckets[0].effective_date if parser.upcoming_buckets else None
    )

    active_effective = _normalize_date(effective_date) or parser.active_bucket.effective_date
    if active_effective is None:
        if parser.has_active_section and parser.archive_effective_dates:
            active_effective = parser.archive_effective_dates[0]
        elif not parser.has_active_section and upcoming_date:
            active_effective = upcoming_date
        else:
            active_effective = parser.page_date

    entries: dict[str, _MutableRestriction] = {}
    upcoming_applied = False

    if parser.has_active_section:
        _apply_bucket(
            entries,
            parser.active_bucket,
            effective_date=active_effective,
            source_url=source_url,
        )
        if include_upcoming and parser.upcoming_buckets:
            upcoming_applied = True
            for bucket in parser.upcoming_buckets:
                _apply_bucket(
                    entries,
                    bucket,
                    effective_date=bucket.effective_date or active_effective,
                    source_url=source_url,
                )
    elif parser.upcoming_buckets:
        # Standalone dated announcement page without a separate active list section.
        for bucket in parser.upcoming_buckets:
            _apply_bucket(
                entries,
                bucket,
                effective_date=bucket.effective_date or active_effective,
                source_url=source_url,
            )

    doc_effective = (
        _normalize_date(effective_date)
        or (upcoming_date if upcoming_applied and upcoming_date else active_effective)
    )
    records = _finalize_entries(
        entries,
        default_effective_date=active_effective,
        default_source_url=source_url,
    )
    return ParsedBanlistDocument(
        restrictions=records,
        effective_date=doc_effective,
        source_url=source_url,
        source_name=source_name,
        upcoming_effective_date=upcoming_date,
        upcoming_included=upcoming_applied,
    )


def parse_banlist_json(
    raw_json: str | dict[str, Any] | list[dict[str, Any]],
    *,
    effective_date: str | None = None,
    source_url: str | None = None,
    source_name: str = "local_json",
) -> tuple[ParsedBanlistDocument, set[str]]:
    """Parse a manual local banlist JSON payload.

    Returns ``(ParsedBanlistDocument, explicitly_legal_card_ids)`` so directory overlays
    can explicitly mark a previously restricted card as ``"legal"``.
    """
    payload = json.loads(raw_json) if isinstance(raw_json, str) else raw_json

    doc_effective = _normalize_date(effective_date)
    doc_source_url = source_url
    raw_entries: list[Any] = []
    raw_pairs: list[Any] = []

    if isinstance(payload, dict):
        if doc_effective is None and isinstance(payload.get("effective_date"), str):
            doc_effective = _normalize_date(payload["effective_date"])
        if doc_source_url is None and isinstance(payload.get("source_url"), str):
            doc_source_url = payload["source_url"].strip() or None
        restrictions_val = payload.get("restrictions", [])
        if not isinstance(restrictions_val, list):
            raise ValueError("Banlist JSON 'restrictions' must be a list")
        raw_entries = restrictions_val
        pairs_val = payload.get("banned_pairs", [])
        if not isinstance(pairs_val, list):
            raise ValueError("Banlist JSON 'banned_pairs' must be a list")
        raw_pairs = pairs_val
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raise ValueError("Banlist JSON root must be an object or a list of restriction objects")

    entries: dict[str, _MutableRestriction] = {}
    explicitly_legal: set[str] = set()

    for idx, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise ValueError(f"Banlist restriction at index {idx} must be a JSON object")
        raw_cid = item.get("card_id")
        if not isinstance(raw_cid, str) or not raw_cid.strip():
            raise ValueError(f"Banlist restriction at index {idx} is missing 'card_id'")
        canonical_id = parse_card_id(raw_cid)[0]

        raw_status = item.get("ban_status") or item.get("status")
        if not isinstance(raw_status, str) or not raw_status.strip():
            raise ValueError(f"Banlist restriction for {canonical_id} is missing 'ban_status'")
        status = normalize_ban_status(raw_status)
        if status == "legal":
            explicitly_legal.add(canonical_id)
            continue

        raw_max = item.get("max_copies")
        if raw_max is not None:
            if (
                isinstance(raw_max, bool)
                or not isinstance(raw_max, int)
                or raw_max not in (0, 1, 4)
            ):
                raise ValueError(
                    f"Banlist restriction for {canonical_id} has invalid max_copies: {raw_max!r}"
                )
            max_copies = raw_max
        else:
            max_copies = MAX_COPIES_BY_BAN_STATUS[status]

        raw_paired = item.get("paired_card_ids") or []
        if not isinstance(raw_paired, list):
            raise ValueError(
                f"Banlist restriction for {canonical_id} has non-list 'paired_card_ids'"
            )
        paired_ids: list[str] = []
        for partner in raw_paired:
            if not isinstance(partner, str) or not partner.strip():
                raise ValueError(
                    f"Banlist restriction for {canonical_id} has invalid partner ID: {partner!r}"
                )
            partner_id = parse_card_id(partner)[0]
            if partner_id == canonical_id:
                raise ValueError(f"Card {canonical_id} cannot be pair_banned with itself")
            paired_ids.append(partner_id)

        if status == "pair_banned" and not paired_ids:
            raise ValueError(
                f"Card {canonical_id} has ban_status='pair_banned' without paired_card_ids"
            )

        note_val = item.get("restriction_note") or item.get("note")
        note = note_val.strip() if isinstance(note_val, str) and note_val.strip() else None
        item_eff = (
            _normalize_date(item["effective_date"])
            if isinstance(item.get("effective_date"), str)
            else doc_effective
        )
        item_url = (
            item["source_url"].strip()
            if isinstance(item.get("source_url"), str) and item["source_url"].strip()
            else doc_source_url
        )

        _merge_entry(
            entries,
            card_id=canonical_id,
            ban_status=status,
            max_copies=max_copies,
            paired_card_ids=tuple(paired_ids),
            restriction_note=note,
            effective_date=item_eff,
            source_url=item_url,
        )
        # Ensure pair_banned relationships are symmetric across both partner cards.
        for partner_id in paired_ids:
            _merge_entry(
                entries,
                card_id=partner_id,
                ban_status="pair_banned",
                paired_card_ids=(canonical_id,),
                effective_date=item_eff,
                source_url=item_url,
            )

    for idx, pair in enumerate(raw_pairs):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(c, str) and c.strip() for c in pair)
        ):
            raise ValueError(f"banned_pairs entry at index {idx} must be a 2-element card_id list")
        card_a = parse_card_id(pair[0])[0]
        card_b = parse_card_id(pair[1])[0]
        if card_a == card_b:
            raise ValueError(f"Card {card_a} cannot be pair_banned with itself")
        _merge_entry(
            entries,
            card_id=card_a,
            ban_status="pair_banned",
            paired_card_ids=(card_b,),
            effective_date=doc_effective,
            source_url=doc_source_url,
        )
        _merge_entry(
            entries,
            card_id=card_b,
            ban_status="pair_banned",
            paired_card_ids=(card_a,),
            effective_date=doc_effective,
            source_url=doc_source_url,
        )

    records = _finalize_entries(
        entries,
        default_effective_date=doc_effective,
        default_source_url=doc_source_url,
    )
    return (
        ParsedBanlistDocument(
            restrictions=records,
            effective_date=doc_effective,
            source_url=doc_source_url,
            source_name=source_name,
        ),
        explicitly_legal,
    )


def parse_banlist_path(
    path: Path | str,
    *,
    effective_date: str | None = None,
    source_url: str | None = None,
    include_upcoming: bool = False,
) -> ParsedBanlistDocument:
    """Parse a banlist HTML file, manual JSON file, or directory under ``data/raw/banlist/``."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(
            f"Banlist path not found: {target}. "
            "Run `optcg-ingest fetch-official --only banlist` or place a local JSON/HTML file "
            "under data/raw/banlist/."
        )

    if target.is_file():
        suffix = target.suffix.lower()
        if suffix in (".html", ".htm"):
            return parse_banlist_html(
                target.read_text(encoding="utf-8", errors="replace"),
                effective_date=effective_date,
                source_url=source_url or OFFICIAL_BANLIST_URL,
                include_upcoming=include_upcoming,
                source_name=target.name,
            )
        if suffix == ".json":
            doc, _ = parse_banlist_json(
                target.read_text(encoding="utf-8"),
                effective_date=effective_date,
                source_url=source_url,
                source_name=target.name,
            )
            return doc
        raise ValueError(
            f"Unsupported banlist file type {target.suffix!r} for {target}; "
            "expected .html, .htm, or .json"
        )

    html_files = sorted(
        p for p in target.iterdir() if p.is_file() and p.suffix.lower() in (".html", ".htm")
    )
    json_files = sorted(
        p for p in target.iterdir() if p.is_file() and p.suffix.lower() == ".json"
    )
    if not html_files and not json_files:
        raise FileNotFoundError(
            f"No .html or .json banlist files found in {target}. "
            "Run `optcg-ingest fetch-official --only banlist` or add a local JSON file."
        )

    merged: dict[str, CardRestrictionRecord] = {}
    resolved_effective: str | None = _normalize_date(effective_date)
    resolved_url: str | None = source_url
    upcoming_date: str | None = None
    upcoming_included = False
    sources_used: list[str] = []

    for html_path in html_files:
        doc = parse_banlist_html(
            html_path.read_text(encoding="utf-8", errors="replace"),
            effective_date=effective_date,
            source_url=source_url or OFFICIAL_BANLIST_URL,
            include_upcoming=include_upcoming,
            source_name=html_path.name,
        )
        sources_used.append(html_path.name)
        for rec in doc.restrictions:
            merged[rec.card_id] = rec
        if resolved_effective is None and doc.effective_date:
            resolved_effective = doc.effective_date
        if resolved_url is None and doc.source_url:
            resolved_url = doc.source_url
        if doc.upcoming_effective_date:
            upcoming_date = doc.upcoming_effective_date
        upcoming_included = upcoming_included or doc.upcoming_included

    for json_path in json_files:
        doc, explicitly_legal = parse_banlist_json(
            json_path.read_text(encoding="utf-8"),
            effective_date=effective_date,
            source_url=source_url,
            source_name=json_path.name,
        )
        sources_used.append(json_path.name)
        for legal_id in explicitly_legal:
            merged.pop(legal_id, None)
            for cid, existing in list(merged.items()):
                if legal_id in existing.paired_card_ids:
                    remaining_pairs = tuple(p for p in existing.paired_card_ids if p != legal_id)
                    if existing.ban_status == "pair_banned" and not remaining_pairs:
                        merged.pop(cid, None)
                    else:
                        merged[cid] = CardRestrictionRecord(
                            card_id=existing.card_id,
                            ban_status=existing.ban_status,
                            max_copies=existing.max_copies,
                            paired_card_ids=remaining_pairs,
                            restriction_note=_default_restriction_note(
                                existing.ban_status, remaining_pairs
                            ),
                            effective_date=existing.effective_date,
                            source_url=existing.source_url,
                        )
        for rec in doc.restrictions:
            merged[rec.card_id] = rec
        if effective_date is None and doc.effective_date:
            resolved_effective = doc.effective_date
        if source_url is None and doc.source_url:
            resolved_url = doc.source_url

    ordered_records = tuple(merged[cid] for cid in sorted(merged))
    return ParsedBanlistDocument(
        restrictions=ordered_records,
        effective_date=resolved_effective,
        source_url=resolved_url,
        source_name=",".join(sources_used),
        upcoming_effective_date=upcoming_date,
        upcoming_included=upcoming_included,
    )


def ingest_banlist(
    conn: sqlite3.Connection,
    doc: ParsedBanlistDocument,
) -> BanlistIngestSummary:
    """Idempotently replace active card restrictions in SQLite without touching chunks_vec."""
    existing_ids = {
        str(row[0])
        for row in conn.execute("SELECT card_id FROM card_restrictions").fetchall()
    }
    incoming_ids = {rec.card_id for rec in doc.restrictions}
    stale_ids = sorted(existing_ids - incoming_ids)

    with conn:
        if stale_ids:
            conn.executemany(
                "DELETE FROM card_restrictions WHERE card_id = ?",
                [(cid,) for cid in stale_ids],
            )

        if doc.restrictions:
            conn.executemany(
                """
                INSERT INTO card_restrictions (
                    card_id, ban_status, max_copies, paired_card_ids,
                    restriction_note, effective_date, source_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    ban_status = excluded.ban_status,
                    max_copies = excluded.max_copies,
                    paired_card_ids = excluded.paired_card_ids,
                    restriction_note = excluded.restriction_note,
                    effective_date = excluded.effective_date,
                    source_url = excluded.source_url
                """,
                [
                    (
                        rec.card_id,
                        rec.ban_status,
                        rec.max_copies,
                        json.dumps(list(rec.paired_card_ids), ensure_ascii=False),
                        rec.restriction_note,
                        rec.effective_date,
                        rec.source_url,
                    )
                    for rec in doc.restrictions
                ],
            )

        now = utc_now_iso()
        if doc.effective_date:
            set_meta(conn, "banlist_effective_date", doc.effective_date)
        else:
            conn.execute("DELETE FROM meta WHERE key = 'banlist_effective_date'")
        set_meta(conn, "banlist_source", doc.source_name)
        set_meta(conn, "banlist_last_ingested_at", now)
        set_meta(conn, "last_ingested_at", now)

    banned_count = sum(1 for r in doc.restrictions if r.ban_status == "banned")
    restricted_count = sum(1 for r in doc.restrictions if r.ban_status == "restricted")
    pair_banned_count = sum(1 for r in doc.restrictions if r.ban_status == "pair_banned")

    return BanlistIngestSummary(
        restrictions_upserted=len(doc.restrictions),
        restrictions_removed=len(stale_ids),
        banned_count=banned_count,
        restricted_count=restricted_count,
        pair_banned_count=pair_banned_count,
        effective_date=doc.effective_date,
        source_name=doc.source_name,
    )
