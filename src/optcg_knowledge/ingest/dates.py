"""Date normalization utilities for official Bandai OPTCG documents."""

from __future__ import annotations

import re
from datetime import datetime

_YMD_DATE_RE = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
_MDY_SLASH_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_ABBREV_MONTH_DOT_RE = re.compile(r"^([A-Za-z]{3})\.\s+")

_NAMED_MONTH_FORMATS: tuple[str, ...] = (
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
)


def normalize_official_date(raw: str | None) -> str | None:
    """Normalize an official Bandai date string to ISO ``YYYY-MM-DD``.

    Supports:
    - ``YYYY-MM-DD``, ``YYYY/M/D``, ``YYYY.M.D``
    - ``M/D/YYYY`` (US format used in official EN Bandai PDFs, e.g. ``8/28/2026``)
    - ``Month D, YYYY``, ``Month D YYYY``, ``Mon D, YYYY``, ``Mon. D, YYYY``

    Returns ``None`` for ``None`` or blank input, and falls back to the cleaned
    input string if it does not match a recognized date format.
    """
    if not raw:
        return None

    cleaned = " ".join(raw.strip().split())
    if not cleaned:
        return None

    ymd_match = _YMD_DATE_RE.fullmatch(cleaned)
    if ymd_match:
        year, month, day = (int(part) for part in ymd_match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return cleaned

    mdy_match = _MDY_SLASH_DATE_RE.fullmatch(cleaned)
    if mdy_match:
        month, day, year = (int(part) for part in mdy_match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return cleaned

    candidate = _ABBREV_MONTH_DOT_RE.sub(r"\1 ", cleaned)
    for fmt in _NAMED_MONTH_FORMATS:
        try:
            return datetime.strptime(candidate, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return cleaned
