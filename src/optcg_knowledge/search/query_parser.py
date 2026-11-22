"""OPTCG query token detection and safe SQLite FTS5 query construction."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches standard OPTCG card IDs: OP05-119, ST10-001, EB01-001, PRB01-001, P-001
_CARD_ID_RE = re.compile(
    r"\b(?:(?:OP|ST|EB|PRB)\d{2}-\d{3}|P-\d{3})\b",
    re.IGNORECASE,
)

# Matches hierarchical Comprehensive Rules IDs: 7-1-4, 7-1-4-1, 10-2-1-3-1
_RULE_ID_RE = re.compile(r"\b\d+(?:-\d+){1,4}\b")

# Matches canonical Tournament Rules Manual section IDs: TRM-1.0, TRM-7.4.1, TRM-7.4.1.2
_TRM_ID_RE = re.compile(r"\bTRM-\d+(?:\.\d+)+\b", re.IGNORECASE)

# Matches bracketed OPTCG keywords/triggers: [Blocker], [On K.O.], [DON!! x1], [Activate: Main]
_BRACKET_KEYWORD_RE = re.compile(r"\[([^\[\]]+)\]")

# Tokenizes query fragments for safe FTS5 quoting while keeping bracketed phrases & structured IDs.
_FTS_TOKEN_RE = re.compile(
    r"\[[^\[\]]+\]|[^\W_]+(?:[-.][^\W_]+)+|[^\W_]+",
    re.UNICODE,
)

# High-frequency English and Portuguese function words stripped only from relaxed (OR)
# FTS5 fallback queries so boilerplate tokens do not drown out semantic/exact hits.
_RELAXED_FTS_STOPWORDS: frozenset[str] = frozenset(
    {
        # English
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "may",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
        "you",
        "your",
        # Portuguese (PT-BR, excluding terms already listed above such as "a", "as", "do", "no")
        "ao",
        "aos",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dos",
        "e",
        "é",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "essa",
        "esse",
        "esta",
        "este",
        "eu",
        "foi",
        "mais",
        "mas",
        "me",
        "meu",
        "minha",
        "na",
        "nas",
        "nos",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "posso",
        "qual",
        "quais",
        "quando",
        "que",
        "quem",
        "são",
        "se",
        "sem",
        "ser",
        "seu",
        "sua",
        "também",
        "tem",
        "ter",
        "um",
        "uma",
        "umas",
        "uns",
    }
)


@dataclass(frozen=True, slots=True)
class DetectedTokens:
    """Structured OPTCG identifiers and keywords extracted from a user query."""

    card_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    trm_ids: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    @property
    def has_any(self) -> bool:
        return bool(self.card_ids or self.rule_ids or self.trm_ids or self.keywords)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "card_ids": list(self.card_ids),
            "rule_ids": list(self.rule_ids),
            "trm_ids": list(self.trm_ids),
            "keywords": list(self.keywords),
        }


def detect_query_tokens(query: str) -> DetectedTokens:
    """Extract canonical card IDs, rule numbers, and bracketed keywords from a query."""
    if not query or not query.strip():
        return DetectedTokens()

    card_ids: list[str] = []
    seen_cards: set[str] = set()
    card_spans: list[tuple[int, int]] = []
    for match in _CARD_ID_RE.finditer(query):
        normalized = match.group(0).upper()
        card_spans.append(match.span())
        if normalized not in seen_cards:
            seen_cards.add(normalized)
            card_ids.append(normalized)

    trm_ids: list[str] = []
    seen_trm: set[str] = set()
    trm_spans: list[tuple[int, int]] = []
    for match in _TRM_ID_RE.finditer(query):
        normalized_trm = match.group(0).upper()
        trm_spans.append(match.span())
        if normalized_trm not in seen_trm:
            seen_trm.add(normalized_trm)
            trm_ids.append(normalized_trm)

    rule_ids: list[str] = []
    seen_rules: set[str] = set()
    for match in _RULE_ID_RE.finditer(query):
        start, end = match.span()
        if any(start >= cs and end <= ce for cs, ce in card_spans):
            continue
        if any(start >= ts and end <= te for ts, te in trm_spans):
            continue
        rule_id = match.group(0)
        if rule_id not in seen_rules:
            seen_rules.add(rule_id)
            rule_ids.append(rule_id)

    keywords: list[str] = []
    seen_keywords: set[str] = set()
    for match in _BRACKET_KEYWORD_RE.finditer(query):
        inner = " ".join(match.group(1).split())
        if not inner:
            continue
        formatted = f"[{inner}]"
        lowered = formatted.lower()
        if lowered not in seen_keywords:
            seen_keywords.add(lowered)
            keywords.append(formatted)

    return DetectedTokens(
        card_ids=tuple(card_ids),
        rule_ids=tuple(rule_ids),
        trm_ids=tuple(trm_ids),
        keywords=tuple(keywords),
    )


def _quote_fts_term(term: str) -> str | None:
    """Safely quote a single term for FTS5 MATCH, returning None if it has no indexable chars."""
    cleaned = term.strip()
    if not cleaned or not any(ch.isalnum() for ch in cleaned):
        return None
    escaped = cleaned.replace('"', '""')
    return f'"{escaped}"'


def _is_relaxed_stopword(raw_term: str) -> bool:
    """Return True for high-frequency EN/PT-BR function words in relaxed OR fallback queries."""
    cleaned = raw_term.strip()
    if not cleaned:
        return True
    # Preserve bracketed keywords, structured IDs (card/rule/TRM), and numbers.
    if cleaned.startswith("[") or "-" in cleaned or "." in cleaned or cleaned.isdigit():
        return False
    if len(cleaned) <= 1:
        return True
    return cleaned.lower() in _RELAXED_FTS_STOPWORDS


def build_fts_queries(query: str) -> tuple[str | None, str | None]:
    """Build crash-safe ``(strict_and, relaxed_or)`` FTS5 MATCH queries from raw user input.

    Every extracted token is quoted so hyphens (``OP05-119``, ``7-1-4``), brackets
    (``[On K.O.]``), exclamation marks (``[DON!! x1]``), colons, and FTS operators
    (``NOT``, ``OR``, ``NEAR``) never trigger ``sqlite3.OperationalError``.

    ``strict_and`` retains all tokens for exact English/literal precision, while
    ``relaxed_or`` filters ultra-common EN/PT-BR stopwords (falling back to all tokens
    if filtering would leave none) to reduce noise on mixed-language queries.
    """
    if not query or not query.strip():
        return (None, None)

    quoted_terms: list[str] = []
    relaxed_terms: list[str] = []
    seen: set[str] = set()
    for match in _FTS_TOKEN_RE.finditer(query):
        raw_token = match.group(0)
        quoted = _quote_fts_term(raw_token)
        if quoted is None:
            continue
        dedup_key = quoted.lower()
        if dedup_key not in seen:
            seen.add(dedup_key)
            quoted_terms.append(quoted)
            if not _is_relaxed_stopword(raw_token):
                relaxed_terms.append(quoted)

    if not quoted_terms:
        return (None, None)

    strict_query = " AND ".join(quoted_terms)
    if len(quoted_terms) == 1:
        relaxed_query = None
    elif relaxed_terms:
        relaxed_query = " OR ".join(relaxed_terms)
    else:
        relaxed_query = " OR ".join(quoted_terms)
    return (strict_query, relaxed_query)
