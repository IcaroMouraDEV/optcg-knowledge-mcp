"""OPTCG query token detection and safe SQLite FTS5 query construction."""

from __future__ import annotations

import re
import unicodedata
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

# Compact, deterministic PT-BR -> canonical English OPTCG phrase glossary used ONLY
# to expand relaxed (OR) FTS5 fallback queries so pure Portuguese queries can still hit
# English Comprehensive Rules and TRM sections when local embeddings are absent.
# Keys are accent-folded lowercase phrases; zero external APIs or LLMs are used.
_PTBR_PHRASE_GLOSSARY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fase de renovacao", ("Refresh Phase",)),
    ("fase de compra", ("Draw Phase",)),
    ("fase principal", ("Main Phase",)),
    ("fase final", ("End Phase",)),
    ("passo de ataque", ("Attack Step",)),
    ("passo de bloqueio", ("Block Step",)),
    ("passo de contra ataque", ("Counter Step",)),
    ("passo de dano", ("Damage Step",)),
    ("carta de recurso", ("DON!!",)),
    ("cartas de recurso", ("DON!!",)),
    ("baralho de recurso", ("DON!! deck",)),
    ("baralho de recursos", ("DON!! deck",)),
    ("carta de vida", ("Life card", "Life")),
    ("cartas de vida", ("Life card", "Life")),
    ("sem cartas de vida", ("0 Life",)),
    ("zero de vida", ("0 Life",)),
    ("0 de vida", ("0 Life",)),
    ("igual ou maior", ("greater than or equal",)),
    ("maior ou igual", ("greater than or equal",)),
    ("ataque duplo", ("Double Attack",)),
    ("dois de dano", ("Double Attack",)),
    ("em vez disso", ("instead", "replacement")),
    ("efeito de substituicao", ("replacement effects", "instead")),
    ("efeitos de substituicao", ("replacement effects", "instead")),
    ("repeticao infinita", ("infinite loop",)),
    ("ciclo infinito", ("infinite loop",)),
    ("loop infinito", ("infinite loop",)),
    ("empate intencional", ("Intentional Draws", "Intentional Draw")),
    ("empates intencionais", ("Intentional Draws", "Intentional Draw")),
    ("juiz chefe", ("Head Judge",)),
    ("pedido de recurso", ("Appeals", "appeal")),
    ("pedir recurso", ("Appeals", "appeal")),
    ("jogo lento", ("Slow Play",)),
    ("demora demais", ("Slow Play",)),
    ("enrola o relogio", ("Slow Play",)),
    ("jogador ativo", ("turn player",)),
    ("jogador do turno", ("turn player",)),
    ("ao mesmo tempo", ("same time",)),
    ("manual de regras", ("Comprehensive Rules",)),
    ("regras gerais", ("Comprehensive Rules",)),
    ("texto da carta", ("card text",)),
    ("texto de uma carta", ("card text",)),
    ("baralho principal", ("50-card deck", "deck")),
    ("montagem do baralho", ("deck", "Leader")),
    ("copias iguais", ("same card number", "copies")),
    ("primeiro turno", ("first turn",)),
    ("sem ativar", ("without activating",)),
)

# Single-word PT-BR -> OPTCG EN domain vocabulary for relaxed FTS5 fallback.
# Keys are accent-folded Portuguese words chosen to avoid collisions with English terms
# and to omit ultra-high-frequency corpus boilerplate ("card", "effect", "rules",
# "tournament", "penalty", "play") that would dilute FTS5 BM25 precision.
# Polysemous words like "recurso" (DON!! vs judge appeal) are intentionally handled
# only via multi-word phrases above.
_PTBR_TERM_GLOSSARY: dict[str, tuple[str, ...]] = {
    "lider": ("Leader",),
    "lideres": ("Leader",),
    "personagem": ("Character",),
    "personagens": ("Character",),
    "evento": ("Event",),
    "eventos": ("Event",),
    "estagio": ("Stage",),
    "estagios": ("Stage",),
    "vida": ("Life",),
    "vidas": ("Life",),
    "descarte": ("trash",),
    "descartar": ("trash",),
    "descartado": ("trash",),
    "descartada": ("trash",),
    "cemiterio": ("trash",),
    "baralho": ("deck",),
    "baralhos": ("deck",),
    "mao": ("hand",),
    "turno": ("turn",),
    "turnos": ("turn",),
    "ataque": ("attack",),
    "atacar": ("attack",),
    "atacante": ("attack",),
    "ataca": ("attack",),
    "batalha": ("battle",),
    "batalhar": ("battle",),
    "dano": ("damage",),
    "danos": ("damage",),
    "poder": ("power",),
    "bloquear": ("Blocker", "block"),
    "bloqueador": ("Blocker", "block"),
    "bloqueio": ("Blocker", "block"),
    "bloqueia": ("Blocker", "block"),
    "redirecionar": ("Blocker", "target"),
    "substituicao": ("replacement", "instead"),
    "substituir": ("replacement", "instead"),
    "substitui": ("replacement", "instead"),
    "compra": ("draw",),
    "comprar": ("draw",),
    "comprando": ("draw",),
    "jogado": ("played",),
    "jogada": ("played",),
    "jogar": ("played",),
    "ativar": ("activate",),
    "ativacao": ("activate",),
    "ativado": ("activate",),
    "acionar": ("activate",),
    "acionado": ("activate",),
    "acionados": ("activate",),
    "copia": ("copies",),
    "copias": ("copies",),
    "oponente": ("opponent",),
    "adversario": ("opponent",),
    "demora": ("Slow Play",),
    "enrola": ("Slow Play",),
    "lento": ("Slow Play",),
    "apelacao": ("appeal",),
    "recorrer": ("appeal",),
    "revelar": ("reveal",),
    "revelada": ("reveal",),
    "revelado": ("reveal",),
    "descansado": ("rested",),
    "descansada": ("rested",),
    "virado": ("rested",),
    "deitado": ("rested",),
    "custo": ("cost",),
    "contragolpe": ("Counter",),
    "banir": ("Banish",),
    "banimento": ("Banish",),
    "impeto": ("Rush",),
    "gatilho": ("Trigger",),
    "prevalece": ("precedence",),
    "prevalecer": ("precedence",),
    "contradiz": ("contradicts",),
    "contradizer": ("contradicts",),
    "interromper": ("stop",),
    "infinita": ("infinite", "loop"),
    "infinito": ("infinite", "loop"),
    "simultaneo": ("simultaneous",),
    "simultanea": ("simultaneous",),
    "simultaneos": ("simultaneous",),
    "simultaneas": ("simultaneous",),
    "habilidade": ("Keyword",),
    "habilidades": ("Keyword",),
    "vence": ("wins",),
    "vencer": ("wins",),
    "vitoria": ("wins",),
    "derrota": ("loses",),
    "perde": ("loses",),
    "perder": ("loses",),
}

_FOLD_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


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


def _fold_ptbr_text(text: str) -> str:
    """Fold accents and punctuation to ASCII-like lowercase tokens for glossary lookup."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    parts = [part for part in _FOLD_SPLIT_RE.split(stripped.lower()) if part]
    return " ".join(parts)


def _expand_ptbr_glossary_terms(query: str) -> list[str]:
    """Return canonical English OPTCG terms mapped from PT-BR domain words/phrases."""
    folded = _fold_ptbr_text(query)
    if not folded:
        return []

    padded = f" {folded} "
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            expanded.append(term)

    for phrase, targets in _PTBR_PHRASE_GLOSSARY:
        if f" {phrase} " in padded:
            for target in targets:
                _add(target)

    for word in folded.split():
        targets = _PTBR_TERM_GLOSSARY.get(word)
        if targets is not None:
            for target in targets:
                _add(target)

    return expanded


def build_fts_queries(query: str) -> tuple[str | None, str | None]:
    """Build crash-safe ``(strict_and, relaxed_or)`` FTS5 MATCH queries from raw user input.

    Every extracted token is quoted so hyphens (``OP05-119``, ``7-1-4``), brackets
    (``[On K.O.]``), exclamation marks (``[DON!! x1]``), colons, and FTS operators
    (``NOT``, ``OR``, ``NEAR``) never trigger ``sqlite3.OperationalError``.

    ``strict_and`` retains all tokens for exact English/literal precision, while
    ``relaxed_or`` filters ultra-common EN/PT-BR stopwords (falling back to all tokens
    if filtering would leave none) and appends deterministic PT-BR -> OPTCG EN glossary
    terms so pure Portuguese queries can still retrieve English rules/TRM in lexical mode.
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

    glossary_quoted: list[str] = []
    relaxed_seen = {term.lower() for term in relaxed_terms}
    for mapped in _expand_ptbr_glossary_terms(query):
        quoted_mapped = _quote_fts_term(mapped)
        if quoted_mapped is None:
            continue
        key = quoted_mapped.lower()
        if key not in relaxed_seen:
            relaxed_seen.add(key)
            glossary_quoted.append(quoted_mapped)

    strict_query = " AND ".join(quoted_terms)
    if glossary_quoted:
        base_relaxed = relaxed_terms if relaxed_terms else quoted_terms
        relaxed_query = " OR ".join([*base_relaxed, *glossary_quoted])
    elif len(quoted_terms) == 1:
        relaxed_query = None
    elif relaxed_terms:
        relaxed_query = " OR ".join(relaxed_terms)
    else:
        relaxed_query = " OR ".join(quoted_terms)
    return (strict_query, relaxed_query)
