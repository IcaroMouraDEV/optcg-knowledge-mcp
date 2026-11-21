"""Offline PT-BR/EN retrieval evaluation harness for OPTCG hybrid search."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from optcg_knowledge.db.schema import get_meta
from optcg_knowledge.ingest.embeddings import Embedder
from optcg_knowledge.search.hybrid import VALID_DOC_TYPES, hybrid_search

EVAL_SCHEMA_VERSION = "1.0"
DEFAULT_EVAL_DATASET_PATH = Path("data/eval/benchmark_queries.json")

EvalMode = Literal["lexical", "hybrid"]
RequestedEvalMode = Literal["lexical", "hybrid", "both"]

VALID_REQUESTED_MODES = frozenset({"lexical", "hybrid", "both"})
VALID_EVAL_MODES = frozenset({"lexical", "hybrid"})
VALID_LANGS = frozenset({"pt-BR", "en"})
VALID_CATEGORIES = frozenset({"rules", "trm", "cards", "rulings", "mixed"})
VALID_ASSERTION_TYPES = frozenset(
    {"rank_before", "top1_max_authority", "top_k_max_authority", "all_max_authority"}
)

# Privacy guards matching curation rules: never allow Discord mentions, @handles, or tags.
_DISCORD_MENTION_RE = re.compile(r"<@!?\d{15,22}>")
_AT_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_.-]{2,32}\b")
_DISCORD_DISCRIMINATOR_RE = re.compile(r"\b[A-Za-z0-9_.-]{2,32}#\d{4}\b")


class EvalDatasetError(ValueError):
    """Raised when an evaluation benchmark dataset fails schema or privacy validation."""


def _assert_no_pii(text: str | None, *, field_name: str) -> None:
    if not text:
        return
    if _DISCORD_MENTION_RE.search(text):
        raise EvalDatasetError(f"{field_name} contains a raw Discord user mention (<@...>).")
    if _AT_HANDLE_RE.search(text):
        raise EvalDatasetError(f"{field_name} contains an @handle.")
    if _DISCORD_DISCRIMINATOR_RE.search(text):
        raise EvalDatasetError(f"{field_name} contains a Discord username#0000 tag.")


def _normalize_doc_id(doc_type: str, doc_id: str) -> str:
    cleaned = doc_id.strip()
    if doc_type in ("card", "tournament_rule"):
        return cleaned.upper()
    return cleaned


@dataclass(frozen=True, slots=True)
class EvalTarget:
    """Expected relevant document target for a benchmark case."""

    doc_type: str
    doc_id: str
    match_prefix: bool = False

    def matches(self, result_doc_type: str, result_doc_id: str) -> bool:
        """Return True when a search result matches this expected target."""
        if result_doc_type != self.doc_type:
            return False

        expected = _normalize_doc_id(self.doc_type, self.doc_id)
        actual = _normalize_doc_id(result_doc_type, result_doc_id)
        if actual == expected:
            return True
        if not self.match_prefix:
            return False

        if self.doc_type == "rule":
            return actual.startswith(f"{expected}-")
        if self.doc_type == "tournament_rule":
            raw_num = expected[4:] if expected.startswith("TRM-") else expected
            parts = raw_num.split(".")
            if len(parts) == 2 and parts[1] == "0":
                return actual.startswith(f"TRM-{parts[0]}.")
            return actual.startswith(f"{expected}.")
        return actual.startswith(expected)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_type": self.doc_type,
            "doc_id": self.doc_id,
        }
        if self.match_prefix:
            payload["match_prefix"] = True
        return payload


@dataclass(frozen=True, slots=True)
class AuthorityAssertion:
    """Explicit authority-order check evaluated against returned search results."""

    type: str
    higher_doc_id: str | None = None
    lower_doc_id: str | None = None
    require_both: bool = True
    max_authority: int | None = None
    k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.higher_doc_id is not None:
            payload["higher_doc_id"] = self.higher_doc_id
        if self.lower_doc_id is not None:
            payload["lower_doc_id"] = self.lower_doc_id
        if self.type == "rank_before":
            payload["require_both"] = self.require_both
        if self.max_authority is not None:
            payload["max_authority"] = self.max_authority
        if self.k is not None:
            payload["k"] = self.k
        return payload


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Single curated retrieval evaluation query and its ground-truth expectations."""

    case_id: str
    lang: str
    category: str
    query: str
    expected_targets: tuple[EvalTarget, ...]
    doc_types: tuple[str, ...] | None = None
    min_authority: int | None = None
    include_superseded: bool = False
    authority_checks: tuple[AuthorityAssertion, ...] = ()
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """Validated collection of retrieval evaluation cases."""

    schema_version: str
    name: str
    description: str | None
    cases: tuple[EvalCase, ...]
    source_path: Path | None = None


@dataclass(frozen=True, slots=True)
class AssertionOutcome:
    """Pass/fail result of a single authority or safety assertion."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CaseEvalResult:
    """Retrieval metrics and diagnostics for a single case in one search mode."""

    case_id: str
    lang: str
    category: str
    query: str
    search_mode: str
    recall_at_3: float
    recall_at_8: float
    hit_at_3: float
    hit_at_8: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    matched_targets_at_3: int
    matched_targets_at_8: int
    total_targets: int
    top_doc_ids: tuple[str, ...]
    assertions: tuple[AssertionOutcome, ...] = ()

    @property
    def authority_passed(self) -> bool:
        return all(item.passed for item in self.assertions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "lang": self.lang,
            "category": self.category,
            "query": self.query,
            "search_mode": self.search_mode,
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_8": round(self.recall_at_8, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_8": round(self.hit_at_8, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "first_relevant_rank": self.first_relevant_rank,
            "matched_targets_at_3": self.matched_targets_at_3,
            "matched_targets_at_8": self.matched_targets_at_8,
            "total_targets": self.total_targets,
            "top_doc_ids": list(self.top_doc_ids),
            "authority_passed": self.authority_passed,
            "assertions": [item.to_dict() for item in self.assertions],
        }


@dataclass(frozen=True, slots=True)
class MetricSlice:
    """Aggregated retrieval metrics for a mode, language, or category slice."""

    cases_total: int
    recall_at_3: float
    recall_at_8: float
    hit_at_3: float
    hit_at_8: float
    mrr: float
    authority_checks_total: int
    authority_checks_passed: int
    authority_pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases_total": self.cases_total,
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_8": round(self.recall_at_8, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_8": round(self.hit_at_8, 4),
            "mrr": round(self.mrr, 4),
            "authority_checks_total": self.authority_checks_total,
            "authority_checks_passed": self.authority_checks_passed,
            "authority_pass_rate": round(self.authority_pass_rate, 4),
        }


@dataclass(frozen=True, slots=True)
class ModeEvalSummary:
    """Complete evaluation summary for a single mode (lexical or hybrid)."""

    mode: EvalMode
    metrics: MetricSlice
    effective_search_modes: dict[str, int]
    by_lang: dict[str, MetricSlice]
    by_category: dict[str, MetricSlice]
    cases: tuple[CaseEvalResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "effective_search_modes": dict(self.effective_search_modes),
            "metrics": self.metrics.to_dict(),
            "by_lang": {lang: sl.to_dict() for lang, sl in sorted(self.by_lang.items())},
            "by_category": {cat: sl.to_dict() for cat, sl in sorted(self.by_category.items())},
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Structured retrieval evaluation report across one or both search modes."""

    schema_version: str
    dataset_name: str | None
    dataset_path: str | None
    k: int
    requested_mode: RequestedEvalMode
    rules_version: str | None
    embedding_model: str | None
    embedding_dim: int | None
    modes: dict[str, ModeEvalSummary]
    comparison: dict[str, float] | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "k": self.k,
            "requested_mode": self.requested_mode,
            "rules_version": self.rules_version,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "modes": {mode: summary.to_dict() for mode, summary in self.modes.items()},
            "comparison": dict(self.comparison) if self.comparison is not None else None,
            "warnings": list(self.warnings),
        }

    def format_text(self, *, verbose_cases: bool = False) -> str:
        """Render a human-readable benchmark report suitable for CLI output."""
        lines = [
            f"OPTCG Retrieval Benchmark ({self.dataset_name or 'inline'})",
            (
                f"  dataset:         {self.dataset_path or 'in-memory'} "
                f"(k={self.k}, requested_mode={self.requested_mode})"
            ),
            (
                f"  db_context:      rules_version={self.rules_version or 'unknown'}, "
                f"embedding_model={self.embedding_model or 'none'} "
                f"(dim={self.embedding_dim if self.embedding_dim is not None else 'none'})"
            ),
        ]

        for warning in self.warnings:
            lines.append(f"  WARNING:         {warning}")

        for mode_name in ("lexical", "hybrid"):
            summary = self.modes.get(mode_name)
            if summary is None:
                continue
            m = summary.metrics
            effective = ", ".join(
                f"{key}={val}" for key, val in sorted(summary.effective_search_modes.items())
            )
            lines.append("")
            lines.append(f"[{mode_name.upper()}] ({effective})")
            lines.append(
                f"  Overall ({m.cases_total} cases): "
                f"Recall@3={m.recall_at_3:.3f}  Recall@8={m.recall_at_8:.3f}  "
                f"Hit@3={m.hit_at_3:.3f}  Hit@8={m.hit_at_8:.3f}  MRR={m.mrr:.3f}  "
                f"Authority={m.authority_checks_passed}/{m.authority_checks_total} "
                f"({m.authority_pass_rate * 100:.1f}%)"
            )
            for lang, sl in sorted(summary.by_lang.items()):
                lines.append(
                    f"    lang={lang:<5} ({sl.cases_total:>2}): "
                    f"Recall@3={sl.recall_at_3:.3f}  Recall@8={sl.recall_at_8:.3f}  "
                    f"MRR={sl.mrr:.3f}"
                )
            for cat, sl in sorted(summary.by_category.items()):
                lines.append(
                    f"    cat={cat:<7} ({sl.cases_total:>2}): "
                    f"Recall@3={sl.recall_at_3:.3f}  Recall@8={sl.recall_at_8:.3f}  "
                    f"MRR={sl.mrr:.3f}"
                )

            if verbose_cases:
                lines.append("  Cases:")
                for case in summary.cases:
                    status = "OK" if (case.recall_at_8 >= 1.0 and case.authority_passed) else "MISS"
                    rank_label = (
                        f"rank={case.first_relevant_rank}"
                        if case.first_relevant_rank is not None
                        else "rank=-"
                    )
                    lines.append(
                        f"    [{status:<4}] {case.case_id} ({case.lang}/{case.category}): "
                        f"R@3={case.recall_at_3:.2f} R@8={case.recall_at_8:.2f} "
                        f"{rank_label} top3={list(case.top_doc_ids[:3])}"
                    )
                    for assertion in case.assertions:
                        if not assertion.passed:
                            lines.append(
                                f"           FAIL {assertion.name}: {assertion.detail}"
                            )

        if self.comparison is not None:
            lines.append("")
            lines.append("[COMPARISON: hybrid - lexical]")
            lines.append(
                f"  Delta Recall@3={self.comparison['delta_recall_at_3']:+.3f}  "
                f"Delta Recall@8={self.comparison['delta_recall_at_8']:+.3f}  "
                f"Delta MRR={self.comparison['delta_mrr']:+.3f}  "
                f"Delta Authority={self.comparison['delta_authority_pass_rate']:+.3f}"
            )

        return "\n".join(lines)


def _parse_int_authority(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3, 4):
        raise EvalDatasetError(f"{field_name} must be an integer between 1 and 4; got {value!r}")
    return value


def _parse_assertion(raw: Any, *, case_id: str, idx: int) -> AuthorityAssertion:
    prefix = f"case {case_id!r} authority_checks[{idx}]"
    if not isinstance(raw, dict):
        raise EvalDatasetError(f"{prefix} must be an object.")

    assertion_type = raw.get("type")
    if not isinstance(assertion_type, str) or assertion_type not in VALID_ASSERTION_TYPES:
        valid = ", ".join(sorted(VALID_ASSERTION_TYPES))
        raise EvalDatasetError(
            f"{prefix}.type must be one of ({valid}); got {assertion_type!r}"
        )

    if assertion_type == "rank_before":
        higher = raw.get("higher_doc_id")
        lower = raw.get("lower_doc_id")
        if not isinstance(higher, str) or not higher.strip():
            raise EvalDatasetError(f"{prefix}.higher_doc_id must be a non-empty string.")
        if not isinstance(lower, str) or not lower.strip():
            raise EvalDatasetError(f"{prefix}.lower_doc_id must be a non-empty string.")
        require_both = raw.get("require_both", True)
        if not isinstance(require_both, bool):
            raise EvalDatasetError(f"{prefix}.require_both must be a boolean.")
        return AuthorityAssertion(
            type=assertion_type,
            higher_doc_id=higher.strip(),
            lower_doc_id=lower.strip(),
            require_both=require_both,
        )

    max_auth = _parse_int_authority(
        raw.get("max_authority"),
        field_name=f"{prefix}.max_authority",
    )
    if assertion_type == "top_k_max_authority":
        top_k = raw.get("k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise EvalDatasetError(f"{prefix}.k must be a positive integer; got {top_k!r}")
        return AuthorityAssertion(type=assertion_type, max_authority=max_auth, k=top_k)

    return AuthorityAssertion(type=assertion_type, max_authority=max_auth)


def parse_eval_dataset(payload: dict[str, Any], *, source_path: Path | None = None) -> EvalDataset:
    """Validate and parse a benchmark dataset dictionary into an EvalDataset."""
    if not isinstance(payload, dict):
        raise EvalDatasetError("Benchmark dataset root must be a JSON object.")

    schema_version = payload.get("schema_version")
    if schema_version != EVAL_SCHEMA_VERSION:
        raise EvalDatasetError(
            f"Unsupported eval dataset schema_version {schema_version!r}; "
            f"expected {EVAL_SCHEMA_VERSION!r}."
        )

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise EvalDatasetError("Benchmark dataset requires a non-empty string 'name'.")
    _assert_no_pii(name, field_name="dataset.name")

    description = payload.get("description")
    if description is not None:
        if not isinstance(description, str):
            raise EvalDatasetError("dataset.description must be a string when provided.")
        _assert_no_pii(description, field_name="dataset.description")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvalDatasetError("Benchmark dataset requires a non-empty 'cases' array.")

    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for idx, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise EvalDatasetError(f"cases[{idx}] must be an object.")

        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise EvalDatasetError(f"cases[{idx}].case_id must be a non-empty string.")
        case_id = case_id.strip()
        if case_id in seen_ids:
            raise EvalDatasetError(f"Duplicate case_id {case_id!r} in benchmark dataset.")
        seen_ids.add(case_id)

        lang = raw_case.get("lang")
        if not isinstance(lang, str) or lang not in VALID_LANGS:
            raise EvalDatasetError(
                f"case {case_id!r} lang must be one of {sorted(VALID_LANGS)}; got {lang!r}"
            )

        category = raw_case.get("category")
        if not isinstance(category, str) or category not in VALID_CATEGORIES:
            raise EvalDatasetError(
                f"case {case_id!r} category must be one of {sorted(VALID_CATEGORIES)}; "
                f"got {category!r}"
            )

        query = raw_case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise EvalDatasetError(f"case {case_id!r} query must be a non-empty string.")
        _assert_no_pii(query, field_name=f"case {case_id!r} query")

        notes = raw_case.get("notes")
        if notes is not None:
            if not isinstance(notes, str):
                raise EvalDatasetError(f"case {case_id!r} notes must be a string when provided.")
            _assert_no_pii(notes, field_name=f"case {case_id!r} notes")

        raw_targets = raw_case.get("expected_targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise EvalDatasetError(
                f"case {case_id!r} requires a non-empty 'expected_targets' list."
            )

        targets: list[EvalTarget] = []
        for t_idx, raw_target in enumerate(raw_targets):
            if not isinstance(raw_target, dict):
                raise EvalDatasetError(
                    f"case {case_id!r} expected_targets[{t_idx}] must be an object."
                )
            doc_type = raw_target.get("doc_type")
            if not isinstance(doc_type, str) or doc_type not in VALID_DOC_TYPES:
                raise EvalDatasetError(
                    f"case {case_id!r} expected_targets[{t_idx}].doc_type must be one of "
                    f"{sorted(VALID_DOC_TYPES)}; got {doc_type!r}"
                )
            doc_id = raw_target.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise EvalDatasetError(
                    f"case {case_id!r} expected_targets[{t_idx}].doc_id must be a non-empty string."
                )
            match_prefix = raw_target.get("match_prefix", False)
            if not isinstance(match_prefix, bool):
                raise EvalDatasetError(
                    f"case {case_id!r} expected_targets[{t_idx}].match_prefix must be a boolean."
                )
            targets.append(
                EvalTarget(
                    doc_type=doc_type,
                    doc_id=doc_id.strip(),
                    match_prefix=match_prefix,
                )
            )

        raw_doc_types = raw_case.get("doc_types")
        doc_types: tuple[str, ...] | None = None
        if raw_doc_types is not None:
            if not isinstance(raw_doc_types, list) or not raw_doc_types:
                raise EvalDatasetError(f"case {case_id!r} doc_types must be a non-empty list.")
            validated_dts: list[str] = []
            for dt in raw_doc_types:
                if not isinstance(dt, str) or dt not in VALID_DOC_TYPES:
                    raise EvalDatasetError(
                        f"case {case_id!r} invalid doc_type {dt!r}; "
                        f"expected one of {sorted(VALID_DOC_TYPES)}"
                    )
                if dt not in validated_dts:
                    validated_dts.append(dt)
            doc_types = tuple(validated_dts)

        raw_min_auth = raw_case.get("min_authority")
        min_authority = (
            _parse_int_authority(raw_min_auth, field_name=f"case {case_id!r} min_authority")
            if raw_min_auth is not None
            else None
        )

        include_superseded = raw_case.get("include_superseded", False)
        if not isinstance(include_superseded, bool):
            raise EvalDatasetError(f"case {case_id!r} include_superseded must be a boolean.")

        raw_checks = raw_case.get("authority_checks", [])
        if not isinstance(raw_checks, list):
            raise EvalDatasetError(f"case {case_id!r} authority_checks must be a list.")
        checks = tuple(
            _parse_assertion(raw_chk, case_id=case_id, idx=c_idx)
            for c_idx, raw_chk in enumerate(raw_checks)
        )

        cases.append(
            EvalCase(
                case_id=case_id,
                lang=lang,
                category=category,
                query=query.strip(),
                expected_targets=tuple(targets),
                doc_types=doc_types,
                min_authority=min_authority,
                include_superseded=include_superseded,
                authority_checks=checks,
                notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
            )
        )

    return EvalDataset(
        schema_version=EVAL_SCHEMA_VERSION,
        name=name.strip(),
        description=description.strip() if isinstance(description, str) else None,
        cases=tuple(cases),
        source_path=source_path,
    )


def load_eval_dataset(path: Path) -> EvalDataset:
    """Load and validate a retrieval benchmark JSON file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalDatasetError(f"Invalid JSON in {path}: {exc}") from exc
    return parse_eval_dataset(payload, source_path=path)


def _evaluate_assertions(
    case: EvalCase,
    results: Sequence[dict[str, Any]],
) -> tuple[AssertionOutcome, ...]:
    outcomes: list[AssertionOutcome] = []

    # Built-in invariant 1: superseded rulings must never appear unless explicitly requested.
    if not case.include_superseded:
        leaked = [r["doc_id"] for r in results if r.get("superseded_by") is not None]
        outcomes.append(
            AssertionOutcome(
                name="no_superseded",
                passed=len(leaked) == 0,
                detail="ok" if not leaked else f"superseded rulings leaked: {leaked}",
            )
        )

    # Built-in invariant 2: min_authority ceiling must be respected when specified.
    if case.min_authority is not None:
        violations = [
            f"{r['doc_id']}(auth={r['authority']})"
            for r in results
            if int(r["authority"]) > case.min_authority
        ]
        outcomes.append(
            AssertionOutcome(
                name=f"min_authority<={case.min_authority}",
                passed=len(violations) == 0,
                detail="ok" if not violations else f"exceeded min_authority: {violations}",
            )
        )

    # Built-in invariant 3: equal-score ties must preserve authority order (1 < 2 < 3 < 4).
    tie_violation: str | None = None
    for idx in range(len(results) - 1):
        curr = results[idx]
        nxt = results[idx + 1]
        if curr["score"] == nxt["score"] and int(curr["authority"]) > int(nxt["authority"]):
            tie_violation = (
                f"score tie {curr['score']}: {curr['doc_id']}(auth={curr['authority']}) "
                f"ranked before {nxt['doc_id']}(auth={nxt['authority']})"
            )
            break
    outcomes.append(
        AssertionOutcome(
            name="tie_authority_order",
            passed=tie_violation is None,
            detail="ok" if tie_violation is None else tie_violation,
        )
    )

    # Explicit case-level authority checks.
    for assertion in case.authority_checks:
        if assertion.type == "rank_before":
            assert assertion.higher_doc_id is not None and assertion.lower_doc_id is not None
            higher_idx = next(
                (
                    i
                    for i, r in enumerate(results)
                    if r["doc_id"].upper() == assertion.higher_doc_id.upper()
                ),
                None,
            )
            lower_idx = next(
                (
                    i
                    for i, r in enumerate(results)
                    if r["doc_id"].upper() == assertion.lower_doc_id.upper()
                ),
                None,
            )
            name = f"rank_before({assertion.higher_doc_id}<{assertion.lower_doc_id})"
            if higher_idx is not None and lower_idx is not None:
                passed = higher_idx < lower_idx
                detail = (
                    f"{assertion.higher_doc_id}@{higher_idx + 1} vs "
                    f"{assertion.lower_doc_id}@{lower_idx + 1}"
                )
            elif not assertion.require_both and higher_idx is not None:
                passed = True
                detail = (
                    f"{assertion.higher_doc_id}@{higher_idx + 1} "
                    f"({assertion.lower_doc_id} absent)"
                )
            else:
                passed = False
                detail = (
                    f"missing doc(s): {assertion.higher_doc_id}@{higher_idx}, "
                    f"{assertion.lower_doc_id}@{lower_idx}"
                )
            outcomes.append(AssertionOutcome(name=name, passed=passed, detail=detail))

        elif assertion.type == "top1_max_authority":
            assert assertion.max_authority is not None
            name = f"top1_max_authority<={assertion.max_authority}"
            if not results:
                outcomes.append(AssertionOutcome(name=name, passed=False, detail="no results"))
            else:
                top_auth = int(results[0]["authority"])
                passed = top_auth <= assertion.max_authority
                detail = f"top1={results[0]['doc_id']}(auth={top_auth})"
                outcomes.append(AssertionOutcome(name=name, passed=passed, detail=detail))

        elif assertion.type == "top_k_max_authority":
            assert assertion.max_authority is not None and assertion.k is not None
            name = f"top_{assertion.k}_has_authority<={assertion.max_authority}"
            window = results[: assertion.k]
            matching = [
                r["doc_id"] for r in window if int(r["authority"]) <= assertion.max_authority
            ]
            passed = len(matching) > 0
            detail = (
                f"matched={matching[:2]}"
                if passed
                else f"no authority<={assertion.max_authority} in top {assertion.k}"
            )
            outcomes.append(AssertionOutcome(name=name, passed=passed, detail=detail))

        elif assertion.type == "all_max_authority":
            assert assertion.max_authority is not None
            name = f"all_max_authority<={assertion.max_authority}"
            violations = [
                f"{r['doc_id']}(auth={r['authority']})"
                for r in results
                if int(r["authority"]) > assertion.max_authority
            ]
            outcomes.append(
                AssertionOutcome(
                    name=name,
                    passed=len(results) > 0 and len(violations) == 0,
                    detail="ok" if (results and not violations) else f"violations={violations}",
                )
            )

    return tuple(outcomes)


def _evaluate_case(
    conn: sqlite3.Connection,
    case: EvalCase,
    *,
    embedder: Embedder | None,
    k_eval: int,
) -> CaseEvalResult:
    response = hybrid_search(
        conn,
        case.query,
        embedder=embedder,
        k=k_eval,
        doc_types=case.doc_types,
        min_authority=case.min_authority,
        include_superseded=case.include_superseded,
    )
    results: list[dict[str, Any]] = response["results"]

    top3 = results[:3]
    top8 = results[:8]

    matched_at_3 = sum(
        1
        for target in case.expected_targets
        if any(target.matches(str(r["doc_type"]), str(r["doc_id"])) for r in top3)
    )
    matched_at_8 = sum(
        1
        for target in case.expected_targets
        if any(target.matches(str(r["doc_type"]), str(r["doc_id"])) for r in top8)
    )
    total_targets = len(case.expected_targets)

    recall_at_3 = matched_at_3 / total_targets
    recall_at_8 = matched_at_8 / total_targets
    hit_at_3 = 1.0 if matched_at_3 > 0 else 0.0
    hit_at_8 = 1.0 if matched_at_8 > 0 else 0.0

    first_relevant_rank: int | None = None
    for rank, res in enumerate(results, start=1):
        if any(
            target.matches(str(res["doc_type"]), str(res["doc_id"]))
            for target in case.expected_targets
        ):
            first_relevant_rank = rank
            break

    reciprocal_rank = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
    assertions = _evaluate_assertions(case, results)

    return CaseEvalResult(
        case_id=case.case_id,
        lang=case.lang,
        category=case.category,
        query=case.query,
        search_mode=str(response["search_mode"]),
        recall_at_3=recall_at_3,
        recall_at_8=recall_at_8,
        hit_at_3=hit_at_3,
        hit_at_8=hit_at_8,
        reciprocal_rank=reciprocal_rank,
        first_relevant_rank=first_relevant_rank,
        matched_targets_at_3=matched_at_3,
        matched_targets_at_8=matched_at_8,
        total_targets=total_targets,
        top_doc_ids=tuple(str(r["doc_id"]) for r in results[:k_eval]),
        assertions=assertions,
    )


def _aggregate_slice(cases: Sequence[CaseEvalResult]) -> MetricSlice:
    total = len(cases)
    if total == 0:
        return MetricSlice(
            cases_total=0,
            recall_at_3=0.0,
            recall_at_8=0.0,
            hit_at_3=0.0,
            hit_at_8=0.0,
            mrr=0.0,
            authority_checks_total=0,
            authority_checks_passed=0,
            authority_pass_rate=1.0,
        )

    checks_total = sum(len(c.assertions) for c in cases)
    checks_passed = sum(sum(1 for a in c.assertions if a.passed) for c in cases)
    auth_rate = (checks_passed / checks_total) if checks_total > 0 else 1.0

    return MetricSlice(
        cases_total=total,
        recall_at_3=sum(c.recall_at_3 for c in cases) / total,
        recall_at_8=sum(c.recall_at_8 for c in cases) / total,
        hit_at_3=sum(c.hit_at_3 for c in cases) / total,
        hit_at_8=sum(c.hit_at_8 for c in cases) / total,
        mrr=sum(c.reciprocal_rank for c in cases) / total,
        authority_checks_total=checks_total,
        authority_checks_passed=checks_passed,
        authority_pass_rate=auth_rate,
    )


def _summarize_mode(mode: EvalMode, case_results: Sequence[CaseEvalResult]) -> ModeEvalSummary:
    effective_modes: dict[str, int] = {}
    by_lang_groups: dict[str, list[CaseEvalResult]] = {}
    by_cat_groups: dict[str, list[CaseEvalResult]] = {}

    for res in case_results:
        effective_modes[res.search_mode] = effective_modes.get(res.search_mode, 0) + 1
        by_lang_groups.setdefault(res.lang, []).append(res)
        by_cat_groups.setdefault(res.category, []).append(res)

    return ModeEvalSummary(
        mode=mode,
        metrics=_aggregate_slice(case_results),
        effective_search_modes=effective_modes,
        by_lang={lang: _aggregate_slice(items) for lang, items in by_lang_groups.items()},
        by_category={cat: _aggregate_slice(items) for cat, items in by_cat_groups.items()},
        cases=tuple(case_results),
    )


def _check_hybrid_readiness(
    conn: sqlite3.Connection,
    embedder: Embedder | None,
) -> str | None:
    """Return None if hybrid vector search can run against conn, else an explanation string."""
    if embedder is None:
        return "No embedder configured for hybrid mode."

    table_row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_vec'"
    ).fetchone()
    if table_row is None:
        return "Database has no chunks_vec table; run `optcg-ingest embed` first."

    vec_count_row = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()
    vec_count = int(vec_count_row[0]) if vec_count_row is not None else 0
    if vec_count <= 0:
        return "Database has 0 indexed vectors in chunks_vec; run `optcg-ingest embed` first."

    recorded_model = get_meta(conn, "embedding_model")
    if recorded_model is not None and recorded_model != embedder.model_name:
        return (
            f"DB embedding_model={recorded_model!r} does not match configured "
            f"embedder={embedder.model_name!r}; pass --embedding-model {recorded_model} "
            "or re-run `optcg-ingest embed`."
        )

    recorded_dim = get_meta(conn, "embedding_dim")
    if recorded_dim is not None and int(recorded_dim) != embedder.dimension:
        return (
            f"DB embedding_dim={recorded_dim} does not match configured "
            f"embedder.dimension={embedder.dimension}."
        )

    return None


def run_retrieval_eval(
    conn: sqlite3.Connection,
    cases: Sequence[EvalCase],
    *,
    mode: RequestedEvalMode = "both",
    embedder: Embedder | None = None,
    k: int = 8,
    lang: str | None = None,
    dataset_name: str | None = None,
    dataset_path: Path | None = None,
) -> EvalReport:
    """Run retrieval evaluation across lexical and/or hybrid modes on a SQLite DB."""
    if mode not in VALID_REQUESTED_MODES:
        valid = ", ".join(sorted(VALID_REQUESTED_MODES))
        raise ValueError(f"Unsupported eval mode {mode!r}; expected one of: {valid}")
    if isinstance(k, bool) or not isinstance(k, int) or k < 8:
        raise ValueError("k must be an integer >= 8 so Recall@3 and Recall@8 are well-defined.")
    if lang is not None and lang not in VALID_LANGS:
        raise ValueError(f"Unsupported lang filter {lang!r}; expected one of {sorted(VALID_LANGS)}")

    filtered_cases = [c for c in cases if lang is None or c.lang == lang]
    if not filtered_cases:
        raise ValueError("No benchmark cases match the requested filters.")

    rules_version = get_meta(conn, "rules_version")
    recorded_model = get_meta(conn, "embedding_model")
    recorded_dim_raw = get_meta(conn, "embedding_dim")
    recorded_dim = (
        int(recorded_dim_raw) if recorded_dim_raw and recorded_dim_raw.isdigit() else None
    )

    modes_summary: dict[str, ModeEvalSummary] = {}
    warnings: list[str] = []

    if mode in ("lexical", "both"):
        lexical_results = [
            _evaluate_case(conn, case, embedder=None, k_eval=k) for case in filtered_cases
        ]
        modes_summary["lexical"] = _summarize_mode("lexical", lexical_results)

    if mode in ("hybrid", "both"):
        not_ready_reason = _check_hybrid_readiness(conn, embedder)
        if not_ready_reason is not None:
            if mode == "hybrid":
                raise RuntimeError(f"Hybrid evaluation cannot run: {not_ready_reason}")
            warnings.append(f"Skipped hybrid evaluation: {not_ready_reason}")
        else:
            try:
                hybrid_results = [
                    _evaluate_case(conn, case, embedder=embedder, k_eval=k)
                    for case in filtered_cases
                ]
            except (ImportError, RuntimeError) as exc:
                if mode == "hybrid":
                    raise
                warnings.append(f"Skipped hybrid evaluation: {exc}")
            else:
                modes_summary["hybrid"] = _summarize_mode("hybrid", hybrid_results)

    comparison: dict[str, float] | None = None
    if "lexical" in modes_summary and "hybrid" in modes_summary:
        lex = modes_summary["lexical"].metrics
        hyb = modes_summary["hybrid"].metrics
        comparison = {
            "delta_recall_at_3": round(hyb.recall_at_3 - lex.recall_at_3, 4),
            "delta_recall_at_8": round(hyb.recall_at_8 - lex.recall_at_8, 4),
            "delta_hit_at_3": round(hyb.hit_at_3 - lex.hit_at_3, 4),
            "delta_hit_at_8": round(hyb.hit_at_8 - lex.hit_at_8, 4),
            "delta_mrr": round(hyb.mrr - lex.mrr, 4),
            "delta_authority_pass_rate": round(
                hyb.authority_pass_rate - lex.authority_pass_rate, 4
            ),
        }

    return EvalReport(
        schema_version=EVAL_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_path=str(dataset_path) if dataset_path is not None else None,
        k=k,
        requested_mode=mode,
        rules_version=rules_version,
        embedding_model=embedder.model_name if embedder is not None else recorded_model,
        embedding_dim=embedder.dimension if embedder is not None else recorded_dim,
        modes=modes_summary,
        comparison=comparison,
        warnings=tuple(warnings),
    )
