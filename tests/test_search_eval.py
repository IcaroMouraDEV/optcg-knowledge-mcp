"""Offline tests for the PT-BR/EN retrieval evaluation benchmark harness."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.ingest import HashEmbedder
from optcg_knowledge.search.eval import (
    DEFAULT_EVAL_DATASET_PATH,
    EVAL_SCHEMA_VERSION,
    AuthorityAssertion,
    EvalCase,
    EvalDatasetError,
    EvalTarget,
    load_eval_dataset,
    parse_eval_dataset,
    run_retrieval_eval,
)


def _build_fixture_cases() -> list[EvalCase]:
    """Small bilingual case set aligned with the offline populated_db fixture."""
    return [
        EvalCase(
            case_id="fixture-card-op05-119-pt",
            lang="pt-BR",
            category="cards",
            query="Qual o efeito da carta OP05-119 e o ruling de extra turn?",
            expected_targets=(
                EvalTarget(doc_type="card", doc_id="OP05-119"),
                EvalTarget(doc_type="ruling", doc_id="faq:op05:119-1"),
            ),
            authority_checks=(
                AuthorityAssertion(
                    type="rank_before",
                    higher_doc_id="faq:op05:119-1",
                    lower_doc_id="judge:2025-09-10:001",
                    require_both=True,
                ),
            ),
        ),
        EvalCase(
            case_id="fixture-rule-damage-step-en",
            lang="en",
            category="rules",
            query="What does rule 7-1-4-1 say about Damage Step when Leader has 0 Life?",
            expected_targets=(
                EvalTarget(doc_type="rule", doc_id="7-1-4-1", match_prefix=True),
            ),
            authority_checks=(
                AuthorityAssertion(type="top1_max_authority", max_authority=1),
            ),
        ),
        EvalCase(
            case_id="fixture-blocker-keyword-pt",
            lang="pt-BR",
            category="rules",
            query="Como funciona [Blocker] durante a Block Step?",
            expected_targets=(
                EvalTarget(doc_type="rule", doc_id="10-1-1"),
                EvalTarget(doc_type="rule", doc_id="7-1-2-1"),
            ),
            authority_checks=(
                AuthorityAssertion(type="top_k_max_authority", k=3, max_authority=1),
            ),
        ),
        EvalCase(
            case_id="fixture-authority-tie-en",
            lang="en",
            category="rulings",
            query="[On K.O.] timing resolve trashed by battle",
            doc_types=("ruling",),
            expected_targets=(
                EvalTarget(doc_type="ruling", doc_id="faq:on-ko:tie"),
            ),
            authority_checks=(
                AuthorityAssertion(
                    type="rank_before",
                    higher_doc_id="faq:on-ko:tie",
                    lower_doc_id="community:on-ko:tie",
                    require_both=True,
                ),
            ),
        ),
        EvalCase(
            case_id="fixture-official-only-en",
            lang="en",
            category="mixed",
            query="OP05-119 extra turn",
            min_authority=2,
            expected_targets=(
                EvalTarget(doc_type="card", doc_id="OP05-119"),
                EvalTarget(doc_type="ruling", doc_id="faq:op05:119-1"),
            ),
            authority_checks=(
                AuthorityAssertion(type="all_max_authority", max_authority=2),
            ),
        ),
    ]


def test_tracked_benchmark_dataset_is_valid_bilingual_and_pii_free() -> None:
    dataset = load_eval_dataset(DEFAULT_EVAL_DATASET_PATH)

    assert dataset.schema_version == EVAL_SCHEMA_VERSION
    assert 40 <= len(dataset.cases) <= 60

    langs = {case.lang for case in dataset.cases}
    categories = {case.category for case in dataset.cases}
    assert langs == {"pt-BR", "en"}
    assert {"rules", "trm", "cards", "mixed"}.issubset(categories)

    pt_count = sum(1 for c in dataset.cases if c.lang == "pt-BR")
    en_count = sum(1 for c in dataset.cases if c.lang == "en")
    assert pt_count >= 25
    assert en_count >= 15

    pure_pt_cases = [c for c in dataset.cases if "pure-pt-br" in c.tags]
    assert len(pure_pt_cases) == 15
    assert {c.lang for c in pure_pt_cases} == {"pt-BR"}
    assert {"rules", "trm"}.issubset({c.category for c in pure_pt_cases})

    # Guardrail: pure-pt-br slice must not rely on bracketed keywords or English OPTCG anglicisms.
    forbidden_en_terms = re.compile(
        r"\[[^\]]+\]|\b("
        r"leader|life|character|don!!|counter step|block step|damage step|draw phase|"
        r"main phase|slow play|intentional draw|head judge|trash|trigger|banish|rush|"
        r"double attack|blocker|replacement|instead|loop|tournament rules manual"
        r")\b",
        re.IGNORECASE,
    )
    for case in pure_pt_cases:
        assert case.authority_checks, f"{case.case_id} missing authority_checks"
        assert not forbidden_en_terms.search(case.query), (
            f"{case.case_id} contains forbidden English OPTCG term: {case.query!r}"
        )


def test_parse_eval_dataset_validates_and_normalizes_tags() -> None:
    parsed = parse_eval_dataset(
        {
            "schema_version": "1.0",
            "name": "tag-test",
            "cases": [
                {
                    "case_id": "t1",
                    "lang": "pt-BR",
                    "category": "rules",
                    "tags": [" Pure-PT-BR ", "pure-pt-br", "core_rules"],
                    "query": "Como funciona a regra de bloqueio?",
                    "expected_targets": [{"doc_type": "rule", "doc_id": "10-1-4"}],
                }
            ],
        }
    )
    assert parsed.cases[0].tags == ("pure-pt-br", "core_rules")

    with pytest.raises(EvalDatasetError, match="lowercase slug"):
        parse_eval_dataset(
            {
                "schema_version": "1.0",
                "name": "bad-tag",
                "cases": [
                    {
                        "case_id": "t-bad",
                        "lang": "pt-BR",
                        "category": "rules",
                        "tags": ["bad tag!"],
                        "query": "Pergunta teste",
                        "expected_targets": [{"doc_type": "rule", "doc_id": "10-1-4"}],
                    }
                ],
            }
        )


def test_eval_target_prefix_matching_rules_trm_and_rulings() -> None:
    rule_prefix = EvalTarget(doc_type="rule", doc_id="7-1-4", match_prefix=True)
    assert rule_prefix.matches("rule", "7-1-4")
    assert rule_prefix.matches("rule", "7-1-4-1")
    assert not rule_prefix.matches("rule", "7-1-40")
    assert not rule_prefix.matches("ruling", "7-1-4")

    trm_section = EvalTarget(doc_type="tournament_rule", doc_id="TRM-7.0", match_prefix=True)
    assert trm_section.matches("tournament_rule", "TRM-7.0")
    assert trm_section.matches("tournament_rule", "trm-7.4.2.3")
    assert not trm_section.matches("tournament_rule", "TRM-8.0")

    faq_prefix = EvalTarget(doc_type="ruling", doc_id="faq:OP04-083:", match_prefix=True)
    assert faq_prefix.matches("ruling", "faq:OP04-083:2bedb9b99d14")
    assert not faq_prefix.matches("ruling", "faq:OP05-119:abcd1234")


def test_parse_eval_dataset_rejects_pii_and_bad_schema() -> None:
    with pytest.raises(EvalDatasetError, match="@handle"):
        parse_eval_dataset(
            {
                "schema_version": "1.0",
                "name": "bad-pii",
                "cases": [
                    {
                        "case_id": "pii-1",
                        "lang": "en",
                        "category": "rules",
                        "query": "Ask @somejudge about [Blocker]",
                        "expected_targets": [{"doc_type": "rule", "doc_id": "10-1-1"}],
                    }
                ],
            }
        )

    with pytest.raises(EvalDatasetError, match="Duplicate case_id"):
        parse_eval_dataset(
            {
                "schema_version": "1.0",
                "name": "dup-id",
                "cases": [
                    {
                        "case_id": "dup",
                        "lang": "en",
                        "category": "rules",
                        "query": "[Blocker]",
                        "expected_targets": [{"doc_type": "rule", "doc_id": "10-1-1"}],
                    },
                    {
                        "case_id": "dup",
                        "lang": "pt-BR",
                        "category": "rules",
                        "query": "[Blocker] regra",
                        "expected_targets": [{"doc_type": "rule", "doc_id": "10-1-1"}],
                    },
                ],
            }
        )


def test_run_retrieval_eval_smoke_both_modes_with_hash_embedder(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, embedder = populated_db
    cases = _build_fixture_cases()

    report = run_retrieval_eval(
        conn,
        cases,
        mode="both",
        embedder=embedder,
        k=8,
        dataset_name="offline-fixture-smoke",
    )

    assert report.schema_version == EVAL_SCHEMA_VERSION
    assert not report.warnings
    assert set(report.modes.keys()) == {"lexical", "hybrid"}
    assert report.comparison is not None

    lexical = report.modes["lexical"]
    hybrid = report.modes["hybrid"]
    assert lexical.effective_search_modes == {"lexical_only": len(cases)}
    assert hybrid.effective_search_modes == {"hybrid": len(cases)}

    for summary in (lexical, hybrid):
        assert summary.metrics.cases_total == len(cases)
        assert summary.metrics.recall_at_8 >= 0.8
        assert summary.metrics.mrr >= 0.8
        assert summary.metrics.authority_checks_total > 0
        assert summary.metrics.authority_checks_passed == summary.metrics.authority_checks_total
        assert set(summary.by_lang.keys()) == {"pt-BR", "en"}

    # Tie-break authority case must rank official_faq above community in both modes.
    for summary in (lexical, hybrid):
        tie_case = next(c for c in summary.cases if c.case_id == "fixture-authority-tie-en")
        assert tie_case.authority_passed
        assert tie_case.first_relevant_rank == 1

    # Superseded rulings stay hidden by default.
    for summary in (lexical, hybrid):
        for case in summary.cases:
            assert "judge:2023-12-01:old" not in case.top_doc_ids

    rendered = report.format_text(verbose_cases=True)
    assert "[LEXICAL]" in rendered
    assert "[HYBRID]" in rendered
    assert "[COMPARISON: hybrid - lexical]" in rendered


def test_run_retrieval_eval_both_degrades_gracefully_without_embedder_and_hybrid_fails(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
) -> None:
    _, conn, _ = populated_db
    cases = _build_fixture_cases()[:2]

    report = run_retrieval_eval(conn, cases, mode="both", embedder=None, k=8)
    assert list(report.modes.keys()) == ["lexical"]
    assert report.comparison is None
    assert len(report.warnings) == 1
    assert "Skipped hybrid evaluation" in report.warnings[0]

    with pytest.raises(RuntimeError, match="Hybrid evaluation cannot run"):
        run_retrieval_eval(conn, cases, mode="hybrid", embedder=None, k=8)


def test_cli_eval_lexical_json_and_threshold_gate(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, _, _ = populated_db
    dataset_path = tmp_path / "mini_eval.json"
    out_path = tmp_path / "eval_out.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "cli-mini-eval",
                "cases": [
                    {
                        "case_id": "cli-card-1",
                        "lang": "en",
                        "category": "cards",
                        "query": "OP05-119",
                        "expected_targets": [{"doc_type": "card", "doc_id": "OP05-119"}],
                    },
                    {
                        "case_id": "cli-rule-1",
                        "lang": "pt-BR",
                        "category": "rules",
                        "query": "Regra 7-1-4-1 Damage Step",
                        "expected_targets": [{"doc_type": "rule", "doc_id": "7-1-4-1"}],
                        "authority_checks": [
                            {"type": "top1_max_authority", "max_authority": 1}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = cli_main(
        [
            "--db",
            str(db_path),
            "eval",
            "--dataset",
            str(dataset_path),
            "--mode",
            "lexical",
            "--json",
            "--out",
            str(out_path),
            "--min-recall3",
            "0.9",
            "--min-mrr",
            "0.9",
            "--fail-on-authority",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["dataset_name"] == "cli-mini-eval"
    assert payload["modes"]["lexical"]["metrics"]["recall_at_3"] == 1.0
    assert payload["modes"]["lexical"]["metrics"]["mrr"] == 1.0
    assert out_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == payload

    # Impossible threshold should trigger exit code 1 with clear stderr message.
    rc_fail = cli_main(
        [
            "--db",
            str(db_path),
            "eval",
            "--dataset",
            str(dataset_path),
            "--mode",
            "lexical",
            "--min-recall3",
            "1.1",
        ]
    )
    assert rc_fail == 1
    captured_fail = capsys.readouterr()
    assert "Gate failed" in captured_fail.err


def test_eval_by_tag_aggregation_and_cli_tag_filter(
    populated_db: tuple[Path, sqlite3.Connection, HashEmbedder],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path, conn, _ = populated_db
    tagged_cases = [
        EvalCase(
            case_id="tagged-card-en",
            lang="en",
            category="cards",
            query="OP05-119",
            expected_targets=(EvalTarget(doc_type="card", doc_id="OP05-119"),),
            tags=("card-exact",),
        ),
        EvalCase(
            case_id="tagged-rule-pt",
            lang="pt-BR",
            category="rules",
            query="Regra 7-1-4-1 Damage Step",
            expected_targets=(EvalTarget(doc_type="rule", doc_id="7-1-4-1"),),
            tags=("pure-pt-br",),
        ),
    ]

    full_report = run_retrieval_eval(conn, tagged_cases, mode="lexical", k=8)
    lex_summary = full_report.modes["lexical"]
    assert set(lex_summary.by_tag.keys()) == {"card-exact", "pure-pt-br"}
    assert lex_summary.by_tag["pure-pt-br"].cases_total == 1
    assert "tag=pure-pt-br" in full_report.format_text()

    filtered_report = run_retrieval_eval(
        conn, tagged_cases, mode="lexical", k=8, tag="pure-pt-br"
    )
    assert filtered_report.modes["lexical"].metrics.cases_total == 1
    assert filtered_report.modes["lexical"].cases[0].case_id == "tagged-rule-pt"

    with pytest.raises(ValueError, match="Unsupported tag filter"):
        run_retrieval_eval(conn, tagged_cases, mode="lexical", k=8, tag="invalid tag!")

    dataset_path = tmp_path / "tagged_eval.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "cli-tagged-eval",
                "cases": [
                    {
                        "case_id": "cli-t1",
                        "lang": "en",
                        "category": "cards",
                        "tags": ["card-exact"],
                        "query": "OP05-119",
                        "expected_targets": [{"doc_type": "card", "doc_id": "OP05-119"}],
                    },
                    {
                        "case_id": "cli-t2",
                        "lang": "pt-BR",
                        "category": "rules",
                        "tags": ["pure-pt-br"],
                        "query": "Regra 7-1-4-1 Damage Step",
                        "expected_targets": [{"doc_type": "rule", "doc_id": "7-1-4-1"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = cli_main(
        [
            "--db",
            str(db_path),
            "eval",
            "--dataset",
            str(dataset_path),
            "--mode",
            "lexical",
            "--tag",
            "pure-pt-br",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["modes"]["lexical"]["metrics"]["cases_total"] == 1
    assert list(payload["modes"]["lexical"]["by_tag"].keys()) == ["pure-pt-br"]

    rc_missing = cli_main(
        [
            "--db",
            str(db_path),
            "eval",
            "--dataset",
            str(dataset_path),
            "--mode",
            "lexical",
            "--tag",
            "nonexistent-tag",
        ]
    )
    assert rc_missing == 1
    assert "No benchmark cases match" in capsys.readouterr().err
