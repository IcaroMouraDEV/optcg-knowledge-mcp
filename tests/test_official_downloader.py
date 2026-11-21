"""Offline tests for the official Bandai rules/FAQ/errata downloader and freshness checker."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.ingest.official_downloader import (
    OfficialDocumentRevision,
    OfficialFetchSummary,
    OfficialFreshnessReport,
    check_official_freshness,
    discover_official_revisions,
    fetch_official_docs,
    load_official_baseline,
)

_FAKE_PDF_V1 = b"%PDF-1.7\n%mock-v1\n%%EOF\n"
_FAKE_PDF_V2 = b"%PDF-1.7\n%mock-v2\n%%EOF\n"
_FAKE_ERRATA_HTML = (
    "<!doctype html><html><head><script>window.analytics=1;</script></head>"
    "<body><h4 class='mediumTit'>November 1, 2025</h4>"
    "<div class='errataModal'><h5>OP01-002 | Trafalgar Law</h5>"
    "<dl><dt>Before</dt><dd>Old text</dd><dt>After</dt><dd>New text</dd></dl></div>"
    "</body></html>"
)
_FAKE_BANLIST_HTML = (
    "<!doctype html><html><head><script>window.token='abc';</script></head>"
    "<body><h3>Cards with Active Restrictions</h3>"
    "<h4>Banned Cards</h4><ul><li>OP06-047 Charlotte Pudding</li></ul>"
    "<img src='/onepiececg/OP06-047.webp' alt='OP06-047'></body></html>"
)


def _build_mock_client(
    *,
    rules_rev: str = "20260828",
    trm_rev: str = "20260116",
    faq_rev: str = "20261030",
    extra_faq_filename: str | None = None,
    pdf_bytes: bytes = _FAKE_PDF_V1,
    errata_html: str = _FAKE_ERRATA_HTML,
    banlist_html: str = _FAKE_BANLIST_HTML,
    request_log: list[str] | None = None,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request_log is not None:
            request_log.append(url)

        if request.url.path == "/rules/":
            html = (
                f'<a href="/pdf/playsheet.pdf?20250929">Playsheet</a>'
                f'<a href="/pdf/rule_comprehensive.pdf?{rules_rev}">Rules</a>'
                f'<a href="/pdf/tournament_rules_manual.pdf?{trm_rev}">TRM</a>'
                f'<a href="https://evil.example/pdf/rule_comprehensive.pdf">Off-host</a>'
                f'<a href="https://evil.example/pdf/tournament_rules_manual.pdf">Off-host TRM</a>'
            )
            return httpx.Response(200, text=html)

        if request.url.path == "/rules/faq/":
            extra_link = (
                f'<a href="/pdf/{extra_faq_filename}?{faq_rev}">Extra QA</a>'
                if extra_faq_filename
                else ""
            )
            html = (
                f'<a href="/pdf/qa_rules.pdf?{faq_rev}">Rules QA</a>'
                f'<a href="/pdf/qa_op05.pdf?{faq_rev}">OP05 QA</a>'
                f'<a href="/pdf/qa_op05.pdf?{faq_rev}">Duplicate</a>'
                f"{extra_link}"
                f'<a href="https://other.example/pdf/ignored.pdf">External</a>'
            )
            return httpx.Response(200, text=html)

        if request.url.path == "/rules/errata_card/":
            return httpx.Response(200, text=errata_html)

        if request.url.path == "/rules/restriction/":
            return httpx.Response(200, text=banlist_html)

        if request.url.path.startswith("/pdf/") and request.url.path.endswith(".pdf"):
            return httpx.Response(200, content=pdf_bytes)

        return httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_official_docs_downloads_skips_and_updates_on_revision(tmp_path: Path) -> None:
    log_first: list[str] = []
    with _build_mock_client(request_log=log_first) as client:
        first = fetch_official_docs(tmp_path, client=client)

    assert first.downloaded == 6
    assert first.skipped == 0
    assert first.rules_files == 1
    assert first.trm_files == 1
    assert first.faq_files == 2
    assert first.errata_files == 1
    assert first.banlist_files == 1
    assert (tmp_path / "rules" / "rule_comprehensive.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "rules" / "tournament_rules_manual.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "qa_rules.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "qa_op05.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "errata_card.html").read_text(encoding="utf-8") == _FAKE_ERRATA_HTML
    assert (
        tmp_path / "banlist" / "restriction.html"
    ).read_text(encoding="utf-8") == _FAKE_BANLIST_HTML
    # Off-host PDF links and card images must never be fetched, and /rules/ is requested once.
    assert not any("evil.example" in u or "other.example" in u for u in log_first)
    assert not any(u.endswith(".webp") for u in log_first)
    assert sum(1 for u in log_first if u.endswith("/rules/")) == 1

    # Second run with identical revisions skips every file without re-downloading PDFs.
    log_second: list[str] = []
    with _build_mock_client(request_log=log_second) as client:
        second = fetch_official_docs(tmp_path, client=client)

    assert second.downloaded == 0
    assert second.skipped == 6
    assert not any("/pdf/" in u for u in log_second)

    # Revision bump on TRM PDF re-downloads ONLY tournament_rules_manual.pdf.
    with _build_mock_client(trm_rev="20261215", pdf_bytes=_FAKE_PDF_V2) as client:
        third = fetch_official_docs(tmp_path, client=client)

    assert third.downloaded == 1
    assert third.skipped == 5
    assert (tmp_path / "rules" / "tournament_rules_manual.pdf").read_bytes() == _FAKE_PDF_V2
    assert (tmp_path / "rules" / "rule_comprehensive.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "qa_op05.pdf").read_bytes() == _FAKE_PDF_V1

    # --force re-downloads all 6 files regardless of manifest.
    with _build_mock_client(trm_rev="20261215", pdf_bytes=_FAKE_PDF_V2) as client:
        forced = fetch_official_docs(tmp_path, force=True, client=client)

    assert forced.downloaded == 6
    assert forced.skipped == 0


def test_fetch_official_docs_only_filter_and_validation(tmp_path: Path) -> None:
    with _build_mock_client() as client:
        rules_only = fetch_official_docs(tmp_path, only="rules", client=client)
    assert rules_only.rules_files == 1
    assert rules_only.trm_files == 0
    assert rules_only.faq_files == 0
    assert rules_only.errata_files == 0
    assert rules_only.banlist_files == 0
    assert (tmp_path / "rules" / "rule_comprehensive.pdf").is_file()
    assert not (tmp_path / "rules" / "tournament_rules_manual.pdf").exists()
    assert not (tmp_path / "faq").exists()

    with _build_mock_client() as client:
        trm_only = fetch_official_docs(tmp_path / "trm_only", only="trm", client=client)
    assert trm_only.rules_files == 0
    assert trm_only.trm_files == 1
    assert trm_only.faq_files == 0
    assert trm_only.errata_files == 0
    assert trm_only.banlist_files == 0
    assert (tmp_path / "trm_only" / "rules" / "tournament_rules_manual.pdf").is_file()
    assert not (tmp_path / "trm_only" / "rules" / "rule_comprehensive.pdf").exists()

    banlist_log: list[str] = []
    with _build_mock_client(request_log=banlist_log) as client:
        banlist_only = fetch_official_docs(tmp_path / "ban_only", only="banlist", client=client)
    assert banlist_only.banlist_files == 1
    assert banlist_only.rules_files == 0
    assert banlist_only.trm_files == 0
    assert (tmp_path / "ban_only" / "banlist" / "restriction.html").is_file()
    assert not any(u.endswith(".webp") for u in banlist_log)

    # Non-PDF response for a PDF link is rejected cleanly.
    with (
        _build_mock_client(pdf_bytes=b"<html>Error</html>") as bad_client,
        pytest.raises(ValueError, match="non-PDF payload"),
    ):
        fetch_official_docs(tmp_path / "bad_pdf", only="rules", client=bad_client)

    # Non-errata response for errata page is rejected cleanly.
    with (
        _build_mock_client(errata_html="<html>Maintenance</html>") as bad_client,
        pytest.raises(ValueError, match="errata HTML"),
    ):
        fetch_official_docs(tmp_path / "bad_errata", only="errata", client=bad_client)

    # Non-banlist response for banlist page is rejected cleanly.
    with (
        _build_mock_client(banlist_html="<html>Maintenance</html>") as bad_client,
        pytest.raises(ValueError, match="banlist/restriction HTML"),
    ):
        fetch_official_docs(tmp_path / "bad_banlist", only="banlist", client=bad_client)


def test_check_official_freshness_zero_pdf_downloads_and_semantic_fingerprints(
    tmp_path: Path,
) -> None:
    baseline_file = tmp_path / "official_revisions.json"

    # Missing baseline without write_baseline raises FileNotFoundError.
    with (
        _build_mock_client() as client,
        pytest.raises(FileNotFoundError, match="Official revisions baseline file not found"),
    ):
        check_official_freshness(baseline_file, client=client)

    # 1. Initial seed (--write-baseline) makes only 4 HTML requests and 0 PDF/image requests.
    seed_log: list[str] = []
    with _build_mock_client(request_log=seed_log) as client:
        initial = check_official_freshness(
            baseline_file,
            write_baseline=True,
            client=client,
        )

    assert initial.baseline_updated is True
    assert initial.baseline_exists is False
    assert initial.up_to_date is False
    assert initial.checked_count == 6
    assert len(initial.added) == 6
    assert baseline_file.is_file()
    assert not any("/pdf/" in url for url in seed_log)
    assert not any(url.endswith(".webp") for url in seed_log)
    assert len(seed_log) == 4

    loaded = load_official_baseline(baseline_file)
    assert set(loaded) == {
        "banlist/restriction.html",
        "faq/errata_card.html",
        "faq/qa_op05.pdf",
        "faq/qa_rules.pdf",
        "rules/rule_comprehensive.pdf",
        "rules/tournament_rules_manual.pdf",
    }
    assert loaded["rules/rule_comprehensive.pdf"].revision == "20260828"
    assert loaded["banlist/restriction.html"].entry_count == 1
    assert loaded["faq/errata_card.html"].entry_count == 1
    assert loaded["faq/errata_card.html"].effective_date == "2025-11-01"

    # 2. Cosmetic HTML script/whitespace change does NOT cause a false positive.
    cosmetic_errata = _FAKE_ERRATA_HTML.replace("window.analytics=1;", "window.analytics=999;")
    cosmetic_banlist = _FAKE_BANLIST_HTML.replace("window.token='abc';", "window.token='xyz';")
    check_log: list[str] = []
    with _build_mock_client(
        errata_html=cosmetic_errata,
        banlist_html=cosmetic_banlist,
        request_log=check_log,
    ) as client:
        second = check_official_freshness(baseline_file, client=client)

    assert second.up_to_date is True
    assert second.unchanged_count == 6
    assert second.added == ()
    assert second.updated == ()
    assert second.removed == ()
    assert not any("/pdf/" in url for url in check_log)
    assert "**Status:** Up to date" in second.to_markdown()

    # 3. Detecting PDF revision bump, added FAQ PDF, and semantic banlist update.
    updated_banlist = (
        "<!doctype html><html><body><h3>Cards with Active Restrictions</h3>"
        "<h4>Banned Cards</h4><ul><li>OP06-047 Charlotte Pudding</li>"
        "<li>OP02-024 Moby Dick</li></ul></body></html>"
    )
    with _build_mock_client(
        trm_rev="20261215",
        extra_faq_filename="qa_op14.pdf",
        banlist_html=updated_banlist,
    ) as client:
        third = check_official_freshness(baseline_file, client=client)

    assert third.up_to_date is False
    assert third.checked_count == 7
    assert third.unchanged_count == 4
    assert [d.key for d in third.added] == ["faq/qa_op14.pdf"]
    assert [d.key for d in third.updated] == [
        "banlist/restriction.html",
        "rules/tournament_rules_manual.pdf",
    ]
    assert third.removed == ()
    md = third.to_markdown()
    assert "**Status:** Updates detected" in md
    assert "`rules/tournament_rules_manual.pdf`" in md
    assert "`20260116`" in md
    assert "`20261215`" in md
    assert "`faq/qa_op14.pdf`" in md

    # 4. Scoped check (--only trm) compares and updates only the TRM entry in baseline.
    with _build_mock_client(trm_rev="20261215") as client:
        scoped = check_official_freshness(
            baseline_file,
            only="trm",
            write_baseline=True,
            client=client,
        )
    assert scoped.checked_count == 1
    assert [d.key for d in scoped.updated] == ["rules/tournament_rules_manual.pdf"]
    assert scoped.removed == ()
    reloaded = load_official_baseline(baseline_file)
    assert reloaded["rules/tournament_rules_manual.pdf"].revision == "20261215"
    assert reloaded["rules/rule_comprehensive.pdf"].revision == "20260828"


def test_discover_official_revisions_retries_transient_502(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("optcg_knowledge.ingest.official_downloader.time.sleep", lambda _: None)
    attempts = {"count": 0}

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rules/faq/":
            attempts["count"] += 1
            if attempts["count"] == 1:
                return httpx.Response(502, text="Bad Gateway")
            return httpx.Response(
                200,
                text='<a href="/pdf/qa_rules.pdf?20261030">Rules QA</a>',
            )
        return httpx.Response(404, text="not found")

    with httpx.Client(transport=httpx.MockTransport(flaky_handler)) as client:
        docs = discover_official_revisions(only="faq", client=client)

    assert attempts["count"] == 2
    assert len(docs) == 1
    assert docs[0].key == "faq/qa_rules.pdf"
    assert docs[0].revision == "20261030"


def test_cli_fetch_official_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_fetch(
        out_dir: Path,
        *,
        only: str,
        base_url: str,
        force: bool,
    ) -> OfficialFetchSummary:
        captured.update(out_dir=out_dir, only=only, base_url=base_url, force=force)
        return OfficialFetchSummary(
            downloaded=5,
            skipped=1,
            rules_files=1,
            trm_files=1,
            faq_files=2,
            errata_files=1,
            out_dir=out_dir,
            banlist_files=1,
        )

    monkeypatch.setattr("optcg_knowledge.cli.fetch_official_docs", fake_fetch)

    exit_code = cli_main(
        ["fetch-official", "--out-dir", str(tmp_path), "--only", "all", "--force"]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Fetched official Bandai docs" in stdout
    assert "downloaded=5, skipped=1" in stdout
    assert "trm=1" in stdout
    assert "banlist=1" in stdout
    assert captured["out_dir"] == tmp_path
    assert captured["only"] == "all"
    assert captured["force"] is True


def test_cli_check_official_command_outputs_and_fail_on_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "official_revisions.json"
    json_out = tmp_path / "out" / "report.json"
    md_out = tmp_path / "out" / "report.md"

    doc_prev = OfficialDocumentRevision(
        key="rules/rule_comprehensive.pdf",
        category="rules",
        url="https://en.onepiece-cardgame.com/pdf/rule_comprehensive.pdf?20260828",
        revision="20260828",
    )
    doc_curr = OfficialDocumentRevision(
        key="rules/rule_comprehensive.pdf",
        category="rules",
        url="https://en.onepiece-cardgame.com/pdf/rule_comprehensive.pdf?20261121",
        revision="20261121",
    )

    from optcg_knowledge.ingest.official_downloader import OfficialDocumentDiff

    fake_report = OfficialFreshnessReport(
        up_to_date=False,
        only="rules",
        base_url="https://en.onepiece-cardgame.com",
        baseline_path=baseline_path,
        baseline_exists=True,
        baseline_updated=False,
        checked_count=1,
        unchanged_count=0,
        added=(),
        updated=(
            OfficialDocumentDiff(
                key="rules/rule_comprehensive.pdf",
                category="rules",
                change_type="updated",
                previous=doc_prev,
                current=doc_curr,
            ),
        ),
        removed=(),
        documents=(doc_curr,),
    )

    def fake_check(
        baseline: Path,
        *,
        only: str,
        base_url: str,
        write_baseline: bool,
    ) -> OfficialFreshnessReport:
        assert baseline == baseline_path
        assert only == "rules"
        assert base_url == "https://en.onepiece-cardgame.com"
        assert write_baseline is False
        return fake_report

    monkeypatch.setattr("optcg_knowledge.cli.check_official_freshness", fake_check)

    exit_code = cli_main(
        [
            "check-official",
            "--baseline",
            str(baseline_path),
            "--only",
            "rules",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
            "--fail-on-diff",
        ]
    )
    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "Official Bandai docs check: updates-detected" in stdout
    assert "[UPDATED] rules/rule_comprehensive.pdf: 20260828 -> 20261121" in stdout
    assert json.loads(json_out.read_text(encoding="utf-8"))["up_to_date"] is False
    assert "## Official Bandai Document Freshness Check" in md_out.read_text(encoding="utf-8")

    # Without --fail-on-diff and with --json, exit code is 0 and stdout is valid JSON.
    exit_code_json = cli_main(
        [
            "check-official",
            "--baseline",
            str(baseline_path),
            "--only",
            "rules",
            "--json",
        ]
    )
    assert exit_code_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["up_to_date"] is False
    assert payload["updated_count"] == 1
