"""Offline tests for the official Bandai rules/FAQ/errata downloader."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from optcg_knowledge.cli import main as cli_main
from optcg_knowledge.ingest.official_downloader import OfficialFetchSummary, fetch_official_docs

_FAKE_PDF_V1 = b"%PDF-1.7\n%mock-v1\n%%EOF\n"
_FAKE_PDF_V2 = b"%PDF-1.7\n%mock-v2\n%%EOF\n"
_FAKE_ERRATA_HTML = (
    "<!doctype html><html><body><div class='errataModal'>OP01-002</div></body></html>"
)


def _build_mock_client(
    *,
    rules_rev: str = "20260828",
    trm_rev: str = "20260116",
    faq_rev: str = "20261030",
    pdf_bytes: bytes = _FAKE_PDF_V1,
    errata_html: str = _FAKE_ERRATA_HTML,
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
            html = (
                f'<a href="/pdf/qa_rules.pdf?{faq_rev}">Rules QA</a>'
                f'<a href="/pdf/qa_op05.pdf?{faq_rev}">OP05 QA</a>'
                f'<a href="/pdf/qa_op05.pdf?{faq_rev}">Duplicate</a>'
                f'<a href="https://other.example/pdf/ignored.pdf">External</a>'
            )
            return httpx.Response(200, text=html)

        if request.url.path == "/rules/errata_card/":
            return httpx.Response(200, text=errata_html)

        if request.url.path.startswith("/pdf/") and request.url.path.endswith(".pdf"):
            return httpx.Response(200, content=pdf_bytes)

        return httpx.Response(404, text="not found")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_official_docs_downloads_skips_and_updates_on_revision(tmp_path: Path) -> None:
    log_first: list[str] = []
    with _build_mock_client(request_log=log_first) as client:
        first = fetch_official_docs(tmp_path, client=client)

    assert first.downloaded == 5
    assert first.skipped == 0
    assert first.rules_files == 1
    assert first.trm_files == 1
    assert first.faq_files == 2
    assert first.errata_files == 1
    assert (tmp_path / "rules" / "rule_comprehensive.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "rules" / "tournament_rules_manual.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "qa_rules.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "qa_op05.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "errata_card.html").read_text(encoding="utf-8") == _FAKE_ERRATA_HTML
    # Off-host PDF links must never be fetched, and /rules/ is requested only once.
    assert not any("evil.example" in u or "other.example" in u for u in log_first)
    assert sum(1 for u in log_first if u.endswith("/rules/")) == 1

    # Second run with identical revisions skips every file without re-downloading PDFs.
    log_second: list[str] = []
    with _build_mock_client(request_log=log_second) as client:
        second = fetch_official_docs(tmp_path, client=client)

    assert second.downloaded == 0
    assert second.skipped == 5
    assert not any("/pdf/" in u for u in log_second)

    # Revision bump on TRM PDF re-downloads ONLY tournament_rules_manual.pdf.
    with _build_mock_client(trm_rev="20261215", pdf_bytes=_FAKE_PDF_V2) as client:
        third = fetch_official_docs(tmp_path, client=client)

    assert third.downloaded == 1
    assert third.skipped == 4
    assert (tmp_path / "rules" / "tournament_rules_manual.pdf").read_bytes() == _FAKE_PDF_V2
    assert (tmp_path / "rules" / "rule_comprehensive.pdf").read_bytes() == _FAKE_PDF_V1
    assert (tmp_path / "faq" / "qa_op05.pdf").read_bytes() == _FAKE_PDF_V1

    # --force re-downloads all 5 files regardless of manifest.
    with _build_mock_client(trm_rev="20261215", pdf_bytes=_FAKE_PDF_V2) as client:
        forced = fetch_official_docs(tmp_path, force=True, client=client)

    assert forced.downloaded == 5
    assert forced.skipped == 0


def test_fetch_official_docs_only_filter_and_validation(tmp_path: Path) -> None:
    with _build_mock_client() as client:
        rules_only = fetch_official_docs(tmp_path, only="rules", client=client)
    assert rules_only.rules_files == 1
    assert rules_only.trm_files == 0
    assert rules_only.faq_files == 0
    assert rules_only.errata_files == 0
    assert (tmp_path / "rules" / "rule_comprehensive.pdf").is_file()
    assert not (tmp_path / "rules" / "tournament_rules_manual.pdf").exists()
    assert not (tmp_path / "faq").exists()

    with _build_mock_client() as client:
        trm_only = fetch_official_docs(tmp_path / "trm_only", only="trm", client=client)
    assert trm_only.rules_files == 0
    assert trm_only.trm_files == 1
    assert trm_only.faq_files == 0
    assert trm_only.errata_files == 0
    assert (tmp_path / "trm_only" / "rules" / "tournament_rules_manual.pdf").is_file()
    assert not (tmp_path / "trm_only" / "rules" / "rule_comprehensive.pdf").exists()

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
            downloaded=4,
            skipped=1,
            rules_files=1,
            trm_files=1,
            faq_files=2,
            errata_files=1,
            out_dir=out_dir,
        )

    monkeypatch.setattr("optcg_knowledge.cli.fetch_official_docs", fake_fetch)

    exit_code = cli_main(
        ["fetch-official", "--out-dir", str(tmp_path), "--only", "all", "--force"]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Fetched official Bandai docs" in stdout
    assert "downloaded=4, skipped=1" in stdout
    assert "trm=1" in stdout
    assert captured["out_dir"] == tmp_path
    assert captured["only"] == "all"
    assert captured["force"] is True
