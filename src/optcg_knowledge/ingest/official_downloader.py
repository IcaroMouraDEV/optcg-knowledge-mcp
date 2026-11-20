"""Downloader for official Bandai OPTCG Comprehensive Rules, FAQ PDFs, and Errata HTML.

Never downloads card images—only official rules PDFs, FAQ PDFs, and the text-only
errata HTML page into the local gitignored ``data/raw/`` directory.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_OFFICIAL_BASE_URL = "https://en.onepiece-cardgame.com"
DEFAULT_USER_AGENT = "optcg-knowledge/0.1 (+https://github.com/IcaroMouraDEV/optcg-knowledge)"

FetchTarget = Literal["all", "rules", "faq", "errata"]

_PDF_HREF_RE = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
_MANIFEST_FILENAME = ".official_manifest.json"


@dataclass(frozen=True, slots=True)
class OfficialFetchSummary:
    """Counters returned after fetching official Bandai rules, FAQs, and errata."""

    downloaded: int
    skipped: int
    rules_files: int
    faq_files: int
    errata_files: int
    out_dir: Path


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.is_file():
        return {}
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _save_manifest(manifest_path: Path, manifest: dict[str, str]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        f"{json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _extract_same_host_pdf_links(html: str, *, page_url: str, allowed_host: str) -> list[str]:
    """Extract unique same-host PDF URLs from an HTML page in deterministic order."""
    seen: set[str] = set()
    urls: list[str] = []
    for raw_href in _PDF_HREF_RE.findall(html):
        resolved = urljoin(page_url, raw_href.strip())
        parsed = urlparse(resolved)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc.lower() != allowed_host.lower():
            continue
        if resolved not in seen:
            seen.add(resolved)
            urls.append(resolved)
    return sorted(urls)


def _safe_filename_from_url(url: str) -> str:
    path_part = urlparse(url).path
    name = Path(path_part).name
    if not name or name in (".", ".."):
        raise ValueError(f"Could not derive safe filename from URL: {url!r}")
    return name


def _download_pdf_if_needed(
    client: httpx.Client,
    *,
    url: str,
    dest: Path,
    manifest_key: str,
    manifest: dict[str, str],
    force: bool,
) -> bool:
    """Download a PDF into ``dest`` unless already up-to-date in ``manifest``.

    Returns True when a new file was downloaded, False when skipped.
    """
    if not force and dest.is_file() and manifest.get(manifest_key) == url:
        return False

    resp = client.get(url)
    resp.raise_for_status()
    data = resp.content
    if not data.startswith(b"%PDF"):
        raise ValueError(f"Expected PDF response from {url}, received non-PDF payload")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    manifest[manifest_key] = url
    return True


def fetch_official_docs(
    out_dir: Path | str,
    *,
    only: FetchTarget = "all",
    base_url: str = DEFAULT_OFFICIAL_BASE_URL,
    force: bool = False,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> OfficialFetchSummary:
    """Fetch official Bandai Comprehensive Rules PDF, FAQ PDFs, and Errata HTML.

    Files are saved into ``<out_dir>/rules/`` and ``<out_dir>/faq/`` with revision
    tracking in ``<out_dir>/.official_manifest.json`` so re-runs skip unchanged files.
    """
    if only not in ("all", "rules", "faq", "errata"):
        raise ValueError(f"Unsupported target {only!r}; expected one of: all, rules, faq, errata")

    target_root = Path(out_dir)
    rules_dir = target_root / "rules"
    faq_dir = target_root / "faq"
    manifest_path = target_root / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    normalized_base = base_url.rstrip("/")
    allowed_host = urlparse(normalized_base).netloc

    downloaded = 0
    skipped = 0
    rules_files = 0
    faq_files = 0
    errata_files = 0

    owns_client = client is None
    active_client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )

    try:
        if only in ("all", "rules"):
            rules_page_url = f"{normalized_base}/rules/"
            logger.info("Checking Comprehensive Rules link at %s", rules_page_url)
            resp = active_client.get(rules_page_url)
            resp.raise_for_status()
            pdf_urls = _extract_same_host_pdf_links(
                resp.text,
                page_url=rules_page_url,
                allowed_host=allowed_host,
            )
            comp_urls = [
                u
                for u in pdf_urls
                if "rule_comprehensive" in Path(urlparse(u).path).name.lower()
            ]
            if not comp_urls:
                raise RuntimeError(
                    f"Could not find rule_comprehensive*.pdf link on {rules_page_url}"
                )
            comp_url = comp_urls[0]
            dest = rules_dir / "rule_comprehensive.pdf"
            if _download_pdf_if_needed(
                active_client,
                url=comp_url,
                dest=dest,
                manifest_key="rules/rule_comprehensive.pdf",
                manifest=manifest,
                force=force,
            ):
                downloaded += 1
                logger.info("Downloaded %s", dest)
            else:
                skipped += 1
                logger.info("Up-to-date (skipped): %s", dest)
            rules_files = 1

        if only in ("all", "faq"):
            faq_page_url = f"{normalized_base}/rules/faq/"
            logger.info("Checking official FAQ PDF links at %s", faq_page_url)
            resp = active_client.get(faq_page_url)
            resp.raise_for_status()
            faq_urls = _extract_same_host_pdf_links(
                resp.text,
                page_url=faq_page_url,
                allowed_host=allowed_host,
            )
            if not faq_urls:
                raise RuntimeError(f"No FAQ PDF links found on {faq_page_url}")

            for idx, pdf_url in enumerate(faq_urls, start=1):
                filename = _safe_filename_from_url(pdf_url)
                dest = faq_dir / filename
                manifest_key = f"faq/{filename}"
                if _download_pdf_if_needed(
                    active_client,
                    url=pdf_url,
                    dest=dest,
                    manifest_key=manifest_key,
                    manifest=manifest,
                    force=force,
                ):
                    downloaded += 1
                else:
                    skipped += 1
                faq_files += 1
                if idx % 10 == 0 or idx == len(faq_urls):
                    logger.info("FAQ PDFs processed: %d/%d", idx, len(faq_urls))

        if only in ("all", "errata"):
            errata_page_url = f"{normalized_base}/rules/errata_card/"
            logger.info("Fetching official errata HTML from %s", errata_page_url)
            resp = active_client.get(errata_page_url)
            resp.raise_for_status()
            html_text = resp.text
            if "errata" not in html_text.lower():
                raise ValueError(
                    f"Unexpected response from {errata_page_url}: does not look like errata HTML"
                )
            dest = faq_dir / "errata_card.html"
            dest.parent.mkdir(parents=True, exist_ok=True)
            existing_html = (
                dest.read_text(encoding="utf-8", errors="replace") if dest.is_file() else None
            )
            if force or existing_html != html_text:
                dest.write_text(html_text, encoding="utf-8")
                manifest["faq/errata_card.html"] = errata_page_url
                downloaded += 1
                logger.info("Saved %s", dest)
            else:
                skipped += 1
                logger.info("Up-to-date (skipped): %s", dest)
            errata_files = 1

        _save_manifest(manifest_path, manifest)
    finally:
        if owns_client:
            active_client.close()

    return OfficialFetchSummary(
        downloaded=downloaded,
        skipped=skipped,
        rules_files=rules_files,
        faq_files=faq_files,
        errata_files=errata_files,
        out_dir=target_root,
    )
