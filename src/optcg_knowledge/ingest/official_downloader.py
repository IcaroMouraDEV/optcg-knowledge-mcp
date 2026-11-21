"""Downloader and freshness checker for official Bandai OPTCG documents.

Never downloads card images—only official rules/TRM/FAQ PDFs and text-only
errata/restriction HTML pages into the local gitignored ``data/raw/`` directory,
or inspects live document revisions and semantic SHA-256 fingerprints in memory
without writing any Bandai content to the repository.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import httpx

from optcg_knowledge.ingest.banlist_parser import parse_banlist_html
from optcg_knowledge.ingest.faq_parser import parse_faq_html

logger = logging.getLogger(__name__)

DEFAULT_OFFICIAL_BASE_URL = "https://en.onepiece-cardgame.com"
DEFAULT_USER_AGENT = "optcg-knowledge/0.1 (+https://github.com/IcaroMouraDEV/optcg-knowledge-mcp)"
OFFICIAL_REVISIONS_SCHEMA_VERSION = "1.0"
DEFAULT_OFFICIAL_BASELINE_PATH = Path("data/official_revisions.json")

FetchTarget = Literal["all", "rules", "trm", "faq", "errata", "banlist"]
DocumentCategory = Literal["rules", "trm", "faq", "errata", "banlist"]
ChangeType = Literal["added", "updated", "removed"]

_VALID_TARGETS: tuple[FetchTarget, ...] = ("all", "rules", "trm", "faq", "errata", "banlist")
_VALID_CATEGORIES: frozenset[str] = frozenset({"rules", "trm", "faq", "errata", "banlist"})

_PDF_HREF_RE = re.compile(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', re.IGNORECASE)
_MANIFEST_FILENAME = ".official_manifest.json"
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
) -> httpx.Response:
    """Perform an HTTP GET with bounded retries on transient 5xx/429/transport errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            resp = client.get(url)
            if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < max_attempts:
                logger.warning(
                    "Transient HTTP %d from %s (attempt %d/%d); retrying...",
                    resp.status_code,
                    url,
                    attempt,
                    max_attempts,
                )
                if backoff_seconds > 0:
                    time.sleep(backoff_seconds * attempt)
                continue
            resp.raise_for_status()
            return resp
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            logger.warning(
                "Transient transport error fetching %s (attempt %d/%d): %s; retrying...",
                url,
                attempt,
                max_attempts,
                exc,
            )
            if backoff_seconds > 0:
                time.sleep(backoff_seconds * attempt)
    assert last_exc is not None
    raise last_exc


@dataclass(frozen=True, slots=True)
class OfficialFetchSummary:
    """Counters returned after fetching official Bandai rules, TRM, FAQs, errata, and banlist."""

    downloaded: int
    skipped: int
    rules_files: int
    faq_files: int
    errata_files: int
    out_dir: Path
    trm_files: int = 0
    banlist_files: int = 0


@dataclass(frozen=True, slots=True)
class OfficialDocumentRevision:
    """Tracked revision metadata for a single official Bandai OPTCG document.

    Contains only URLs, revision query tokens, SHA-256 semantic fingerprints,
    counts, and effective dates—never copyrighted Bandai text or binaries.
    """

    key: str
    category: DocumentCategory
    url: str
    revision: str
    sha256: str | None = None
    entry_count: int | None = None
    upcoming_count: int | None = None
    effective_date: str | None = None
    upcoming_effective_date: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "key": self.key,
            "category": self.category,
            "url": self.url,
            "revision": self.revision,
        }
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.entry_count is not None:
            payload["entry_count"] = self.entry_count
        if self.upcoming_count is not None:
            payload["upcoming_count"] = self.upcoming_count
        if self.effective_date is not None:
            payload["effective_date"] = self.effective_date
        if self.upcoming_effective_date is not None:
            payload["upcoming_effective_date"] = self.upcoming_effective_date
        return payload


@dataclass(frozen=True, slots=True)
class OfficialDocumentDiff:
    """A single detected difference between the tracked baseline and live Bandai pages."""

    key: str
    category: DocumentCategory
    change_type: ChangeType
    previous: OfficialDocumentRevision | None
    current: OfficialDocumentRevision | None

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "category": self.category,
            "change_type": self.change_type,
            "previous": self.previous.to_dict() if self.previous is not None else None,
            "current": self.current.to_dict() if self.current is not None else None,
        }


@dataclass(frozen=True, slots=True)
class OfficialFreshnessReport:
    """Structured freshness comparison report across official Bandai OPTCG documents."""

    up_to_date: bool
    only: FetchTarget
    base_url: str
    baseline_path: Path | None
    baseline_exists: bool
    baseline_updated: bool
    checked_count: int
    unchanged_count: int
    added: tuple[OfficialDocumentDiff, ...]
    updated: tuple[OfficialDocumentDiff, ...]
    removed: tuple[OfficialDocumentDiff, ...]
    documents: tuple[OfficialDocumentRevision, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": OFFICIAL_REVISIONS_SCHEMA_VERSION,
            "up_to_date": self.up_to_date,
            "only": self.only,
            "base_url": self.base_url,
            "baseline_path": str(self.baseline_path) if self.baseline_path is not None else None,
            "baseline_exists": self.baseline_exists,
            "baseline_updated": self.baseline_updated,
            "checked_count": self.checked_count,
            "unchanged_count": self.unchanged_count,
            "added_count": len(self.added),
            "updated_count": len(self.updated),
            "removed_count": len(self.removed),
            "added": [item.to_dict() for item in self.added],
            "updated": [item.to_dict() for item in self.updated],
            "removed": [item.to_dict() for item in self.removed],
            "documents": [doc.to_dict() for doc in self.documents],
        }

    def to_markdown(self) -> str:
        status_text = "Up to date" if self.up_to_date else "Updates detected"
        lines = [
            "## Official Bandai Document Freshness Check",
            "",
            f"- **Status:** {status_text}",
            f"- **Source:** `{self.base_url}`",
            f"- **Scope:** `{self.only}`",
            (
                f"- **Checked documents:** {self.checked_count} "
                f"(`unchanged={self.unchanged_count}`, `added={len(self.added)}`, "
                f"`updated={len(self.updated)}`, `removed={len(self.removed)}`)"
            ),
        ]
        if self.baseline_path is not None:
            lines.append(f"- **Baseline:** `{self.baseline_path}`")

        diffs = (*self.added, *self.updated, *self.removed)
        if diffs:
            lines.extend(
                [
                    "",
                    "### Detected Changes",
                    "",
                    (
                        "| Change | Category | Document Key | Previous Revision | "
                        "Current Revision | Details |"
                    ),
                    "| :--- | :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for diff in diffs:
                prev_rev = diff.previous.revision if diff.previous is not None else "-"
                curr_rev = diff.current.revision if diff.current is not None else "-"
                detail = _format_diff_details(diff.current or diff.previous)
                lines.append(
                    f"| `{diff.change_type}` | `{diff.category}` | `{diff.key}` | "
                    f"`{prev_rev}` | `{curr_rev}` | {detail} |"
                )
            lines.extend(
                [
                    "",
                    "### How to Sync Locally & Update Baseline",
                    "",
                    "```bash",
                    "# 1. Fetch and ingest updated official documents into your local SQLite DB",
                    "uv run optcg-ingest sync",
                    "",
                    "# 2. Refresh the tracked revisions baseline and commit via Pull Request",
                    "uv run optcg-ingest check-official --write-baseline",
                    "```",
                ]
            )
        return "\n".join(lines)


def _format_diff_details(doc: OfficialDocumentRevision | None) -> str:
    if doc is None:
        return "-"
    parts: list[str] = [f"[Link]({doc.url})"]
    if doc.effective_date:
        parts.append(f"effective `{doc.effective_date}`")
    if doc.upcoming_effective_date:
        parts.append(f"upcoming `{doc.upcoming_effective_date}`")
    if doc.entry_count is not None:
        parts.append(f"entries={doc.entry_count}")
    if doc.upcoming_count is not None and doc.upcoming_count > 0:
        parts.append(f"upcoming_entries={doc.upcoming_count}")
    return ", ".join(parts)


class _VisibleTextHTMLStripper(HTMLParser):
    """Extract normalized visible text outside script/style/navigation Chrome."""

    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "header", "footer", "nav"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in self._IGNORED_TAGS:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_TAGS and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth == 0 and data.strip():
            self._chunks.append(data)

    def get_normalized_text(self) -> str:
        return " ".join(" ".join(self._chunks).split())


def _extract_visible_html_text(html_text: str) -> str:
    stripper = _VisibleTextHTMLStripper()
    stripper.feed(html_text)
    stripper.close()
    return stripper.get_normalized_text()


def _fingerprint_errata_html(
    html_text: str,
    *,
    page_url: str,
) -> tuple[str, int, str | None]:
    """Return ``(sha256_hex, entry_count, detected_date)`` for an errata HTML page."""
    doc = parse_faq_html(html_text, url=page_url)
    if doc.entries:
        lines = [
            "|".join(
                [
                    entry.ruling.ruling_id,
                    entry.ruling.date or "",
                    ",".join(entry.ruling.card_ids),
                    " ".join(entry.ruling.question.split()),
                    " ".join(entry.ruling.answer.split()),
                ]
            )
            for entry in sorted(doc.entries, key=lambda item: item.ruling.ruling_id)
        ]
        payload = "\n".join(lines)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return digest, len(doc.entries), doc.date

    visible = _extract_visible_html_text(html_text)
    digest = hashlib.sha256(visible.encode("utf-8")).hexdigest()
    return digest, 0, doc.date


def _fingerprint_banlist_html(
    html_text: str,
    *,
    page_url: str,
) -> tuple[str, int, int, str | None, str | None]:
    """Return ``(sha256_hex, active_count, upcoming_count, effective_date, upcoming_date)``."""
    doc = parse_banlist_html(html_text, source_url=page_url, include_upcoming=True)
    all_records = (*doc.restrictions, *doc.upcoming_restrictions)
    if all_records:
        lines = [
            (
                f"active_date={doc.effective_date or ''}|"
                f"upcoming_date={doc.upcoming_effective_date or ''}"
            )
        ]
        for rec in sorted(all_records, key=lambda item: (item.scope, item.card_id)):
            lines.append(
                "|".join(
                    [
                        rec.scope,
                        rec.card_id,
                        rec.ban_status,
                        rec.effective_date or "",
                        ",".join(rec.paired_card_ids),
                    ]
                )
            )
        payload = "\n".join(lines)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return (
            digest,
            len(doc.restrictions),
            len(doc.upcoming_restrictions),
            doc.effective_date,
            doc.upcoming_effective_date,
        )

    visible = _extract_visible_html_text(html_text)
    digest = hashlib.sha256(visible.encode("utf-8")).hexdigest()
    return (
        digest,
        0,
        0,
        doc.effective_date,
        doc.upcoming_effective_date,
    )


def _extract_pdf_revision(url: str) -> str:
    """Extract the revision query parameter (e.g. ``20260828``) or filename from a PDF URL."""
    parsed = urlparse(url)
    query = parsed.query.strip()
    if query:
        return query
    return _safe_filename_from_url(url)


def _infer_category_from_key(key: str) -> DocumentCategory:
    lower = key.lower()
    if lower == "rules/rule_comprehensive.pdf":
        return "rules"
    if lower == "rules/tournament_rules_manual.pdf":
        return "trm"
    if lower == "faq/errata_card.html" or "errata" in lower:
        return "errata"
    if lower.startswith("banlist/") or "restriction" in lower:
        return "banlist"
    return "faq"


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


def _parse_baseline_entry(key: str, raw: Any) -> OfficialDocumentRevision:
    if isinstance(raw, str):
        category = _infer_category_from_key(key)
        revision = _extract_pdf_revision(raw) if raw.lower().split("?")[0].endswith(".pdf") else raw
        return OfficialDocumentRevision(
            key=key,
            category=category,
            url=raw,
            revision=revision,
        )
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid official revision entry for {key!r}: expected object or URL")

    raw_cat = str(raw.get("category") or _infer_category_from_key(key))
    category: DocumentCategory = (
        raw_cat  # type: ignore[assignment]
        if raw_cat in _VALID_CATEGORIES
        else _infer_category_from_key(key)
    )
    url = str(raw.get("url") or "")
    if not url:
        raise ValueError(f"Official revision entry {key!r} is missing 'url'")
    revision = str(raw.get("revision") or _extract_pdf_revision(url))
    sha256_val = str(raw["sha256"]) if raw.get("sha256") is not None else None
    entry_count = int(raw["entry_count"]) if raw.get("entry_count") is not None else None
    upcoming_count = int(raw["upcoming_count"]) if raw.get("upcoming_count") is not None else None
    effective_date = str(raw["effective_date"]) if raw.get("effective_date") is not None else None
    upcoming_effective_date = (
        str(raw["upcoming_effective_date"])
        if raw.get("upcoming_effective_date") is not None
        else None
    )
    return OfficialDocumentRevision(
        key=key,
        category=category,
        url=url,
        revision=revision,
        sha256=sha256_val,
        entry_count=entry_count,
        upcoming_count=upcoming_count,
        effective_date=effective_date,
        upcoming_effective_date=upcoming_effective_date,
    )


def load_official_baseline(baseline_path: Path | str) -> dict[str, OfficialDocumentRevision]:
    """Load a tracked official revisions baseline JSON (or legacy flat manifest)."""
    path = Path(baseline_path)
    if not path.is_file():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in official revisions baseline: {path}")

    raw_docs = parsed.get("documents") if isinstance(parsed.get("documents"), dict) else parsed
    assert isinstance(raw_docs, dict)
    result: dict[str, OfficialDocumentRevision] = {}
    for key, value in sorted(raw_docs.items()):
        if key in ("schema_version", "base_url"):
            continue
        str_key = str(key)
        result[str_key] = _parse_baseline_entry(str_key, value)
    return result


def save_official_baseline(
    baseline_path: Path | str,
    documents: dict[str, OfficialDocumentRevision],
    *,
    base_url: str = DEFAULT_OFFICIAL_BASE_URL,
) -> None:
    """Persist tracked official document revisions in deterministic sorted JSON."""
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": OFFICIAL_REVISIONS_SCHEMA_VERSION,
        "base_url": base_url.rstrip("/"),
        "documents": {key: documents[key].to_dict() for key in sorted(documents)},
    }
    path.write_text(
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n",
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


def _discover_rules_and_trm_urls(
    client: httpx.Client,
    *,
    normalized_base: str,
    allowed_host: str,
    only: FetchTarget,
) -> tuple[str | None, str | None]:
    rules_page_url = f"{normalized_base}/rules/"
    logger.info("Checking official rules & TRM links at %s", rules_page_url)
    resp = _get_with_retry(client, rules_page_url)
    pdf_urls = _extract_same_host_pdf_links(
        resp.text,
        page_url=rules_page_url,
        allowed_host=allowed_host,
    )

    comp_url: str | None = None
    trm_url: str | None = None

    if only in ("all", "rules"):
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

    if only in ("all", "trm"):
        trm_urls = [
            u
            for u in pdf_urls
            if "tournament_rules_manual" in Path(urlparse(u).path).name.lower()
        ]
        if not trm_urls:
            raise RuntimeError(
                f"Could not find tournament_rules_manual*.pdf link on {rules_page_url}"
            )
        trm_url = trm_urls[0]

    return comp_url, trm_url


def _discover_faq_pdf_urls(
    client: httpx.Client,
    *,
    normalized_base: str,
    allowed_host: str,
) -> list[str]:
    faq_page_url = f"{normalized_base}/rules/faq/"
    logger.info("Checking official FAQ PDF links at %s", faq_page_url)
    resp = _get_with_retry(client, faq_page_url)
    faq_urls = _extract_same_host_pdf_links(
        resp.text,
        page_url=faq_page_url,
        allowed_host=allowed_host,
    )
    if not faq_urls:
        raise RuntimeError(f"No FAQ PDF links found on {faq_page_url}")
    return faq_urls


def _fetch_errata_html(
    client: httpx.Client,
    *,
    normalized_base: str,
) -> tuple[str, str]:
    errata_page_url = f"{normalized_base}/rules/errata_card/"
    logger.info("Fetching official errata HTML from %s", errata_page_url)
    resp = _get_with_retry(client, errata_page_url)
    html_text = resp.text
    if "errata" not in html_text.lower():
        raise ValueError(
            f"Unexpected response from {errata_page_url}: does not look like errata HTML"
        )
    return errata_page_url, html_text


def _fetch_banlist_html(
    client: httpx.Client,
    *,
    normalized_base: str,
) -> tuple[str, str]:
    banlist_page_url = f"{normalized_base}/rules/restriction/"
    logger.info("Fetching official banlist HTML from %s", banlist_page_url)
    resp = _get_with_retry(client, banlist_page_url)
    html_text = resp.text
    lower_html = html_text.lower()
    if "banned" not in lower_html and "restricted" not in lower_html:
        raise ValueError(
            f"Unexpected response from {banlist_page_url}: "
            "does not look like banlist/restriction HTML"
        )
    resolved_url = str(resp.url) if resp.url else banlist_page_url
    return resolved_url, html_text


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

    resp = _get_with_retry(client, url)
    data = resp.content
    if not data.startswith(b"%PDF"):
        raise ValueError(f"Expected PDF response from {url}, received non-PDF payload")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    manifest[manifest_key] = url
    return True


def discover_official_revisions(
    *,
    only: FetchTarget = "all",
    base_url: str = DEFAULT_OFFICIAL_BASE_URL,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> tuple[OfficialDocumentRevision, ...]:
    """Inspect live official Bandai OPTCG pages without downloading PDFs or images.

    Performs at most 4 text-only HTML requests (``/rules/``, ``/rules/faq/``,
    ``/rules/errata_card/``, and ``/rules/restriction/``) and returns deterministic
    revision records sorted by document key.
    """
    if only not in _VALID_TARGETS:
        raise ValueError(
            f"Unsupported target {only!r}; "
            "expected one of: all, rules, trm, faq, errata, banlist"
        )

    normalized_base = base_url.rstrip("/")
    allowed_host = urlparse(normalized_base).netloc
    revisions: dict[str, OfficialDocumentRevision] = {}

    owns_client = client is None
    active_client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )

    try:
        if only in ("all", "rules", "trm"):
            comp_url, trm_url = _discover_rules_and_trm_urls(
                active_client,
                normalized_base=normalized_base,
                allowed_host=allowed_host,
                only=only,
            )
            if comp_url is not None:
                key = "rules/rule_comprehensive.pdf"
                revisions[key] = OfficialDocumentRevision(
                    key=key,
                    category="rules",
                    url=comp_url,
                    revision=_extract_pdf_revision(comp_url),
                )
            if trm_url is not None:
                key = "rules/tournament_rules_manual.pdf"
                revisions[key] = OfficialDocumentRevision(
                    key=key,
                    category="trm",
                    url=trm_url,
                    revision=_extract_pdf_revision(trm_url),
                )

        if only in ("all", "faq"):
            faq_urls = _discover_faq_pdf_urls(
                active_client,
                normalized_base=normalized_base,
                allowed_host=allowed_host,
            )
            for pdf_url in faq_urls:
                filename = _safe_filename_from_url(pdf_url)
                key = f"faq/{filename}"
                revisions[key] = OfficialDocumentRevision(
                    key=key,
                    category="faq",
                    url=pdf_url,
                    revision=_extract_pdf_revision(pdf_url),
                )

        if only in ("all", "errata"):
            errata_url, errata_html = _fetch_errata_html(
                active_client,
                normalized_base=normalized_base,
            )
            digest, entry_count, detected_date = _fingerprint_errata_html(
                errata_html,
                page_url=errata_url,
            )
            key = "faq/errata_card.html"
            revisions[key] = OfficialDocumentRevision(
                key=key,
                category="errata",
                url=errata_url,
                revision=f"sha256:{digest[:12]}",
                sha256=digest,
                entry_count=entry_count,
                effective_date=detected_date,
            )

        if only in ("all", "banlist"):
            banlist_url, banlist_html = _fetch_banlist_html(
                active_client,
                normalized_base=normalized_base,
            )
            (
                digest,
                active_count,
                upcoming_count,
                effective_date,
                upcoming_effective_date,
            ) = _fingerprint_banlist_html(
                banlist_html,
                page_url=banlist_url,
            )
            key = "banlist/restriction.html"
            revisions[key] = OfficialDocumentRevision(
                key=key,
                category="banlist",
                url=banlist_url,
                revision=f"sha256:{digest[:12]}",
                sha256=digest,
                entry_count=active_count,
                upcoming_count=upcoming_count,
                effective_date=effective_date,
                upcoming_effective_date=upcoming_effective_date,
            )
    finally:
        if owns_client:
            active_client.close()

    return tuple(revisions[k] for k in sorted(revisions))


def _revisions_differ(
    previous: OfficialDocumentRevision,
    current: OfficialDocumentRevision,
) -> bool:
    if previous.sha256 is not None or current.sha256 is not None:
        if previous.sha256 is not None and current.sha256 is not None:
            return previous.sha256 != current.sha256
        return previous.revision != current.revision
    return previous.url != current.url or previous.revision != current.revision


def check_official_freshness(
    baseline_path: Path | str | None = DEFAULT_OFFICIAL_BASELINE_PATH,
    *,
    only: FetchTarget = "all",
    base_url: str = DEFAULT_OFFICIAL_BASE_URL,
    write_baseline: bool = False,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> OfficialFreshnessReport:
    """Compare live official Bandai document revisions against a tracked baseline JSON.

    Never downloads PDF binaries or card images, and only writes to ``baseline_path``
    when ``write_baseline=True``.
    """
    resolved_baseline = Path(baseline_path) if baseline_path is not None else None
    if write_baseline and resolved_baseline is None:
        raise ValueError("baseline_path is required when write_baseline=True")

    baseline_exists = resolved_baseline.is_file() if resolved_baseline is not None else False
    if resolved_baseline is not None and not baseline_exists and not write_baseline:
        raise FileNotFoundError(
            f"Official revisions baseline file not found: {resolved_baseline}. "
            "Run `optcg-ingest check-official --write-baseline` to create it."
        )

    existing_all = (
        load_official_baseline(resolved_baseline)
        if resolved_baseline is not None and baseline_exists
        else {}
    )
    discovered = discover_official_revisions(
        only=only,
        base_url=base_url,
        timeout=timeout,
        client=client,
    )
    discovered_by_key = {doc.key: doc for doc in discovered}

    if only == "all":
        existing_in_scope = dict(existing_all)
    else:
        existing_in_scope = {
            k: v for k, v in existing_all.items() if v.category == only
        }

    added: list[OfficialDocumentDiff] = []
    updated: list[OfficialDocumentDiff] = []
    removed: list[OfficialDocumentDiff] = []
    unchanged_count = 0

    all_keys = sorted(set(existing_in_scope) | set(discovered_by_key))
    for key in all_keys:
        prev = existing_in_scope.get(key)
        curr = discovered_by_key.get(key)
        if prev is None and curr is not None:
            added.append(
                OfficialDocumentDiff(
                    key=key,
                    category=curr.category,
                    change_type="added",
                    previous=None,
                    current=curr,
                )
            )
        elif prev is not None and curr is None:
            removed.append(
                OfficialDocumentDiff(
                    key=key,
                    category=prev.category,
                    change_type="removed",
                    previous=prev,
                    current=None,
                )
            )
        elif prev is not None and curr is not None:
            if _revisions_differ(prev, curr):
                updated.append(
                    OfficialDocumentDiff(
                        key=key,
                        category=curr.category,
                        change_type="updated",
                        previous=prev,
                        current=curr,
                    )
                )
            else:
                unchanged_count += 1

    baseline_updated = False
    if write_baseline and resolved_baseline is not None:
        if only == "all":
            merged = dict(discovered_by_key)
        else:
            merged = {
                k: v for k, v in existing_all.items() if v.category != only
            } | discovered_by_key
        save_official_baseline(resolved_baseline, merged, base_url=base_url)
        baseline_updated = True

    up_to_date = not (added or updated or removed)
    return OfficialFreshnessReport(
        up_to_date=up_to_date,
        only=only,
        base_url=base_url.rstrip("/"),
        baseline_path=resolved_baseline,
        baseline_exists=baseline_exists,
        baseline_updated=baseline_updated,
        checked_count=len(discovered),
        unchanged_count=unchanged_count,
        added=tuple(added),
        updated=tuple(updated),
        removed=tuple(removed),
        documents=discovered,
    )


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
    if only not in _VALID_TARGETS:
        raise ValueError(
            f"Unsupported target {only!r}; "
            "expected one of: all, rules, trm, faq, errata, banlist"
        )

    target_root = Path(out_dir)
    rules_dir = target_root / "rules"
    faq_dir = target_root / "faq"
    banlist_dir = target_root / "banlist"
    manifest_path = target_root / _MANIFEST_FILENAME
    manifest = _load_manifest(manifest_path)

    normalized_base = base_url.rstrip("/")
    allowed_host = urlparse(normalized_base).netloc

    downloaded = 0
    skipped = 0
    rules_files = 0
    trm_files = 0
    faq_files = 0
    errata_files = 0
    banlist_files = 0

    owns_client = client is None
    active_client = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )

    try:
        if only in ("all", "rules", "trm"):
            comp_url, trm_url = _discover_rules_and_trm_urls(
                active_client,
                normalized_base=normalized_base,
                allowed_host=allowed_host,
                only=only,
            )

            if comp_url is not None:
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

            if trm_url is not None:
                dest = rules_dir / "tournament_rules_manual.pdf"
                if _download_pdf_if_needed(
                    active_client,
                    url=trm_url,
                    dest=dest,
                    manifest_key="rules/tournament_rules_manual.pdf",
                    manifest=manifest,
                    force=force,
                ):
                    downloaded += 1
                    logger.info("Downloaded %s", dest)
                else:
                    skipped += 1
                    logger.info("Up-to-date (skipped): %s", dest)
                trm_files = 1

        if only in ("all", "faq"):
            faq_urls = _discover_faq_pdf_urls(
                active_client,
                normalized_base=normalized_base,
                allowed_host=allowed_host,
            )

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
            errata_page_url, html_text = _fetch_errata_html(
                active_client,
                normalized_base=normalized_base,
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

        if only in ("all", "banlist"):
            resolved_url, html_text = _fetch_banlist_html(
                active_client,
                normalized_base=normalized_base,
            )
            dest = banlist_dir / "restriction.html"
            dest.parent.mkdir(parents=True, exist_ok=True)
            existing_html = (
                dest.read_text(encoding="utf-8", errors="replace") if dest.is_file() else None
            )
            if force or existing_html != html_text:
                dest.write_text(html_text, encoding="utf-8")
                manifest["banlist/restriction.html"] = resolved_url
                downloaded += 1
                logger.info("Saved %s", dest)
            else:
                skipped += 1
                logger.info("Up-to-date (skipped): %s", dest)
            banlist_files = 1

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
        trm_files=trm_files,
        banlist_files=banlist_files,
    )
