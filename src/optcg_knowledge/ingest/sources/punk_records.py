"""Static GitHub JSON source for buhbbl/punk-records (generated via vegapull)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from optcg_knowledge.config import DEFAULT_PUNK_RECORDS_URL

logger = logging.getLogger(__name__)


class PunkRecordsSource:
    """Fetch static OPTCG card JSON files from punk-records (never downloads images)."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_PUNK_RECORDS_URL,
        pack_ids: Sequence[str] | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._pack_ids = tuple(pack_ids) if pack_ids else None
        self._timeout = timeout
        self._client = client

    @property
    def source_name(self) -> str:
        return f"punk-records:{self._base_url}"

    def _fetch_json(self, client: httpx.Client, url: str) -> Any:
        response = client.get(url)
        response.raise_for_status()
        return response.json()

    def _resolve_pack_ids(self, client: httpx.Client) -> list[str]:
        if self._pack_ids is not None:
            return list(self._pack_ids)

        packs_url = f"{self._base_url}/packs.json"
        packs_payload = self._fetch_json(client, packs_url)
        if not isinstance(packs_payload, list):
            raise ValueError(f"Expected JSON list from {packs_url}")

        pack_ids: list[str] = []
        for entry in packs_payload:
            if isinstance(entry, dict) and entry.get("id"):
                pack_ids.append(str(entry["id"]))
        return pack_ids

    def fetch_raw_cards(self) -> list[dict[str, Any]]:
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        try:
            pack_ids = self._resolve_pack_ids(client)
            cards: list[dict[str, Any]] = []
            for pack_id in pack_ids:
                pack_url = f"{self._base_url}/cards/{pack_id}.json"
                try:
                    payload = self._fetch_json(client, pack_url)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        logger.warning("Pack file not found (404), skipping: %s", pack_url)
                        continue
                    raise
                if isinstance(payload, list):
                    cards.extend(item for item in payload if isinstance(item, dict))
                else:
                    logger.warning("Unexpected payload type from %s; skipping", pack_url)
            return cards
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to fetch card data from {self._base_url}: {exc}. "
                "You can also import local JSON files from data/raw/cards/ via --source local."
            ) from exc
        finally:
            if owns_client:
                client.close()
