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

        pack_ids: list[str] = []
        if isinstance(packs_payload, dict):
            for key, entry in packs_payload.items():
                if isinstance(entry, dict) and entry.get("id"):
                    pack_ids.append(str(entry["id"]))
                else:
                    pack_ids.append(str(key))
        elif isinstance(packs_payload, list):
            for entry in packs_payload:
                if isinstance(entry, dict) and entry.get("id"):
                    pack_ids.append(str(entry["id"]))
        else:
            raise ValueError(f"Expected JSON object or list from {packs_url}")

        return sorted(dict.fromkeys(pack_ids))

    def _fetch_pack_cards(self, client: httpx.Client, pack_id: str) -> Any | None:
        # punk-records v2.0 stores per-pack arrays under data/{pack_id}.json;
        # older layouts store them under cards/{pack_id}.json.
        candidate_urls = (
            f"{self._base_url}/data/{pack_id}.json",
            f"{self._base_url}/cards/{pack_id}.json",
        )
        for pack_url in candidate_urls:
            try:
                return self._fetch_json(client, pack_url)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    continue
                raise
        logger.warning("Pack file not found (404), skipping pack_id=%s", pack_id)
        return None

    def fetch_raw_cards(self) -> list[dict[str, Any]]:
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self._timeout, follow_redirects=True)
        try:
            pack_ids = self._resolve_pack_ids(client)
            cards: list[dict[str, Any]] = []
            for pack_id in pack_ids:
                payload = self._fetch_pack_cards(client, pack_id)
                if payload is None:
                    continue
                if isinstance(payload, list):
                    cards.extend(item for item in payload if isinstance(item, dict))
                else:
                    logger.warning("Unexpected payload type for pack_id=%s; skipping", pack_id)
            return cards
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Failed to fetch card data from {self._base_url}: {exc}. "
                "You can also import local JSON files from data/raw/cards/ via --source local."
            ) from exc
        finally:
            if owns_client:
                client.close()
