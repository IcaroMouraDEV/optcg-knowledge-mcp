"""Local filesystem card JSON source (data/raw/cards/ or explicit file/dir)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IGNORED_FILENAMES = {"packs.json", "manifest.json", "by_name.json"}


def _extract_card_dicts(payload: Any, *, origin: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("cards", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        # Support map-style index files like cards_by_id.json: {"OP01-001": {...}}
        if payload and all(isinstance(v, dict) for v in payload.values()):
            cards: list[dict[str, Any]] = []
            for key, item in payload.items():
                if "id" not in item and "card_id" not in item:
                    item = {"card_id": key, **item}
                cards.append(item)
            return cards
    logger.warning("Skipping unrecognized JSON structure in %s", origin)
    return []


class LocalCardSource:
    """Load card JSON records from a local file or directory."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def source_name(self) -> str:
        return f"local:{self._path}"

    def fetch_raw_cards(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            raise FileNotFoundError(f"Local cards path not found: {self._path}")

        if self._path.is_file():
            files = [self._path]
        else:
            files = sorted(
                p
                for p in self._path.rglob("*.json")
                if p.is_file()
                and p.name not in _IGNORED_FILENAMES
                and "index" not in p.relative_to(self._path).parts
            )

        if not files:
            raise FileNotFoundError(
                f"No card JSON files found under {self._path}. "
                "Place exported JSON files in data/raw/cards/ or use --source punk-records."
            )

        records: list[dict[str, Any]] = []
        for json_file in files:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            extracted = _extract_card_dicts(data, origin=json_file)
            logger.debug("Loaded %d raw card records from %s", len(extracted), json_file)
            records.extend(extracted)
        return records
