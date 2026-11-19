"""Pluggable card data source protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CardSource(Protocol):
    """Protocol implemented by local and remote card JSON providers."""

    @property
    def source_name(self) -> str:
        """Human-readable name recorded in metadata/logs."""
        ...

    def fetch_raw_cards(self) -> list[dict[str, Any]]:
        """Return raw card dictionaries prior to canonical normalization."""
        ...
