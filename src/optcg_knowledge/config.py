"""Runtime configuration for optcg-knowledge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = Path("data/optcg.db")
DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_PUNK_RECORDS_URL = "https://raw.githubusercontent.com/buhbbl/punk-records/main/english"

KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-small": 384,
}


def resolve_embedding_dim(model_name: str, override: int | None = None) -> int:
    """Resolve vector dimension for an embedding model."""
    if override is not None:
        if override <= 0:
            raise ValueError("embedding_dim must be a positive integer")
        return override
    return KNOWN_EMBEDDING_DIMS.get(model_name, 1024)


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables or explicit overrides."""

    db_path: Path = DEFAULT_DB_PATH
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = 1024
    punk_records_url: str = DEFAULT_PUNK_RECORDS_URL


def get_settings(
    *,
    db_path: Path | str | None = None,
    raw_data_dir: Path | str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    punk_records_url: str | None = None,
) -> Settings:
    """Build a Settings instance from explicit arguments and environment variables."""
    resolved_db = Path(db_path or os.getenv("OPTCG_DB_PATH") or DEFAULT_DB_PATH)
    resolved_raw = Path(raw_data_dir or os.getenv("OPTCG_RAW_DATA_DIR") or DEFAULT_RAW_DATA_DIR)
    resolved_model = (
        embedding_model or os.getenv("OPTCG_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    )

    env_dim_raw = os.getenv("OPTCG_EMBEDDING_DIM")
    env_dim = int(env_dim_raw) if env_dim_raw else None
    resolved_dim = resolve_embedding_dim(resolved_model, override=embedding_dim or env_dim)

    resolved_punk_url = (
        punk_records_url or os.getenv("OPTCG_PUNK_RECORDS_URL") or DEFAULT_PUNK_RECORDS_URL
    ).rstrip("/")

    return Settings(
        db_path=resolved_db,
        raw_data_dir=resolved_raw,
        embedding_model=resolved_model,
        embedding_dim=resolved_dim,
        punk_records_url=resolved_punk_url,
    )
