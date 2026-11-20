"""Runtime configuration for optcg-knowledge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = Path("data/optcg.db")
DEFAULT_RAW_DATA_DIR = Path("data/raw")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_EMBEDDING_BATCH_SIZE = 8
DEFAULT_EMBEDDING_FP16 = True
DEFAULT_SEARCH_MODE = "auto"
DEFAULT_PUNK_RECORDS_URL = "https://raw.githubusercontent.com/buhbbl/punk-records/main/english"

VALID_SEARCH_MODES = frozenset({"auto", "hybrid", "lexical"})
VALID_EMBEDDING_DEVICES = frozenset({"auto", "cuda", "cpu"})

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


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables or explicit overrides."""

    db_path: Path = DEFAULT_DB_PATH
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = 1024
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_fp16: bool = DEFAULT_EMBEDDING_FP16
    search_mode: str = DEFAULT_SEARCH_MODE
    punk_records_url: str = DEFAULT_PUNK_RECORDS_URL


def get_settings(
    *,
    db_path: Path | str | None = None,
    raw_data_dir: Path | str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_device: str | None = None,
    embedding_batch_size: int | None = None,
    embedding_fp16: bool | None = None,
    search_mode: str | None = None,
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

    resolved_device = (
        embedding_device or os.getenv("OPTCG_EMBEDDING_DEVICE") or DEFAULT_EMBEDDING_DEVICE
    ).strip().lower()
    if resolved_device not in VALID_EMBEDDING_DEVICES and not resolved_device.startswith("cuda:"):
        raise ValueError(
            f"Unsupported embedding device {resolved_device!r}; "
            f"expected one of {sorted(VALID_EMBEDDING_DEVICES)} or 'cuda:<index>'"
        )

    env_batch_raw = os.getenv("OPTCG_EMBEDDING_BATCH_SIZE")
    resolved_batch_size = (
        embedding_batch_size
        if embedding_batch_size is not None
        else (int(env_batch_raw) if env_batch_raw else DEFAULT_EMBEDDING_BATCH_SIZE)
    )
    if resolved_batch_size <= 0:
        raise ValueError("embedding_batch_size must be a positive integer")

    resolved_fp16 = (
        embedding_fp16
        if embedding_fp16 is not None
        else _parse_bool_env("OPTCG_EMBEDDING_FP16", DEFAULT_EMBEDDING_FP16)
    )

    resolved_search_mode = (
        search_mode or os.getenv("OPTCG_SEARCH_MODE") or DEFAULT_SEARCH_MODE
    ).strip().lower()
    if resolved_search_mode not in VALID_SEARCH_MODES:
        raise ValueError(
            f"Unsupported search_mode {resolved_search_mode!r}; "
            f"expected one of {sorted(VALID_SEARCH_MODES)}"
        )

    resolved_punk_url = (
        punk_records_url or os.getenv("OPTCG_PUNK_RECORDS_URL") or DEFAULT_PUNK_RECORDS_URL
    ).rstrip("/")

    return Settings(
        db_path=resolved_db,
        raw_data_dir=resolved_raw,
        embedding_model=resolved_model,
        embedding_dim=resolved_dim,
        embedding_device=resolved_device,
        embedding_batch_size=resolved_batch_size,
        embedding_fp16=resolved_fp16,
        search_mode=resolved_search_mode,
        punk_records_url=resolved_punk_url,
    )
