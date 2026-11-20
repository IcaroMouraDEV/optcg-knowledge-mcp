"""Database layer for optcg-knowledge."""

from optcg_knowledge.db.connection import connect, open_db
from optcg_knowledge.db.models import (
    AUTHORITY_BY_SOURCE,
    JUDGE_TIER_RANK,
    CardRecord,
    ChunkRecord,
    JudgeTier,
    RuleRecord,
    RulingRecord,
    authority_for_source,
    judge_tier_rank,
    normalize_judge_tier,
)
from optcg_knowledge.db.schema import (
    SCHEMA_VERSION,
    ensure_vec_table,
    get_meta,
    init_schema,
    migrate_schema,
    set_meta,
    utc_now_iso,
)

__all__ = [
    "AUTHORITY_BY_SOURCE",
    "JUDGE_TIER_RANK",
    "CardRecord",
    "ChunkRecord",
    "JudgeTier",
    "RuleRecord",
    "RulingRecord",
    "SCHEMA_VERSION",
    "authority_for_source",
    "connect",
    "ensure_vec_table",
    "get_meta",
    "init_schema",
    "judge_tier_rank",
    "migrate_schema",
    "normalize_judge_tier",
    "open_db",
    "set_meta",
    "utc_now_iso",
]
