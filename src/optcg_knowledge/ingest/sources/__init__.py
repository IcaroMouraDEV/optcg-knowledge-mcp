"""Pluggable card sources for optcg-ingest cards."""

from optcg_knowledge.ingest.sources.base import CardSource
from optcg_knowledge.ingest.sources.local import LocalCardSource
from optcg_knowledge.ingest.sources.punk_records import PunkRecordsSource

__all__ = ["CardSource", "LocalCardSource", "PunkRecordsSource"]
