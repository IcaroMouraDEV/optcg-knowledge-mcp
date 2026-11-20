"""MCP server and prompts for optcg-knowledge."""

from optcg_knowledge.mcp.prompts import build_curar_prompt, build_juiz_prompt
from optcg_knowledge.mcp.server import create_server, main

__all__ = [
    "build_curar_prompt",
    "build_juiz_prompt",
    "create_server",
    "main",
]
