"""Model Context Protocol server exposing OmniProfit as tools.

Any MCP client (an IDE assistant, a desktop chat app, another agent) can call
these tools to query profit figures and get guarded recommendations.
Run with ``omniprofit serve-mcp --data data`` (stdio transport).
"""

from __future__ import annotations

from pathlib import Path

from .llm import get_llm
from .mock import mock_handler
from .orchestrator import Orchestrator


def channel_profit(data_dir: str) -> list[dict]:
    """Contribution margin for every sales channel."""
    return [line.model_dump() for line in Orchestrator().run(Path(data_dir), advise=False).report.channels]


def sku_profit(data_dir: str, channel: str) -> list[dict]:
    """Contribution margin for every SKU sold on one channel."""
    report = Orchestrator().run(Path(data_dir), advise=False).report
    lines = [line.model_dump() for line in report.skus if line.channel == channel]
    if not lines:
        raise ValueError(f"no sales found for channel '{channel}'")
    return lines


def recommend(data_dir: str, provider: str | None = None) -> dict:
    """Guardrail-checked actions, with rejected proposals and the reasons they were rejected."""
    llm = get_llm(provider, mock_handler=mock_handler)
    return Orchestrator(llm).run(Path(data_dir)).advice.model_dump()


def build_server(data_dir: str):
    """Create the MCP server. Works with both v1 and v2 of the ``mcp`` package."""
    try:
        from mcp.server.mcpserver import MCPServer as Server  # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server  # mcp 1.x

    server = Server("omniprofit")

    @server.tool(description="Contribution margin for every sales channel.")
    def get_channel_profit() -> list[dict]:
        return channel_profit(data_dir)

    @server.tool(description="Contribution margin for every SKU sold on one channel.")
    def get_sku_profit(channel: str) -> list[dict]:
        return sku_profit(data_dir, channel)

    @server.tool(description="Recommended actions that passed the guardrails, plus rejected ones with reasons.")
    def get_recommendations() -> dict:
        return recommend(data_dir)

    return server
