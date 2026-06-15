"""
MCP client factory for sub-agents.

MCP server assignments are read from the unified ``agents/agent_config.json``.
Each agent OWNS the servers listed under its ``mcp_servers`` key — the config
loader validates at startup that no server is claimed by more than one agent,
and ``create_mcp_client`` re-checks this invariant at runtime as a safety net.
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

from config_loader import AGENT_CONFIG


def _check_mcp_ownership(agent_name: str, server_names: set[str]) -> None:
    """Raise ``ValueError`` if any server in *server_names* is also declared by
    another agent in ``AGENT_CONFIG`` (defence-in-depth — the primary check
    happens in ``config_loader`` at import time)."""
    for other_agent, cfg in AGENT_CONFIG.items():
        if other_agent == agent_name:
            continue
        other_servers = set(cfg.get("mcp_servers", {}).keys())
        conflict = server_names & other_servers
        if conflict:
            raise ValueError(
                f"MCP server(s) {sorted(conflict)} are declared by both "
                f"'{agent_name}' and '{other_agent}'. Each MCP server must "
                f"have exactly one owner."
            )


def create_mcp_client(agent_name: str) -> MultiServerMCPClient | None:
    """
    Return an MCP client pre-configured with the servers owned by *agent_name*,
    or ``None`` if the agent has no MCP servers.

    The caller MUST use ``async with client:`` around tool usage to keep the
    MCP transport alive for the duration of the agent call.
    """
    agent_cfg = AGENT_CONFIG.get(agent_name, {})
    server_map: dict[str, str] = agent_cfg.get("mcp_servers", {})

    if not server_map:
        return None

    # Defence-in-depth: ensure no other agent also claims these servers
    _check_mcp_ownership(agent_name, set(server_map.keys()))

    client = MultiServerMCPClient(
        {
            name: {"url": url, "transport": "streamable_http"}
            for name, url in server_map.items()
        }
    )

    return client
