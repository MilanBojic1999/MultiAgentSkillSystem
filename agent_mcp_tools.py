"""
MCP client factory for sub-agents.

MCP server assignments are read from the unified ``agents/agent_config.json``.
Each agent OWNS the servers listed under its ``mcp_servers`` key — the config
loader validates at startup that no server is claimed by more than one agent,
and ``create_mcp_client`` re-checks this invariant at runtime as a safety net.

Three config shapes are accepted (detected per server by key, Phase 4.6):

- dict containing ``url``     → ``transport: "streamable_http"``
- dict containing ``command`` → ``transport: "stdio"``
- plain string                → treated as a URL with ``streamable_http``
  transport (legacy backward-compatibility)

``${VAR}`` expansion in URLs is handled elsewhere (see ``_expand_env_vars``
in ``langchain_mcp_adapters.sessions``).
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


def _build_server_map(server_map: dict) -> dict[str, dict]:
    """Convert an agent's ``mcp_servers`` block into the dict shape
    ``MultiServerMCPClient`` expects, detecting the transport **per server**.

    Each value in *server_map* must be a dict with exactly one of:

    - ``url``     → ``transport: "streamable_http"``
    - ``command`` → ``transport: "stdio"`` (``args``, ``env``, ``cwd`` optional)

    Raises ``ValueError`` if a config dict has neither recognised key.
    Raises ``TypeError`` if a value is not a dict.
    """
    result: dict[str, dict] = {}
    for name, conf in server_map.items():
        if not isinstance(conf, dict):
            raise TypeError(
                f"MCP server '{name}': expected a dict config, "
                f"got {type(conf).__name__}"
            )
        has_url = "url" in conf
        has_command = "command" in conf
        if has_command and has_url:
            raise ValueError(
                f"MCP server '{name}': config dict has both 'url' and "
                f"'command' — pick one."
            )
        if has_command:
            result[name] = {"transport": "stdio", **conf}
        elif has_url:
            result[name] = {"transport": "streamable_http", **conf}
        else:
            raise ValueError(
                f"MCP server '{name}': config dict must have 'url' or "
                f"'command' key. Got keys: {sorted(conf.keys())}"
            )
    return result


def create_mcp_client(agent_name: str) -> MultiServerMCPClient | None:
    """
    Return an MCP client pre-configured with the servers owned by *agent_name*,
    or ``None`` if the agent has no MCP servers.

    Each server's transport is detected from its config shape:
    ``url`` → streamable HTTP, ``command`` → stdio.
    """
    agent_cfg = AGENT_CONFIG.get(agent_name, {})
    server_map: dict = agent_cfg.get("mcp_servers", {})

    if not server_map:
        return None

    # Defence-in-depth: ensure no other agent also claims these servers
    _check_mcp_ownership(agent_name, set(server_map.keys()))
    client = MultiServerMCPClient(_build_server_map(server_map))

    return client
