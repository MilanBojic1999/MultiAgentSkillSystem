"""
Unified agent-configuration loader.

Reads a single ``agents/agent_config.json`` file that describes every agent:
- description   — human-readable role summary
- tools         — list of tool names (resolved against the auto-discovered TOOL_REGISTRY)
- mcp_servers   — dict of MCP server name → URL

Each agent OWNS the MCP servers listed under its key; the loader validates that no
server is claimed by more than one agent.
"""

import json
from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = os.getenv("CONFIG_PATH")


def _load_raw_config() -> dict[str, dict[str, Any]]:
    """Read and parse the unified agent-configuration JSON file."""
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _validate_mcp_ownership(config: dict[str, dict[str, Any]]) -> None:
    """Raise ``ValueError`` if any MCP server is declared by more than one agent."""
    server_owners: dict[str, str] = {}
    for agent_name, cfg in config.items():
        for server_name in cfg.get("mcp_servers", {}):
            if server_name in server_owners:
                raise ValueError(
                    f"MCP server '{server_name}' has multiple owners: "
                    f"'{server_owners[server_name]}' and '{agent_name}'."
                )
            server_owners[server_name] = agent_name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_raw = _load_raw_config()
_validate_mcp_ownership(_raw)
AGENT_CONFIG: dict[str, dict[str, Any]] = _raw
"""Agent-keyed dictionary loaded from ``agents/agent_config.json``.

Each value is a dict with:
- ``description``  (str)
- ``tools``        (list[str])
- ``mcp_servers``  (dict[str, str])
"""
