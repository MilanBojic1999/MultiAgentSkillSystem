"""
Unified agent-configuration loader.

Reads a single ``agents/agent_config.json`` file that describes every agent:
- description   — human-readable role summary
- tools         — list of tool names (resolved against the auto-discovered TOOL_REGISTRY)
- mcp_servers   — dict of MCP server name → URL
- llm           — optional per-agent LLM overrides (Phase 4.3)

Each agent OWNS the MCP servers listed under its key; the loader validates that no
server is claimed by more than one agent.
"""

import json
from pathlib import Path
from typing import Any
import os
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "agents" / "agent_config.json"
_CONFIG_PATH = os.getenv("CONFIG_PATH") or str(_DEFAULT_CONFIG_PATH)


def _load_raw_config() -> dict[str, dict[str, Any]]:
    """Read and parse the unified agent-configuration JSON file."""
    if not Path(_CONFIG_PATH).is_file():
        raise FileNotFoundError(
            f"Agent config not found at '{_CONFIG_PATH}'. "
            f"Set CONFIG_PATH in your .env (see .env.example), or place the "
            f"config at '{_DEFAULT_CONFIG_PATH}'."
        )
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


_LLM_CONFIG_KEYS = frozenset({"model", "url", "api_key_env", "temperature", "max_tokens"})


def _validate_llm_blocks(config: dict[str, dict[str, Any]]) -> None:
    """Raise ``ValueError`` if any agent's ``llm`` block has unknown keys (Phase 4.3).

    Catches typos at import time so they fail loudly instead of silently
    falling back to the default endpoint mid-run.
    """
    for agent_name, cfg in config.items():
        llm_block = cfg.get("llm")
        if llm_block is None:
            continue
        if not isinstance(llm_block, dict):
            raise TypeError(
                f"Agent '{agent_name}': 'llm' must be a dict, "
                f"got {type(llm_block).__name__}"
            )
        bad = set(llm_block) - _LLM_CONFIG_KEYS
        if bad:
            raise ValueError(
                f"Agent '{agent_name}': unknown key(s) in 'llm' block: "
                f"{sorted(bad)}. Accepted keys: {sorted(_LLM_CONFIG_KEYS)}."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_raw = _load_raw_config()
_validate_mcp_ownership(_raw)
_validate_llm_blocks(_raw)
AGENT_CONFIG: dict[str, dict[str, Any]] = _raw
"""Agent-keyed dictionary loaded from ``agents/agent_config.json``.

Each value is a dict with:
- ``description``  (str)
- ``tools``        (list[str])
- ``mcp_servers``  (dict[str, str])
- ``llm``          (dict | None)  — optional per-agent LLM overrides
"""
