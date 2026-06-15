"""
Agent-to-tool mapping.

Reads tool assignments from the unified agent config and resolves each tool name
against the auto-discovered TOOL_REGISTRY from tools/__init__.py.

To assign tools to an agent, edit the "tools" list in agents/agent_config.json —
no code changes needed.
"""

from config_loader import AGENT_CONFIG
from tools import TOOL_REGISTRY


def _load_agent_tools() -> dict[str, list]:
    """Read the unified agent config and resolve every tool name to its object."""
    agent_tools: dict[str, list] = {}

    for agent_name, cfg in AGENT_CONFIG.items():
        tools = []
        for tname in cfg.get("tools", []):
            tool = TOOL_REGISTRY.get(tname)
            if tool is None:
                print(
                    f"Warning: tool '{tname}' (assigned to agent '{agent_name}') "
                    f"not found in TOOL_REGISTRY. Available: {sorted(TOOL_REGISTRY.keys())}"
                )
                continue
            tools.append(tool)
        agent_tools[agent_name] = tools

    return agent_tools


AGENT_TOOLS = _load_agent_tools()
