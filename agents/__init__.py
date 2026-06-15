from config_loader import AGENT_CONFIG  # noqa: E402

# Backward-compatible AGENT_ROSTER: name → description
AGENT_ROSTER = {name: cfg["description"] for name, cfg in AGENT_CONFIG.items()}

from agents.orchestrator_node import orchestrator_agent  # noqa: E402
from agents.sub_agents_nodes import sub_agent_node, run_sub_agent_async  # noqa: E402


__all__ = [
    "AGENT_CONFIG",
    "AGENT_ROSTER",
    "orchestrator_agent",
    "run_sub_agent_async",
    "sub_agent_node",
]
