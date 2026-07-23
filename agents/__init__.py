from config_loader import AGENT_CONFIG  # noqa: E402

# Backward-compatible AGENT_ROSTER: name → description
AGENT_ROSTER = {name: cfg["description"] for name, cfg in AGENT_CONFIG.items()}

from agents.orchestrator_node import make_orchestrator_agent, orchestrator_agent  # noqa: E402
from agents.sub_agents_nodes import (  # noqa: E402
    make_sub_agent_node,
    make_parallel_sub_agent_node,
    parallel_sub_agent_node,
    sub_agent_node,
    run_sub_agent_async,
)


__all__ = [
    "AGENT_CONFIG",
    "AGENT_ROSTER",
    "make_orchestrator_agent",
    "make_parallel_sub_agent_node",
    "make_sub_agent_node",
    "orchestrator_agent",
    "parallel_sub_agent_node",
    "run_sub_agent_async",
    "sub_agent_node",
]
