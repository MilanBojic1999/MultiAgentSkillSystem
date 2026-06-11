import json
from pathlib import Path


def _load_agent_roster() -> dict:
    roster_path = Path(__file__).parent / "agent_rouster.json"
    with open(roster_path, "r", encoding="utf-8") as f:
        return json.load(f)


AGENT_ROSTER = _load_agent_roster()

from agents.orchestrator_node import orchestrator_agent  # noqa: E402
from agents.sub_agents_nodes import sub_agent_node, run_sub_agent_async  # noqa: E402


__all__ = ["orchestrator_agent", "sub_agent_node", "run_sub_agent_async", "AGENT_ROSTER"]