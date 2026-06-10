from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import sub_agent_node, run_sub_agent_async


__all__ = ["orchestrator_agent", "sub_agent_node", "run_sub_agent_async"]


AGENT_ROSTER = {
    "mathematician": "Expert in solving complex mathematical problems and plotting functions.",
    "researcher": "Skilled in gathering and synthesizing information from various sources.",
    "writer": "Proficient in crafting clear and engaging written content on a wide range of topics.",
}