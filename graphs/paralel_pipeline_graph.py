from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agents.agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import parallel_sub_agent_node
from assemble_node import assemble_node

from utils.logger import log_event


def fan_out_router(state: dict):
    """
    After orchestration, dispatch ALL independent steps in parallel via Send.
    Steps with depends_on=[1] wait until step 1 is in results (handled by
    the dependency layer grouping below).
    """

    plan    = state["plan"]
    results = state.get("results", {})
    current_datetime = state.get("current_datetime", "")
    # Find all steps whose dependencies are satisfied
    ready = [
        s for s in plan
        if s["step"] not in results
        and all(d in results for d in s.get("depends_on", []))
    ]

    if not ready:
        return "assemble"

    # Send each ready step to the sub_agent_node in parallel
    return [Send("parallel_sub_agent", {"step": s, "results": results, "current_datetime": current_datetime}) for s in ready]


def scheduler_node(state: dict) -> dict:
    return {}

builder = StateGraph(AgentState)
# ValueError from plan validation (1.2) should re-plan, not kill the run.
builder.add_node("orchestrator", orchestrator_agent, retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
builder.add_node("parallel_sub_agent",    parallel_sub_agent_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("assemble",     assemble_node)
builder.add_node("scheduler", scheduler_node)

builder.set_entry_point("orchestrator")
builder.add_edge("orchestrator", "scheduler")
# Workers route back through the scheduler via a PLAIN edge: plain-node routing
# deduplicates, so fan_out_router runs exactly once per dependency layer even
# when several parallel steps finish in the same superstep (item 1.1). The
# conditional edge below is the ONLY conditional edge; its path map is a plain
# list of target node names (Send dispatches bypass it).
builder.add_edge("parallel_sub_agent", "scheduler")
builder.add_conditional_edges("scheduler", fan_out_router, ["assemble", "parallel_sub_agent"])
builder.add_edge("assemble", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)