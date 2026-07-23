from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agents.agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import sub_agent_node
from assemble_node import assemble_node


def should_continue(state: dict) -> str:
    plan = state.get("plan", [])
    results = state.get("results", {})
    if len(results) < len(plan):
        return "sub_agent"
    return "assemble"


builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent)
builder.add_node("sub_agent", sub_agent_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("assemble", assemble_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", should_continue)
builder.add_conditional_edges("sub_agent", should_continue)
builder.add_edge("assemble", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)