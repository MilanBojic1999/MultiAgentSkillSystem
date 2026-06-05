from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import sub_agent_node


def should_continue(state: dict) -> str:
    plan = state.get("plan", [])
    results = state.get("results", {})
    if len(results) < len(plan):
        return "sub_agent"
    return "assemble"


def assemble_node(state: dict) -> dict:
    plan = state.get("plan", [])
    results = state.get("results", {})
    parts = [
        f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
        for s in plan
    ]
    return {"final_output": "\n\n".join(parts)}


builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent)
builder.add_node("sub_agent", sub_agent_node)
builder.add_node("assemble", assemble_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", should_continue)
builder.add_conditional_edges("sub_agent", should_continue)
builder.add_edge("assemble", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)