from langgraph.graph import StateGraph, END
from langgraph.types import Send
from agent_states import AgentState
from agents.orchestrator_node import orchestrator_node
from agents.sub_agents_nodes import run_sub_agent_async
from skill_loader import load_skills

skill_index, skill_dictionary_pairs = load_skills()

def fan_out_router(state: dict):
    """
    After orchestration, dispatch ALL independent steps in parallel via Send.
    Steps with depends_on=[1] wait until step 1 is in results (handled by
    the dependency layer grouping below).
    """
    plan    = state["plan"]
    results = state.get("results", {})

    # Find all steps whose dependencies are satisfied
    ready = [
        s for s in plan
        if s["step"] not in results
        and all(d in results for d in s.get("depends_on", []))
    ]

    if not ready:
        return "assemble"

    # Send each ready step to the sub_agent_node in parallel
    return [Send("parallel_sub_agent", {"step": s, "results": results}) for s in ready]


async def parallel_sub_agent_node(state: dict) -> dict:
    step_num, output = await run_sub_agent_async(state["step"], skill_index, skill_dictionary_pairs, state["results"])

    return {"results": {step_num: output}}

def assemble_node(state: dict) -> dict:
    plan = state.get("plan", [])
    results = state.get("results", {})
    parts   = [f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
               for s in plan]
    return {"final_output": "\n\n".join(parts)}

builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("parallel_sub_agent",    parallel_sub_agent_node)
builder.add_node("assemble",     assemble_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", fan_out_router, path_map={"assemble": "assemble"})
builder.add_conditional_edges("parallel_sub_agent", fan_out_router, path_map={"assemble": "assemble"})
builder.add_edge("assemble", END)

graph = builder.compile()