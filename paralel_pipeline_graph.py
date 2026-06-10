from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import run_sub_agent_async
from skill_loader import load_skills

def fan_out_router(state: dict):
    """
    After orchestration, dispatch ALL independent steps in parallel via Send.
    Steps with depends_on=[1] wait until step 1 is in results (handled by
    the dependency layer grouping below).
    """

    plan    = state["plan"]
    results = state.get("results", {})
    skill_index = state["skill_index"]
    skill_dictionary_pairs = state["skill_dictionary_pairs"]
    # Find all steps whose dependencies are satisfied
    ready = [
        s for s in plan
        if s["step"] not in results
        and all(d in results for d in s.get("depends_on", []))
    ]

    if not ready:
        return "assemble"

    # Send each ready step to the sub_agent_node in parallel
    return [Send("parallel_sub_agent", {"step": s, "results": results, "skill_index":skill_index, "skill_dictionary_pairs":skill_dictionary_pairs}) for s in ready]


async def parallel_sub_agent_node(state: dict) -> dict:
    step_num, output = await run_sub_agent_async(state["step"], state["skill_index"], state["skill_dictionary_pairs"], state["results"])

    return {"results": {step_num: output}}

def assemble_node(state: dict) -> dict:
    plan = state.get("plan", [])
    results = state.get("results", {})
    parts   = [f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
               for s in plan]
    return {"final_output": "\n\n".join(parts)}

builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent)
builder.add_node("parallel_sub_agent",    parallel_sub_agent_node)
builder.add_node("assemble",     assemble_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", fan_out_router, {"assemble": "assemble", Send: "parallel_sub_agent"})
builder.add_conditional_edges("parallel_sub_agent", fan_out_router, {"assemble": "assemble", Send: "parallel_sub_agent"})
builder.add_edge("assemble", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)