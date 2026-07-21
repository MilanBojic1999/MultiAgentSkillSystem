from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import run_sub_agent_async
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


async def parallel_sub_agent_node(state: dict) -> dict:
    try:
        step_num, output = await run_sub_agent_async(state["step"], state["results"], state.get("current_datetime", ""))

        return {"results": {step_num: output}}
    except Exception as e:
        log_event("sub_agent_step_failed", step=state["step"]["step"], error=str(e))
        return {"results": {state["step"]["step"]: f"[STEP FAILED] {e}"},
                "failed_steps": [state["step"]["step"]]}

def scheduler_node(state: dict) -> dict:
    return {}

builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("parallel_sub_agent",    parallel_sub_agent_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("assemble",     assemble_node)
builder.add_node("scheduler", scheduler_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", "scheduler")
builder.add_conditional_edges("parallel_sub_agent", "scheduler")
builder.add_conditional_edges("scheduler", fan_out_router, {"assemble": "assemble", Send: "parallel_sub_agent"})
builder.add_edge("assemble", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)