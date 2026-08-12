"""Parallel pipeline: every ready step of a layer runs concurrently via ``Send``.

Registered automatically as the graph named ``"parallel"`` (see
``graphs/__init__.py``) — the module name minus its ``_pipeline_graph`` suffix.
"""

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agents.agent_states import AgentState, _transitive_dependents
from agents.orchestrator_node import make_orchestrator_agent
from agents.sub_agents_nodes import make_parallel_sub_agent_node
from assemble_node import assemble_node

GRAPH_DESCRIPTION = "Fan out every ready step of a dependency layer in parallel (default)"


def fan_out_router(state: dict):
    """
    After orchestration, dispatch ALL independent steps in parallel via Send.
    Steps with depends_on=[1] wait until step 1 is in results (handled by
    the dependency layer grouping below).

    Distinguishes completion from deadlock (Slice 1):
    - returns ``"assemble"`` only when every planned step has a result;
    - returns ``Send`` objects when one or more unfinished steps are ready;
    - raises ``RuntimeError`` when steps are permanently blocked (no ready step
      but unfinished steps remain with dependencies that can never be satisfied).
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

    if ready:
        # Send each ready step to the sub_agent_node in parallel
        return [Send("parallel_sub_agent", {"step": s, "results": results, "current_datetime": current_datetime}) for s in ready]

    # No ready steps — distinguish "all done" from "permanently blocked"
    unfinished = [s["step"] for s in plan if s["step"] not in results]
    if unfinished:
        detail_parts = []
        for s in plan:
            if s["step"] in unfinished:
                unmet = [d for d in s.get("depends_on", []) if d not in results]
                detail_parts.append(
                    f"step {s['step']} (unmet dependencies: {unmet})"
                )
        raise RuntimeError(
            f"No step is ready to execute, but {len(unfinished)} step(s) "
            f"remain unfinished and are permanently blocked: {unfinished}. "
            f"Details: {'; '.join(detail_parts)}. "
            f"Check that every step's depends_on references valid step numbers."
        )

    return "assemble"


def scheduler_node(state: dict) -> dict:
    """Synchronisation barrier: also propagates skips from failed steps (4.13).

    Writes ``[SKIPPED — dependency failed]`` markers for every step transitively
    blocked by a failed step, so the router never dispatches them.  Also emits
    ``step_stats`` entries so every step in the plan gets a stats row (4.9).
    """
    failed = set(state.get("failed_steps", []))
    if not failed:
        return {}
    blocked = _transitive_dependents(state["plan"], failed)
    results = state.get("results", {})
    plan = state["plan"]
    return {
        "results": {s: "[SKIPPED — dependency failed]" for s in blocked if s not in results},
        "step_stats": [
            {
                "step": s,
                "agent": next((p["agent"] for p in plan if p["step"] == s), "unknown"),
                "status": "skipped",
                "duration_s": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_calls": 0,
            }
            for s in blocked if s not in results
        ],
    }


def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
    """Compile the parallel pipeline.

    Every argument defaults to the production wiring, so ``build()`` takes no
    arguments in production; tests inject fake nodes and their own checkpointer.
    """
    orchestrator = orchestrator or make_orchestrator_agent()
    sub_agent = sub_agent or make_parallel_sub_agent_node()

    builder = StateGraph(AgentState)
    # ValueError from plan validation (1.2) should re-plan, not kill the run.
    builder.add_node("orchestrator", orchestrator, retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
    # No worker RetryPolicy (Slice 3): the worker node owns the bounded
    # attempt loop configured per agent via ``execution.max_attempts``.
    builder.add_node("parallel_sub_agent",    sub_agent)
    builder.add_node("assemble",     assemble_node)
    builder.add_node("scheduler", scheduler_node)

    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")


    builder.add_edge("parallel_sub_agent", "scheduler")
    builder.add_conditional_edges("scheduler", fan_out_router, ["assemble", "parallel_sub_agent"])
    builder.add_edge("assemble", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
