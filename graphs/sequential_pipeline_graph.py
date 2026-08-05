"""Sequential pipeline: one step at a time, in dependency order.

Registered automatically as the graph named ``"sequential"`` (see
``graphs/__init__.py``) — the module name minus its ``_pipeline_graph`` suffix.

Like the parallel graph, this uses a scheduler + router pattern that dispatches
``Send`` objects with ``WorkerState`` payloads to the sub-agent node. The
difference: ``sequential_router`` sends exactly **one** step per dispatch (the
first dependency-ready step in plan order), so steps execute one after another
while still respecting ``depends_on``.
"""

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agents.agent_states import AgentState, _transitive_dependents
from agents.orchestrator_node import make_orchestrator_agent
from agents.sub_agents_nodes import make_parallel_sub_agent_node
from assemble_node import assemble_node

GRAPH_DESCRIPTION = "Run one step at a time in dependency order"


def sequential_router(state: dict):
    """Dispatch the **next** ready step via ``Send``, one at a time.

    After the sub-agent completes, it routes back to the scheduler which
    re-evaluates this router. Only when no ready steps remain do we proceed
    to assembly.
    """
    plan = state["plan"]
    results = state.get("results", {})
    current_datetime = state.get("current_datetime", "")

    # Find the first step whose dependencies are satisfied and not yet completed
    ready = [
        s for s in plan
        if s["step"] not in results
        and all(d in results for d in s.get("depends_on", []))
    ]

    if not ready:
        unfinished = [s["step"] for s in plan if s["step"] not in results]
        if unfinished:
            raise RuntimeError(
                f"No step is ready to execute, but {len(unfinished)} step(s) "
                f"remain unfinished and are permanently blocked: {unfinished}. "
                f"Check that every step's depends_on references valid step numbers."
            )
        return "assemble"

    # Send only the first ready step (sequential — unlike parallel which fans out all)
    next_step = ready[0]
    return Send(
        "sub_agent",
        {"step": next_step, "results": results, "current_datetime": current_datetime},
    )


def scheduler_node(state: dict) -> dict:
    """Synchronisation barrier: also propagates skips from failed steps (4.13).

    Writes ``[SKIPPED — dependency failed]`` markers for every step transitively
    blocked by a failed step, so the router never dispatches them.
    """
    failed = set(state.get("failed_steps", []))
    if not failed:
        return {}
    blocked = _transitive_dependents(state["plan"], failed)
    results = state.get("results", {})
    return {
        "results": {s: "[SKIPPED — dependency failed]" for s in blocked if s not in results}
    }


def should_continue(state: dict) -> str:
    """Legacy direct router (not ``Send``-based).

    Kept for backward compatibility with existing tests; the ``build()``
    function no longer wires this into the graph — it uses ``sequential_router``
    and the scheduler pattern instead.
    """
    plan = state.get("plan", [])
    results = state.get("results", {})
    if len(results) < len(plan):
        return "sub_agent"
    return "assemble"


def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
    """Compile the sequential pipeline.

    Every argument defaults to the production wiring, so ``build()`` takes no
    arguments in production; tests inject fake nodes and their own checkpointer.
    """
    orchestrator = orchestrator or make_orchestrator_agent()
    sub_agent = sub_agent or make_parallel_sub_agent_node()

    builder = StateGraph(AgentState)
    # ValueError from plan validation (1.2) should re-plan, not kill the run.
    builder.add_node(
        "orchestrator", orchestrator,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)),
    )
    builder.add_node(
        "sub_agent", sub_agent,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)),
    )
    builder.add_node("assemble", assemble_node)
    builder.add_node("scheduler", scheduler_node)

    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")
    builder.add_edge("sub_agent", "scheduler")
    builder.add_conditional_edges(
        "scheduler", sequential_router, ["assemble", "sub_agent"]
    )
    builder.add_edge("assemble", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
