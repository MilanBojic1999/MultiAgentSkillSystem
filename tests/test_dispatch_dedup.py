"""The plan 1.1 regression suite: exactly-once dispatch.

Two things are pinned here:

  * The parallel graph imports and compiles (``test_parallel_graph_module_imports``)
    — a bad path map / string edge would break this.
  * The scheduler must dispatch each step exactly once even when several steps
    join on a common dependency (the diamond). A regressed topology (conditional
    edge hung off the worker instead of the scheduler) re-dispatches
    already-completed steps.

The topology built in ``_build_graph`` MUST mirror the wiring in
``paralel_pipeline_graph.py`` (plain edges into ``scheduler``; a single
conditional edge out of it). Until Phase 3 extracts a ``build(...)`` factory,
this duplication is the price of testability — the two must change together.
"""

import asyncio
from collections import Counter

import pytest
from langgraph.graph import END, StateGraph

from agents.agent_states import AgentState
from assemble_node import assemble_node
from tests._helpers import import_parallel_or_xfail
from tests.plans import DIAMOND_PLAN, LINEAR_PLAN, WIDE_PLAN


def test_parallel_graph_module_imports():
    # Item 1.1: the parallel graph compiles and the root import-shim resolves.
    import paralel_pipeline_graph

    assert paralel_pipeline_graph.graph is not None


def _build_graph(plan, worker):
    pg = import_parallel_or_xfail()

    def stub_orchestrator(state):
        return {"plan": plan, "results": {}, "current_step": 0}

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", stub_orchestrator)
    builder.add_node("scheduler", pg.scheduler_node)
    builder.add_node("parallel_sub_agent", worker)
    builder.add_node("assemble", assemble_node)
    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")
    builder.add_conditional_edges(
        "scheduler", pg.fan_out_router, ["assemble", "parallel_sub_agent"]
    )
    builder.add_edge("parallel_sub_agent", "scheduler")
    builder.add_edge("assemble", END)
    return builder.compile()


@pytest.mark.parametrize("plan", [DIAMOND_PLAN, LINEAR_PLAN, WIDE_PLAN])
def test_each_step_dispatched_exactly_once(plan):
    calls = Counter()

    def stub_worker(state):
        n = state["step"]["step"]
        calls[n] += 1
        return {"results": {n: f"out-{n}"}}

    graph = _build_graph(plan, stub_worker)
    out = graph.invoke({"task": "t", "current_datetime": ""})

    expected = {s["step"]: 1 for s in plan}
    assert calls == expected                      # the 1.1 bug over-counts join steps
    last = max(s["step"] for s in plan)
    assert f"out-{last}" in out["final_output"]


def test_failed_step_is_contained_and_recorded():
    # Plan 1.4 containment, verified on the real worker node: a failing step is
    # recorded as a result and flagged in failed_steps (an AgentState channel as
    # of 3.1) instead of killing the run. The failing step is injected through
    # the worker factory's run_step seam rather than by patching a module global.
    from agents.sub_agents_nodes import make_parallel_sub_agent_node

    async def fake_run(step, results, current_datetime=""):
        if step["step"] == 2:
            raise RuntimeError("boom")
        return step["step"], f"out-{step['step']}"

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_graph(DIAMOND_PLAN, worker)
    # The real worker node is async, so the graph must be driven with ainvoke.
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}))

    assert out["final_output"]  # run still completes
    assert "[STEP FAILED]" in out["final_output"]
    assert 2 in out.get("failed_steps", [])
