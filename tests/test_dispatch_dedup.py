"""The plan 1.1 regression suite: exactly-once dispatch.

Two things are pinned here:

  * The parallel graph imports and compiles (``test_parallel_graph_module_imports``)
    — a bad path map / string edge would break this.
  * The scheduler must dispatch each step exactly once even when several steps
    join on a common dependency (the diamond). A regressed topology (conditional
    edge hung off the worker instead of the scheduler) re-dispatches
    already-completed steps.

Since Phase 3.3 these run against the graph's real ``build(...)`` factory with
fake nodes injected, so the production topology itself is what gets exercised —
no mirrored wiring to keep in sync.
"""

import asyncio
from collections import Counter

import pytest

from tests._helpers import import_parallel_or_xfail
from tests.plans import DIAMOND_PLAN, LINEAR_PLAN, WIDE_PLAN

# The compiled graphs keep their default MemorySaver, which needs a thread id.
CONFIG = {"configurable": {"thread_id": "test-dispatch"}}


def test_parallel_graph_module_builds():
    # Item 1.1: the parallel graph's build() compiles with the production wiring.
    from graphs.parallel_pipeline_graph import build

    assert build() is not None


def test_root_import_shim_still_resolves():
    # Item 1.5: the misspelled root path keeps working for one more release.
    import paralel_pipeline_graph
    from graphs import parallel_pipeline_graph

    assert paralel_pipeline_graph.build is parallel_pipeline_graph.build


def _build_graph(plan, worker):
    """The real parallel graph, with the two LLM-bearing nodes faked out."""
    pg = import_parallel_or_xfail()

    def stub_orchestrator(state):
        return {"plan": plan, "results": {}, "current_step": 0}

    return pg.build(orchestrator=stub_orchestrator, sub_agent=worker)


@pytest.mark.parametrize("plan", [DIAMOND_PLAN, LINEAR_PLAN, WIDE_PLAN])
def test_each_step_dispatched_exactly_once(plan):
    calls = Counter()

    def stub_worker(state):
        n = state["step"]["step"]
        calls[n] += 1
        return {"results": {n: f"out-{n}"}}

    graph = _build_graph(plan, stub_worker)
    out = graph.invoke({"task": "t", "current_datetime": ""}, config=CONFIG)

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
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=CONFIG))

    assert out["final_output"]  # run still completes
    assert "[STEP FAILED]" in out["final_output"]
    assert 2 in out.get("failed_steps", [])


# ---------------------------------------------------------------------------
# Plan 4.12: sequential graph end-to-end tests
# ---------------------------------------------------------------------------

_SEQ_CONFIG = {"configurable": {"thread_id": "test-sequential"}}


def _build_sequential_graph(plan, worker):
    """The real sequential graph, with the two LLM-bearing nodes faked out."""
    from graphs.sequential_pipeline_graph import build

    def stub_orchestrator(state):
        return {"plan": plan, "results": {}, "current_step": 0}

    return build(orchestrator=stub_orchestrator, sub_agent=worker)


@pytest.mark.parametrize("plan", [LINEAR_PLAN, DIAMOND_PLAN])
def test_sequential_graph_each_step_runs_exactly_once_invoke(plan):
    """Under invoke, the sequential graph executes every step exactly once."""
    calls = Counter()

    def stub_worker(state):
        n = state["step"]["step"]
        calls[n] += 1
        return {"results": {n: f"out-{n}"}}

    graph = _build_sequential_graph(plan, stub_worker)
    out = graph.invoke({"task": "t", "current_datetime": ""}, config=_SEQ_CONFIG)

    expected = {s["step"]: 1 for s in plan}
    assert calls == expected
    last = max(s["step"] for s in plan)
    assert f"out-{last}" in out["final_output"]


@pytest.mark.parametrize("plan", [LINEAR_PLAN, DIAMOND_PLAN])
def test_sequential_graph_each_step_runs_exactly_once_ainvoke(plan):
    """Under ainvoke, the sequential graph executes every step exactly once."""
    calls = Counter()

    async def fake_run(step, results, current_datetime=""):
        n = step["step"]
        calls[n] += 1
        return n, f"out-{n}"

    from agents.sub_agents_nodes import make_parallel_sub_agent_node
    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_sequential_graph(plan, worker)
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=_SEQ_CONFIG))

    expected = {s["step"]: 1 for s in plan}
    assert calls == expected
    last = max(s["step"] for s in plan)
    assert f"out-{last}" in out["final_output"]
