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
    """Plan 1.4 + 4.13: a failing step is contained; its dependents are skipped.

    Diamond plan where step 1 always throws:
    - Step 1 → ``[STEP FAILED]`` marker
    - Step 2 → real result (independent, no deps on step 1)
    - Step 3 (depends_on=[1, 2]) → ``[SKIPPED — dependency failed]`` because
      step 1 is failed — the scheduler writes the skip marker BEFORE the router
      runs, so step 3 is **never dispatched** to the worker.
    - ``final_output`` carries the warning header from ``assemble_node``.
    """
    from agents.sub_agents_nodes import make_parallel_sub_agent_node

    calls: list[int] = []

    async def fake_run(step, results, current_datetime=""):
        calls.append(step["step"])
        if step["step"] == 1:
            raise RuntimeError("boom")
        return step["step"], f"out-{step['step']}"

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_graph(DIAMOND_PLAN, worker)
    # The real worker node is async, so the graph must be driven with ainvoke.
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=CONFIG))

    # Step 3 must never reach the worker — the scheduler skips it before dispatch
    assert 3 not in calls, "blocked step 3 was dispatched to worker"

    # Step 1: failed and recorded
    assert "[STEP FAILED]" in out["final_output"]
    assert 1 in out.get("failed_steps", [])

    # Step 2: succeeded (independent of the failed step)
    assert "out-2" in out["final_output"]

    # Step 3: transitively skipped because its dependency (step 1) failed
    assert "[SKIPPED — dependency failed]" in out["final_output"]

    # Warning header present
    assert "⚠️" in out["final_output"]
    assert "1 of 3 steps failed" in out["final_output"]
    assert "steps 1" in out["final_output"]
    assert "output below is partial" in out["final_output"]


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


def test_sequential_graph_failure_containment_and_skip_propagation():
    """Plan 4.13: sequential graph also contains failures and skips dependents.

    Linear plan [1→2→3] where step 1 throws:
    - Step 1 → ``[STEP FAILED]``
    - Step 2 (depends_on=[1]) → ``[SKIPPED — dependency failed]``
    - Step 3 (depends_on=[2]) → ``[SKIPPED — dependency failed]`` (transitive)
    - Warning header present.
    """
    from agents.sub_agents_nodes import make_parallel_sub_agent_node

    calls: list[int] = []

    async def fake_run(step, results, current_datetime=""):
        calls.append(step["step"])
        if step["step"] == 1:
            raise RuntimeError("boom")
        return step["step"], f"out-{step['step']}"

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_sequential_graph(LINEAR_PLAN, worker)
    out = asyncio.run(
        graph.ainvoke({"task": "t", "current_datetime": ""}, config=_SEQ_CONFIG)
    )

    # Only step 1 was dispatched; steps 2 and 3 were skipped by the scheduler
    assert calls == [1], f"expected only step 1 to be dispatched, got {calls}"

    assert "[STEP FAILED]" in out["final_output"]
    assert 1 in out.get("failed_steps", [])

    assert "[SKIPPED — dependency failed]" in out["final_output"]
    assert "⚠️" in out["final_output"]
    assert "1 of 3 steps failed" in out["final_output"]
