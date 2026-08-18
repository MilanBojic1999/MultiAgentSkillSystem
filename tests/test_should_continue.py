"""Tests for ``sequential_router`` (the active ``Send``-based router) and the
legacy ``should_continue`` (kept for backward compatibility).

The new ``sequential_router`` dispatches a single ``Send`` with a
``WorkerState`` payload — mirroring the parallel graph's ``fan_out_router``
but sending only one step per dispatch.
"""

import asyncio

import pytest
from langgraph.types import Send

from graphs.sequential_pipeline_graph import sequential_router, should_continue
from tests.plans import LINEAR_PLAN, DIAMOND_PLAN, step


# ---------------------------------------------------------------------------
# sequential_router (active, Send-based)
# ---------------------------------------------------------------------------

def _route(results, plan=None, current_datetime="now"):
    return sequential_router(
        {"plan": plan or DIAMOND_PLAN, "results": results, "current_datetime": current_datetime}
    )


def test_first_ready_step_dispatched():
    """With no results, the first step (step 1) is dispatched."""
    out = _route({})
    assert isinstance(out, Send)
    assert out.node == "sub_agent"
    assert out.arg["step"]["step"] == 1


def test_second_step_dispatched_after_first_completes():
    """After step 1 completes, the next ready step is dispatched."""
    out = _route({1: "a"})
    assert isinstance(out, Send)
    assert out.arg["step"]["step"] == 2


def test_join_step_dispatched_when_all_deps_met():
    """In a diamond plan, step 3 only dispatches after both 1 and 2 complete."""
    out = _route({1: "a", 2: "b"})
    assert isinstance(out, Send)
    assert out.arg["step"]["step"] == 3


def test_all_done_routes_to_assemble():
    assert _route({1: "a", 2: "b", 3: "c"}) == "assemble"


def test_send_payload_carries_expected_keys():
    out = _route({})
    payload = out.arg
    assert set(payload) >= {"step", "results", "current_datetime"}


def test_linear_plan_dispatches_in_order():
    """Linear plan [1→2→3]: step 2 only dispatched after step 1."""
    out = _route({}, plan=LINEAR_PLAN)
    assert out.arg["step"]["step"] == 1
    out = _route({1: "a"}, plan=LINEAR_PLAN)
    assert out.arg["step"]["step"] == 2
    out = _route({1: "a", 2: "b"}, plan=LINEAR_PLAN)
    assert out.arg["step"]["step"] == 3


def test_partial_deps_only_dispatches_unblocked():
    """Step with unmet dependency is skipped; first unblocked step wins."""
    out = _route({1: "a"})  # diamond: step 2 has no deps, step 3 blocked on 2
    assert out.arg["step"]["step"] == 2


def test_unsatisfiable_plan_raises_runtime_error():
    """A step depending on a step that doesn't exist should raise."""
    blocked = [step(1), step(2, deps=[99])]
    with pytest.raises(RuntimeError, match="permanently blocked"):
        sequential_router(
            {"plan": blocked, "results": {1: "a"}, "current_datetime": ""}
        )


# ---------------------------------------------------------------------------
# legacy should_continue (kept for backward compat)
# ---------------------------------------------------------------------------

def test_incomplete_results_continue_to_sub_agent():
    assert should_continue({"plan": LINEAR_PLAN, "results": {1: "a"}}) == "sub_agent"


def test_complete_results_route_to_assemble():
    results = {1: "a", 2: "b", 3: "c"}
    assert should_continue({"plan": LINEAR_PLAN, "results": results}) == "assemble"


def test_blocked_forever_sub_agent_node_is_contained():
    """Tests make_sub_agent_node directly — failure containment (Phase 4.13).

    The node receives a step whose dependencies can never be satisfied
    (referencing a non-existent step 99). The blocked-forever guard raises
    RuntimeError, which the containment wrapper catches and records as a
    ``[STEP FAILED]`` result + ``failed_steps`` entry instead of killing the run.
    """
    from agents.sub_agents_nodes import make_sub_agent_node

    node = make_sub_agent_node()
    state = {
        "step": step(1, deps=[99]),
        "results": {},
        "current_datetime": "",
    }
    result = asyncio.run(node.ainvoke(state))
    assert 1 in result.get("results", {})
    assert "[STEP FAILED]" in result["results"][1]
    assert 1 in result.get("failed_steps", [])


def test_already_completed_step_returns_reducer_compatible_state():
    """Slice 1: the sequential worker's duplicate-result branch is a no-op.

    If a worker receives a step already present in ``results`` (defensive
    path — the router should never dispatch it), it must NOT return the raw
    output string under ``results`` (a scalar would break the state reducer
    ``lambda a, b: {**a, **b}``) and must NOT emit a second stats row.
    """
    from agents.sub_agents_nodes import make_sub_agent_node

    calls: list[int] = []

    async def fake_run(step, results, current_datetime=""):
        calls.append(step["step"])
        return step["step"], "out-1", {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0}

    node = make_sub_agent_node(run_step=fake_run)
    result = asyncio.run(
        node.ainvoke({
            "step": step(1),
            "results": {1: "out-1"},
            "current_datetime": "",
        })
    )

    # The step must not be re-executed…
    assert calls == []

    # …and the returned state must be reducer-compatible: no scalar under
    # ``results``, no duplicate stats row.
    assert "results" not in result or isinstance(result["results"], dict)
    assert "step_stats" not in result or result["step_stats"] == []
