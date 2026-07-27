"""graphs.sequential_pipeline_graph.should_continue and the worker's dead-state
contract.

We assert on the two pure functions directly rather than invoking the compiled
sequential graph — the blocked-forever case would otherwise loop until the
recursion limit.
"""

import asyncio

import pytest

from graphs.sequential_pipeline_graph import should_continue
from tests.plans import LINEAR_PLAN, step


def test_incomplete_results_continue_to_sub_agent():
    assert should_continue({"plan": LINEAR_PLAN, "results": {1: "a"}}) == "sub_agent"


def test_complete_results_route_to_assemble():
    results = {1: "a", 2: "b", 3: "c"}
    assert should_continue({"plan": LINEAR_PLAN, "results": results}) == "assemble"


@pytest.mark.xfail(
    strict=False,
    reason="plan 1.2 blocked-forever guard not yet implemented; sub_agent_node returns {}",
)
def test_blocked_forever_sub_agent_node_raises():
    # No step is ready (step 1 depends on an absent step) and results are
    # incomplete: today the worker returns {}, which makes should_continue loop
    # until the recursion limit. The desired contract is a RuntimeError.
    # Only reachable because no step is ready — so run_step is never called and
    # no live LLM call happens.
    from agents.sub_agents_nodes import make_sub_agent_node

    node = make_sub_agent_node()
    state = {"plan": [step(1, deps=[99])], "results": {}, "current_datetime": ""}
    with pytest.raises(RuntimeError):
        asyncio.run(node(state))
