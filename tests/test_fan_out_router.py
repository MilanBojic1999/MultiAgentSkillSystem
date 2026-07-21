"""paralel_pipeline_graph.fan_out_router — routing as a pure function.

Every test here xfails until Bug 1 (the module fails to import) is fixed; see
tests/_helpers.import_parallel_or_xfail.
"""

import pytest
from langgraph.types import Send

from tests._helpers import import_parallel_or_xfail
from tests.plans import DIAMOND_PLAN, step


def _route(results):
    pg = import_parallel_or_xfail()
    return pg.fan_out_router(
        {"plan": DIAMOND_PLAN, "results": results, "current_datetime": "now"}
    )


def _sent_steps(sends):
    return sorted(s.arg["step"]["step"] for s in sends)


def test_first_layer_dispatches_both_independent_steps():
    out = _route({})
    assert all(isinstance(s, Send) and s.node == "parallel_sub_agent" for s in out)
    assert _sent_steps(out) == [1, 2]


def test_second_layer_dispatches_the_join_step():
    out = _route({1: "a", 2: "b"})
    assert _sent_steps(out) == [3]


def test_partial_layer_dispatches_only_ready_step():
    out = _route({1: "a"})
    assert _sent_steps(out) == [2]  # step 3 still blocked on step 2


def test_all_done_routes_to_assemble():
    assert _route({1: "a", 2: "b", 3: "c"}) == "assemble"


def test_send_payload_carries_expected_keys():
    out = _route({})
    payload = out[0].arg
    assert set(payload) >= {"step", "results", "current_datetime"}


@pytest.mark.xfail(
    strict=False,
    reason="plan 1.2 blocked-forever guard not yet implemented; today returns 'assemble'",
)
def test_unsatisfiable_plan_raises_runtime_error():
    pg = import_parallel_or_xfail()
    blocked = [step(1), step(2, deps=[99])]  # step 2 can never run
    with pytest.raises(RuntimeError):
        pg.fan_out_router({"plan": blocked, "results": {1: "a"}, "current_datetime": ""})
