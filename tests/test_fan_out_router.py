"""graphs.parallel_pipeline_graph.fan_out_router — routing as a pure function.

The module-import guard in tests/_helpers.import_parallel_or_xfail is now a
passthrough (Bug 1 is fixed); it stays only as a safety net.
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


def test_unsatisfiable_plan_raises_runtime_error():
    """Slice 1: permanently blocked state raises instead of silently assembling.

    Step 2 depends on step 99, which is not in the plan, so it can never run.
    The router must raise RuntimeError naming the blocked step and its unmet
    dependencies rather than routing to ``assemble``.
    """
    pg = import_parallel_or_xfail()
    blocked = [step(1), step(2, deps=[99])]  # step 2 can never run
    with pytest.raises(RuntimeError, match=r"permanently blocked") as excinfo:
        pg.fan_out_router({"plan": blocked, "results": {1: "a"}, "current_datetime": ""})
    # Error text names the blocked step numbers and their unmet dependencies
    assert "2" in str(excinfo.value)
    assert "99" in str(excinfo.value)


def test_blocked_error_names_every_unfinished_step():
    """Error text must name ALL blocked step numbers, not just one.

    Steps 2 and 3 each depend on a non-existent step, so both are permanently
    blocked; the error must identify both.
    """
    pg = import_parallel_or_xfail()
    blocked = [step(1), step(2, deps=[99]), step(3, deps=[98])]
    with pytest.raises(RuntimeError) as excinfo:
        pg.fan_out_router({"plan": blocked, "results": {1: "a"}, "current_datetime": ""})
    message = str(excinfo.value)
    assert "2" in message
    assert "3" in message
    assert "99" in message
    assert "98" in message


def test_deadlock_cycle_raises_instead_of_assembling():
    """A cyclic dependency (1↔2) with partial results must raise, not assemble.

    Step 3 completed, but steps 1 and 2 each wait on the other: permanently
    blocked state the router must not present as successful completion.
    """
    pg = import_parallel_or_xfail()
    cyclic = [step(1, deps=[2]), step(2, deps=[1]), step(3)]
    with pytest.raises(RuntimeError, match="permanently blocked"):
        pg.fan_out_router({"plan": cyclic, "results": {3: "c"}, "current_datetime": ""})
