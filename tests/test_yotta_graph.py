"""graphs.yotta_graph — Send fan-out, verification routing, and e2e behavior.

Follows tests/test_execution_attempts.py / tests/test_fan_out_router.py:
a stub orchestrator injects the plan, and ``graphs.yotta_graph.run_sub_agent_async``
is monkeypatched to fake the verifier/writer LLM calls. The worker uses its
PRODUCTION runner, so ``agents.sub_agents_nodes.run_sub_agent_async`` is
patched too — the factory captures it inside ``build()``, which is why the
patch is installed before building.
"""

import asyncio
import json

import pytest
from langgraph.types import Send

from agents.agent_states import RESULTS_RESET, YottaState
from graphs.yotta_graph import _latest_retry_note, after_verify, fan_out_router
from tests.plans import DIAMOND_PLAN, step

CONFIG = {"configurable": {"thread_id": "test-yotta-graph"}}

_STATS = {"input_tokens": 10, "output_tokens": 10, "tool_calls": 0}


class FakeLLMCall:
    """Scripted ``run_sub_agent_async`` covering all three call sites.

    - workers (numeric step): return ``(step, "out-<n>", stats)``, recording
      the step and the ``feedback`` kwarg the production worker binds (F2),
      plus — in ``worker_files`` — the step's own ``files`` assignment and the
      ``files`` kwarg carrying the run's attached documents;
    - verifier (step ``"verify"``): replay the scripted verdict lists, one
      per verify pass;
    - writer (step ``"assemble"``): return ``"FINAL"``, recording the subtask
      so tests can inspect the writer's prompt blocks (direct route).
    """

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.worker_calls: list[tuple[int, str]] = []
        # (step number, the step's own ``files`` assignment, the ``files`` kwarg)
        self.worker_files: list[tuple[int, list[str], dict[str, str]]] = []
        self.writer_subtask = ""
        self.verifier_subtasks: list[str] = []

    async def __call__(self, step, results, current_datetime="", **kwargs):
        if step["step"] == "verify":
            self.verifier_subtasks.append(step["subtask"])
            payload = self.verdicts.pop(0) if self.verdicts else []
            return step["step"], json.dumps(payload), _STATS
        if step["step"] == "assemble":
            self.writer_subtask = step["subtask"]
            return step["step"], "FINAL", _STATS
        self.worker_calls.append((step["step"], kwargs.get("feedback", "")))
        self.worker_files.append(
            (step["step"], step.get("files", []), kwargs.get("files") or {})
        )
        return step["step"], f"out-{step['step']}", _STATS


def _build_yotta(monkeypatch, fake, orchestrator=None):
    """The real yotta graph with every LLM-bearing call faked out.

    ``fake`` replaces ``run_sub_agent_async`` in BOTH modules: the graph
    module's global (verify/writer look it up at call time) and
    ``agents.sub_agents_nodes`` (the production worker's default runner is
    captured by ``_resolve_run_step`` when the factory runs — inside
    ``build`` — so the patch must be installed before that).
    """
    monkeypatch.setattr("agents.sub_agents_nodes.run_sub_agent_async", fake)
    monkeypatch.setattr("graphs.yotta_graph.run_sub_agent_async", fake)

    def stub_orchestrator(state):
        return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}

    from graphs.yotta_graph import build

    return build(orchestrator=orchestrator or stub_orchestrator)


def _invoke(graph, **extra):
    state = {"task": "t", "current_datetime": "", **extra}
    return asyncio.run(graph.ainvoke(state, config=CONFIG))


# ---------------------------------------------------------------------------
# fan_out_router — routing as a pure function
# ---------------------------------------------------------------------------

def _route(results, plan=DIAMOND_PLAN, pending_retries=None, **extra):
    state = {"plan": plan, "results": results, "current_datetime": "now",
             "pending_retries": pending_retries or [], **extra}
    return fan_out_router(state)


def _sent_steps(sends):
    return sorted(s.arg["step"]["step"] for s in sends)


def test_router_empty_plan_routes_to_writer():
    """Direct route: yotta search results were already sufficient."""
    assert fan_out_router({"plan": [], "results": {}, "current_datetime": ""}) == "writer"


def test_router_first_layer_dispatches_both_independent_steps():
    out = _route({})
    assert all(isinstance(s, Send) and s.node == "sub_agent" for s in out)
    assert _sent_steps(out) == [1, 2]


def test_router_send_payload_carries_expected_keys():
    out = _route({}, files={"a.txt": "doc"})
    payload = out[0].arg
    assert set(payload) >= {"step", "results", "current_datetime",
                            "step_verifications", "streaming", "files"}
    # a Send payload is the only state a worker sees — documents must ride along
    assert payload["files"] == {"a.txt": "doc"}


def test_router_send_payload_files_default_to_empty_dict():
    assert _route({})[0].arg["files"] == {}


def test_router_second_layer_dispatches_the_join_step():
    out = _route({1: "a", 2: "b"})
    assert _sent_steps(out) == [3]


def test_router_pending_retries_redispatch_even_with_result_present():
    """A retried step is dispatched even though its old result is in results."""
    out = _route({1: "a", 2: "old", 3: "c"}, pending_retries=[2])
    assert _sent_steps(out) == [2]


def test_router_all_done_routes_to_verify():
    assert _route({1: "a", 2: "b", 3: "c"}) == "verify"


def test_router_unsatisfiable_plan_raises_runtime_error():
    blocked = [step(1), step(2, deps=[99])]  # step 2 can never run
    with pytest.raises(RuntimeError, match=r"permanently blocked") as excinfo:
        _route({1: "a"}, plan=blocked)
    assert "2" in str(excinfo.value)
    assert "99" in str(excinfo.value)


# ---------------------------------------------------------------------------
# after_verify — routing table
# ---------------------------------------------------------------------------

def _after(route="proceed", replan_count=0):
    return after_verify({"verification_route": route, "replan_count": replan_count})


def test_after_verify_retry_routes_to_scheduler_even_at_replan_cap():
    """Retry is never gated by the replan cap (fixes F4)."""
    assert _after(route="retry", replan_count=3) == "scheduler"


def test_after_verify_replan_under_cap_routes_to_orchestrator():
    assert _after(route="replan", replan_count=0) == "orchestrator"
    assert _after(route="replan", replan_count=2) == "orchestrator"


def test_after_verify_replan_at_cap_routes_to_writer():
    assert _after(route="replan", replan_count=3) == "writer"


def test_after_verify_proceed_routes_to_writer():
    assert _after(route="proceed") == "writer"
    assert _after(route="proceed", replan_count=3) == "writer"


# ---------------------------------------------------------------------------
# _latest_retry_note — F2: only the most recent [Retry k/2] line
# ---------------------------------------------------------------------------

def test_latest_retry_note_only_last_retry_line_survives():
    notes = "First output\n[Retry 1/2] fix A\n[Retry 2/2] fix B"
    assert _latest_retry_note({3: {"notes": notes}}, 3) == "[Retry 2/2] fix B"


def test_latest_retry_note_empty_without_retry_lines():
    assert _latest_retry_note({3: {"notes": "no retries here"}}, 3) == ""
    assert _latest_retry_note({}, 3) == ""
    assert _latest_retry_note({3: {"notes": ""}}, 3) == ""


# ---------------------------------------------------------------------------
# YottaState.results reducer — sentinel clears, ordinary merges accumulate
# ---------------------------------------------------------------------------

def test_yotta_results_reducer_sentinel_clears_and_merges_accumulate():
    reducer = YottaState.__annotations__["results"].__metadata__[0]
    assert reducer({1: "a", 2: "b"}, {RESULTS_RESET: ""}) == {}
    assert reducer({1: "a"}, {2: "b"}) == {1: "a", 2: "b"}


# ---------------------------------------------------------------------------
# Compiled e2e — happy path, retry path, replan path, direct route
# ---------------------------------------------------------------------------

def test_e2e_happy_path_parallel_execution_then_writer(monkeypatch):
    fake = FakeLLMCall(verdicts=[[
        {"step": 1, "verification_result": "PASSED", "notes": ""},
        {"step": 2, "verification_result": "PASSED", "notes": ""},
        {"step": 3, "verification_result": "PASSED", "notes": ""},
    ]])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, search_results="sr")

    assert out["final_output"] == "FINAL"
    # steps 1 and 2 run in parallel, then the join step 3 — each exactly once
    assert sorted(s for s, _ in fake.worker_calls) == [1, 2, 3]
    assert out["verification_result"] == "PASSED"
    assert out["pending_retries"] == []
    assert out["results"] == {1: "out-1", 2: "out-2", 3: "out-3"}


def test_e2e_worker_receives_attached_documents(monkeypatch):
    """Both severed links, end to end: a step's ``files`` assignment survives
    into the plan, and the run's documents reach the worker through the Send
    payload (the worker used to be called with no ``files`` at all)."""
    plan = [step(1, files=["a.txt"]), step(2)]

    def files_orchestrator(state):
        return {"plan": plan, "results": {}, "current_step": 0}

    fake = FakeLLMCall(verdicts=[[
        {"step": 1, "verification_result": "PASSED", "notes": ""},
        {"step": 2, "verification_result": "PASSED", "notes": ""},
    ]])
    graph = _build_yotta(monkeypatch, fake, orchestrator=files_orchestrator)
    out = _invoke(graph, search_results="sr", files={"a.txt": "doc text"})

    received = {n: (assigned, docs) for n, assigned, docs in fake.worker_files}
    # every worker is handed the run's whole document channel ...
    assert received[1][1] == {"a.txt": "doc text"}
    assert received[2][1] == {"a.txt": "doc text"}
    # ... and run_sub_agent_async narrows it to the step's own assignment,
    # which now survives validation (step 2 was assigned no documents)
    assert received[1][0] == ["a.txt"]
    assert received[2][0] == []
    assert out["final_output"] == "FINAL"


def test_e2e_retry_reruns_failed_step_exactly_once_with_feedback(monkeypatch):
    """One FAILED verdict → the step is re-executed exactly once (via
    ``pending_retries``, even though its old result is in results) and the
    re-execution carries the latest ``[Retry 1/2]`` note as feedback (F2)."""
    fake = FakeLLMCall(verdicts=[
        [
            {"step": 1, "verification_result": "PASSED", "notes": ""},
            {"step": 2, "verification_result": "FAILED", "notes": "recheck the numbers"},
            {"step": 3, "verification_result": "PASSED", "notes": ""},
        ],
        [
            {"step": 1, "verification_result": "PASSED", "notes": ""},
            {"step": 2, "verification_result": "PASSED", "notes": ""},
            {"step": 3, "verification_result": "PASSED", "notes": ""},
        ],
    ])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, search_results="sr")

    calls = [s for s, _ in fake.worker_calls]
    assert calls.count(2) == 2      # initial + exactly one re-execution
    assert calls.count(1) == 1
    assert calls.count(3) == 1
    retry_feedback = [fb for s, fb in fake.worker_calls if s == 2]
    assert any("[Retry 1/2]" in fb for fb in retry_feedback)
    assert out["final_output"] == "FINAL"
    assert out["pending_retries"] == []            # cleared after the retry pass
    assert out["step_verifications"][2]["retries"] == 1


def test_e2e_replan_wipes_results_and_routes_to_writer_at_cap(monkeypatch):
    """Every verify pass demands a replan: the orchestrator runs again, the
    reset sentinel wipes results so workers re-run, and at the replan cap the
    run falls through to the writer."""
    orchestrator_calls = {"n": 0}

    def counting_orchestrator(state):
        orchestrator_calls["n"] += 1
        return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}

    replan_verdicts = [
        {"step": 1, "verification_result": "FAILED", "notes": "REPLAN: add a step"},
        {"step": 2, "verification_result": "PASSED", "notes": ""},
        {"step": 3, "verification_result": "PASSED", "notes": ""},
    ]
    fake = FakeLLMCall(verdicts=[list(replan_verdicts)] * 3)
    graph = _build_yotta(monkeypatch, fake, orchestrator=counting_orchestrator)
    out = _invoke(graph, search_results="sr")

    # initial plan + 2 replan passes = 3 orchestrator calls, then the cap
    # routes to the writer instead of a fourth pass
    assert orchestrator_calls["n"] == 3
    # results were wiped on each replan — every step re-ran 3 times
    assert sorted(s for s, _ in fake.worker_calls) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert out["replan_count"] == 3
    assert out["final_output"] == "FINAL"


def test_e2e_direct_route_empty_plan_skips_agents(monkeypatch):
    """Empty plan → writer directly: no sub-agents ran, and the writer sees
    the search results and the raw attached files (its own grounding)."""
    fake = FakeLLMCall(verdicts=[])

    def empty_orchestrator(state):
        return {"plan": [], "results": {}, "current_step": 0}

    graph = _build_yotta(monkeypatch, fake, orchestrator=empty_orchestrator)
    out = _invoke(graph, search_results="good findings",
                  files={"a.txt": "doc text"})

    assert out["final_output"] == "FINAL"
    assert fake.worker_calls == []                 # no sub-agents ran
    assert "good findings" in fake.writer_subtask
    assert "doc text" in fake.writer_subtask       # files via render_files_block


def test_e2e_empty_plan_after_replan_skips_second_pass(monkeypatch):
    """The verifier demands a replan, but the orchestrator's replan pass
    returns an empty plan (search results were sufficient after all): the
    wiped results stay empty, no second worker pass runs, and the writer
    falls through to the direct route."""
    orchestrator_calls = {"n": 0}

    def plan_then_empty_orchestrator(state):
        orchestrator_calls["n"] += 1
        if orchestrator_calls["n"] == 1:
            return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}
        return {"plan": [], "results": {}, "current_step": 0}

    fake = FakeLLMCall(verdicts=[
        [
            {"step": 1, "verification_result": "FAILED", "notes": "REPLAN: add a step"},
            {"step": 2, "verification_result": "PASSED", "notes": ""},
            {"step": 3, "verification_result": "PASSED", "notes": ""},
        ],
    ])
    graph = _build_yotta(monkeypatch, fake, orchestrator=plan_then_empty_orchestrator)
    out = _invoke(graph, search_results="good findings")

    assert orchestrator_calls["n"] == 2                 # initial plan + replan pass
    assert sorted(s for s, _ in fake.worker_calls) == [1, 2, 3]   # one pass only
    assert out["replan_count"] == 1
    assert out["results"] == {}                         # wiped, nothing re-ran
    assert out["final_output"] == "FINAL"
    assert "good findings" in fake.writer_subtask       # direct-route grounding
    assert "## Sub-agent results" not in fake.writer_subtask
    assert "## Original plan" not in fake.writer_subtask
