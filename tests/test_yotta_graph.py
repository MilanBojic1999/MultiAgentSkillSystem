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
from langgraph.errors import NodeCancelledError

from agents.agent_states import RESULTS_RESET, YottaState
from execution_policy import ExecutionPolicy, resolve_execution_policy
from graphs.yotta_graph import (
    _carry_forward,
    _latest_retry_note,
    after_verify,
    carry_forward_node,
    fan_out_router,
)
from tests.plans import DIAMOND_PLAN, LINEAR_PLAN, step

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


def _build_yotta(monkeypatch, fake, orchestrator=None, sub_agent=None):
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

    return build(orchestrator=orchestrator or stub_orchestrator, sub_agent=sub_agent)


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
# YottaState.results reducer — sentinel replaces, ordinary merges accumulate
# ---------------------------------------------------------------------------

def test_yotta_results_reducer_sentinel_clears_and_merges_accumulate():
    reducer = YottaState.__annotations__["results"].__metadata__[0]
    assert reducer({1: "a", 2: "b"}, {RESULTS_RESET: ""}) == {}
    assert reducer({1: "a"}, {2: "b"}) == {1: "a", 2: "b"}


def test_yotta_results_reducer_sentinel_replaces_with_rest_of_update():
    """The sentinel gives the update replace semantics: the channel becomes
    the rest of that update, with the sentinel itself dropped."""
    reducer = YottaState.__annotations__["results"].__metadata__[0]
    assert reducer({1: "a", 2: "b", 3: "c"}, {RESULTS_RESET: "", 2: "b"}) == {2: "b"}


# ---------------------------------------------------------------------------
# Replan carry-forward — _carry_forward / carry_forward_node
# ---------------------------------------------------------------------------

def _verdict(n, verdict="PASSED", subtask=None, agent="researcher", **extra):
    """A step_verifications entry as verify_node records it (subtask stamped)."""
    return {"verdict": verdict, "notes": "", "retries": 0,
            "subtask": subtask if subtask is not None else f"subtask {n}",
            "agent": agent, **extra}


DIAMOND_RESULTS = {1: "out-1", 2: "out-2", 3: "out-3"}


def test_carry_forward_keeps_passed_unchanged_steps():
    svs = {n: _verdict(n) for n in (1, 2, 3)}
    kept, verifs = _carry_forward(DIAMOND_PLAN, DIAMOND_RESULTS, svs)
    assert kept == DIAMOND_RESULTS
    assert set(verifs) == {1, 2, 3}


def test_carry_forward_reruns_failed_step_and_its_dependents():
    """Step 1 FAILED → it re-runs; step 3 depends on it → re-runs too even
    though it PASSED; independent step 2 is carried."""
    svs = {1: _verdict(1, "FAILED"), 2: _verdict(2), 3: _verdict(3)}
    kept, verifs = _carry_forward(DIAMOND_PLAN, DIAMOND_RESULTS, svs)
    assert kept == {2: "out-2"}
    # all three still match the new plan — their verdicts (and any retry
    # feedback) survive, carried or not
    assert set(verifs) == {1, 2, 3}


def test_carry_forward_invalidation_is_transitive():
    """1 → 2 → 3 chain: re-running step 1 re-runs the whole chain."""
    svs = {1: _verdict(1, "FAILED"), 2: _verdict(2), 3: _verdict(3)}
    kept, _ = _carry_forward(LINEAR_PLAN, DIAMOND_RESULTS, svs)
    assert kept == {}


def test_carry_forward_accepts_passed_with_notes():
    svs = {n: _verdict(n, "PASSED WITH NOTES") for n in (1, 2, 3)}
    kept, _ = _carry_forward(DIAMOND_PLAN, DIAMOND_RESULTS, svs)
    assert kept == DIAMOND_RESULTS


def test_carry_forward_reworded_subtask_reruns_and_drops_verdict():
    """A reworded subtask is a different step: not carried, and its old
    verdict is dropped."""
    plan = [step(1), {**step(2), "subtask": "a reworded subtask 2"}, step(3, deps=[1, 2])]
    svs = {n: _verdict(n) for n in (1, 2, 3)}
    kept, verifs = _carry_forward(plan, DIAMOND_RESULTS, svs)
    assert kept == {1: "out-1"}             # 2 reworded, 3 depends on 2
    assert set(verifs) == {1, 3}


def test_carry_forward_ignores_whitespace_and_case_in_subtask():
    plan = [step(1), {**step(2), "subtask": "  SubTask   2 "}, step(3, deps=[1, 2])]
    svs = {n: _verdict(n) for n in (1, 2, 3)}
    kept, _ = _carry_forward(plan, DIAMOND_RESULTS, svs)
    assert kept == DIAMOND_RESULTS


def test_carry_forward_changed_agent_reruns():
    plan = [step(1), step(2, agent="mathematician"), step(3, deps=[1, 2])]
    svs = {n: _verdict(n) for n in (1, 2, 3)}
    kept, verifs = _carry_forward(plan, DIAMOND_RESULTS, svs)
    assert kept == {1: "out-1"}
    assert 2 not in verifs


def test_carry_forward_drops_verdicts_for_removed_steps():
    """Old step 3 FAILED and was replaced by new step 4: its stale FAILED
    verdict must not survive, or the aggregate would stay FAILED forever."""
    plan = [step(1), step(2), step(4, deps=[1, 2])]
    svs = {1: _verdict(1), 2: _verdict(2), 3: _verdict(3, "FAILED")}
    kept, verifs = _carry_forward(plan, DIAMOND_RESULTS, svs)
    assert kept == {1: "out-1", 2: "out-2"}
    assert set(verifs) == {1, 2}


def test_carry_forward_never_carries_failure_or_skip_markers():
    results = {1: "[STEP FAILED] boom", 2: "[SKIPPED — dependency failed]", 3: "out-3"}
    plan = [step(1), step(2), step(3)]
    svs = {n: _verdict(n) for n in (1, 2, 3)}
    kept, _ = _carry_forward(plan, results, svs)
    assert kept == {3: "out-3"}


def test_carry_forward_ignores_verdicts_without_a_recorded_subtask():
    """Entries from before verify_node stamped subtasks (or its parse-failure
    path) can't be matched, so nothing is carried — the safe default."""
    svs = {n: {"verdict": "PASSED", "notes": "", "retries": 0} for n in (1, 2, 3)}
    kept, verifs = _carry_forward(DIAMOND_PLAN, DIAMOND_RESULTS, svs)
    assert kept == {} and verifs == {}


def test_carry_forward_node_is_a_noop_on_the_first_pass():
    state = {"plan": DIAMOND_PLAN, "results": DIAMOND_RESULTS,
             "step_verifications": {n: _verdict(n) for n in (1, 2, 3)},
             "replan_count": 0}
    assert carry_forward_node(state) == {}


def test_carry_forward_node_replaces_results_after_a_replan():
    state = {"plan": DIAMOND_PLAN, "results": DIAMOND_RESULTS, "replan_count": 1,
             "step_verifications": {1: _verdict(1, "FAILED"), 2: _verdict(2),
                                    3: _verdict(3)}}
    out = carry_forward_node(state)
    assert out["results"] == {RESULTS_RESET: "", 2: "out-2"}
    assert set(out["step_verifications"]) == {1, 2, 3}


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


def test_e2e_replan_carries_passed_steps_and_routes_to_writer_at_cap(monkeypatch):
    """Every verify pass demands a replan of step 1: the orchestrator runs
    again, and only what needs it re-runs — step 1 (FAILED) and step 3 (its
    upstream re-ran). Step 2 PASSED, is unchanged in the new plan and has no
    re-running upstream, so its output is carried forward. At the replan cap
    the run falls through to the writer with every result intact."""
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
    # step 2 ran once and was carried across both replans; 1 and 3 re-ran
    assert sorted(s for s, _ in fake.worker_calls) == [1, 1, 1, 2, 3, 3, 3]
    assert out["results"] == {1: "out-1", 2: "out-2", 3: "out-3"}
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


def test_e2e_replan_to_new_step_id_passes_without_stale_verdict(monkeypatch):
    """The planner follows its prompt: the FAILED step 3 is reworded under a
    new id 4. Steps 1 and 2 are carried (not re-run), only step 4 runs, and
    old step 3's stale FAILED verdict is dropped — so the second pass ends
    PASSED instead of replanning until the budget is exhausted."""
    replanned = [step(1), step(2), {**step(4, deps=[1, 2]), "subtask": "reworded 3"}]
    orchestrator_calls = {"n": 0}

    def replanning_orchestrator(state):
        orchestrator_calls["n"] += 1
        plan = DIAMOND_PLAN if orchestrator_calls["n"] == 1 else replanned
        return {"plan": plan, "results": {}, "current_step": 0}

    fake = FakeLLMCall(verdicts=[
        [
            {"step": 1, "verification_result": "PASSED", "notes": ""},
            {"step": 2, "verification_result": "PASSED", "notes": ""},
            {"step": 3, "verification_result": "FAILED", "notes": "REPLAN: reword it"},
        ],
        [
            {"step": 1, "verification_result": "PASSED", "notes": ""},
            {"step": 2, "verification_result": "PASSED", "notes": ""},
            {"step": 4, "verification_result": "PASSED", "notes": ""},
        ],
    ])
    graph = _build_yotta(monkeypatch, fake, orchestrator=replanning_orchestrator)
    out = _invoke(graph, search_results="sr")

    assert orchestrator_calls["n"] == 2                  # initial + one replan
    assert sorted(s for s, _ in fake.worker_calls) == [1, 2, 3, 4]
    assert out["results"] == {1: "out-1", 2: "out-2", 4: "out-4"}
    assert set(out["step_verifications"]) == {1, 2, 4}  # step 3 dropped
    assert out["verification_result"] == "PASSED"
    assert out["verification_route"] == "proceed"
    assert out["replan_count"] == 1
    assert out["verification_exhausted"] is False
    assert out["final_output"] == "FINAL"


# ---------------------------------------------------------------------------
# Effort slider — instant route, policy caps, safety stops
# ---------------------------------------------------------------------------

def _policy(effort):
    return resolve_execution_policy(effort).as_dict()


FAILED_STEP2 = [
    {"step": 1, "verification_result": "PASSED", "notes": ""},
    {"step": 2, "verification_result": "FAILED", "notes": "recheck the numbers"},
    {"step": 3, "verification_result": "PASSED", "notes": ""},
]
ALL_PASSED = [
    {"step": 1, "verification_result": "PASSED", "notes": ""},
    {"step": 2, "verification_result": "PASSED", "notes": ""},
    {"step": 3, "verification_result": "PASSED", "notes": ""},
]
REPLAN_VERDICTS = [
    {"step": 1, "verification_result": "FAILED", "notes": "REPLAN: add a step"},
    {"step": 2, "verification_result": "PASSED", "notes": ""},
    {"step": 3, "verification_result": "PASSED", "notes": ""},
]


def test_e2e_instant_single_writer_no_planner_no_verifier(monkeypatch):
    """Instant guarantees: zero orchestrator invocations, exactly one writer
    synthesis call (the direct-route branch), no worker dispatch, no
    tool-verifier, no replan."""
    orchestrator_calls = {"n": 0}

    def counting_orchestrator(state):
        orchestrator_calls["n"] += 1
        return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}

    fake = FakeLLMCall(verdicts=[])
    graph = _build_yotta(monkeypatch, fake, orchestrator=counting_orchestrator)
    out = _invoke(graph, effort="instant", execution_policy=_policy("instant"))

    assert orchestrator_calls["n"] == 0                 # planner never ran
    assert fake.worker_calls == []                      # no worker dispatch
    assert fake.verifier_subtasks == []                 # verifier never ran
    assert fake.writer_subtask                          # exactly one writer synthesis call
    assert out["final_output"] == "FINAL"               # the writer's answer IS the output
    assert out["plan"] == []                            # empty plan → direct-route branch
    assert out["effort"] == "instant"
    assert out["replan_count"] == 0
    assert out["verification_attempts"] == 0
    assert out.get("status") is None                    # status derived at the boundary
    # ``step_stats`` is an ``operator.add`` channel, so LangGraph materializes
    # its empty value ([]) even when no node writes it — unlike plain channels
    # such as ``status``, which stay absent. Either shape means "no worker rows".
    assert not out.get("step_stats")                    # no worker rows on this route


def test_e2e_instant_writer_receives_attached_files(monkeypatch):
    """Instant keeps file grounding: the writer's direct-route branch renders
    the attached documents into its prompt."""
    fake = FakeLLMCall(verdicts=[])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="instant", execution_policy=_policy("instant"),
                  files={"a.txt": "doc text"})

    assert fake.worker_files == []                      # no worker dispatch
    assert "doc text" in fake.writer_subtask            # files rendered into the writer prompt
    assert out["final_output"] == "FINAL"


def test_e2e_instant_writer_failure_propagates(monkeypatch):
    """A failing writer on the instant route is NOT contained: the writer
    node owns no containment (same as the normal route), so the error
    escapes the run rather than producing a partial answer."""
    async def failing_writer(step, results, current_datetime="", **kwargs):
        raise RuntimeError("boom")

    def no_orchestrator(state):
        raise AssertionError("orchestrator must not run on instant")

    graph = _build_yotta(monkeypatch, failing_writer, orchestrator=no_orchestrator)
    with pytest.raises(RuntimeError, match="boom"):
        _invoke(graph, effort="instant", execution_policy=_policy("instant"))


def test_e2e_instant_cancellation_escapes(monkeypatch):
    """Cancellation on the instant route escapes the graph, never contained."""
    async def cancel_writer(step, results, current_datetime="", **kwargs):
        raise asyncio.CancelledError()

    graph = _build_yotta(monkeypatch, cancel_writer, orchestrator=lambda s: None)
    with pytest.raises((asyncio.CancelledError, NodeCancelledError)):
        _invoke(graph, effort="instant", execution_policy=_policy("instant"))


def test_e2e_step_retry_cap_comes_from_policy(monkeypatch):
    """Standard allows one step-verification retry: after the second FAILED
    verdict the step degrades to PASSED WITH NOTES instead of a third retry,
    and the verifier runs exactly once per pass (no duplicate invocation)."""
    fake = FakeLLMCall(verdicts=[list(FAILED_STEP2), list(FAILED_STEP2), list(ALL_PASSED)])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="standard", execution_policy=_policy("standard"))

    assert [s for s, _ in fake.worker_calls].count(2) == 2   # initial + ONE retry
    assert len(fake.verifier_subtasks) == 2                  # once per pass
    assert out["step_verifications"][2]["retries"] == 1
    assert out["step_verifications"][2]["verdict"] == "PASSED WITH NOTES"
    assert out["verification_result"] == "PASSED WITH NOTES"
    assert out["verification_exhausted"] is False
    assert out["final_output"] == "FINAL"


def test_e2e_explicit_unlimited_keeps_two_retries(monkeypatch):
    """Unlimited is compatibility: the historical two-retry cap still applies
    when the policy is present explicitly."""
    fake = FakeLLMCall(verdicts=[list(FAILED_STEP2), list(FAILED_STEP2),
                                 list(FAILED_STEP2)])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="unlimited", execution_policy=_policy("unlimited"))

    assert [s for s, _ in fake.worker_calls].count(2) == 3   # initial + two retries
    assert out["step_verifications"][2]["retries"] == 2
    assert out["final_output"] == "FINAL"


def test_e2e_replan_exhaustion_is_partial_with_warning(monkeypatch):
    """Standard allows one replan: the second replan demand hits the cap and
    routes to the writer with an explicitly partial, warned result — never a
    loop, never a transport error."""
    orchestrator_calls = {"n": 0}

    def counting_orchestrator(state):
        orchestrator_calls["n"] += 1
        return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}

    fake = FakeLLMCall(verdicts=[list(REPLAN_VERDICTS), list(REPLAN_VERDICTS)])
    graph = _build_yotta(monkeypatch, fake, orchestrator=counting_orchestrator)
    out = _invoke(graph, effort="standard", execution_policy=_policy("standard"))

    assert orchestrator_calls["n"] == 2            # initial + the one allowed replan
    assert out["replan_count"] == 2
    assert out["verification_exhausted"] is True
    assert out["status"] == "partial"
    assert out["verification_result"] == "FAILED"
    assert "Partial result warning" in fake.writer_subtask
    assert "verification budget" in fake.writer_subtask
    # exhaustion must not wipe the research: the partial writer still sees
    # every sub-agent result (it used to receive an empty results channel)
    assert "## Sub-agent results" in fake.writer_subtask
    for n in (1, 2, 3):
        assert f"out-{n}" in fake.writer_subtask
    assert out["final_output"] == "FINAL"


def test_e2e_dispatch_ceiling_triggers_safety_stop(monkeypatch):
    """A policy with max_graph_dispatches=1 stops after the first wave with a
    structured safety_stop_reason — the run finalizes instead of looping."""
    policy = _policy("light")
    policy["max_graph_dispatches"] = 1
    fake = FakeLLMCall(verdicts=[])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="light", execution_policy=policy)

    assert out["safety_stop_reason"] == "max_graph_dispatches=1 exceeded"
    assert out["dispatch_count"] == 2
    assert sorted(s for s, _ in fake.worker_calls) == [1, 2]  # only the first wave ran
    assert out["status"] == "partial"
    assert "Partial result warning" in fake.writer_subtask
    assert "max_graph_dispatches" in fake.writer_subtask


def test_e2e_deadline_safety_stop_before_any_dispatch(monkeypatch):
    """An already-expired deadline stops the run before the first worker wave."""
    policy = _policy("standard")
    policy["deadline"] = 0.0
    fake = FakeLLMCall(verdicts=[])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="standard", execution_policy=policy)

    assert out["safety_stop_reason"] == "wall-clock deadline exceeded"
    assert fake.worker_calls == []
    assert out["status"] == "partial"


def test_e2e_deadline_expires_during_final_wave_skips_verify(monkeypatch):
    """A deadline that expires while the last worker wave is in flight is
    enforced when that wave completes: the scheduler finalizes the best
    result available (partial) instead of starting the verification phase.

    The flag must flip during the FINAL wave (step 3 of DIAMOND_PLAN): the
    scheduler runs three times (#1 dispatches {1,2}, #2 dispatches {3}, #3
    sees ``"verify"``). A flip during wave 1 would be caught by the existing
    pre-dispatch gate at #2 and would exercise none of the post-wave check.
    """
    expired = {"flag": False}

    def deadline_flag(policy, now=None):
        return expired["flag"]

    monkeypatch.setattr("graphs.yotta_graph.deadline_exceeded", deadline_flag)

    class DeadlineFakeLLMCall(FakeLLMCall):
        async def __call__(self, step, results, current_datetime="", **kwargs):
            if step["step"] == 3:      # DIAMOND_PLAN: the final wave
                expired["flag"] = True
            return await super().__call__(step, results, current_datetime, **kwargs)

    fake = DeadlineFakeLLMCall(verdicts=[list(ALL_PASSED)])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="standard", execution_policy=_policy("standard"))

    assert out["safety_stop_reason"] == "wall-clock deadline exceeded"
    assert out["dispatch_count"] == 2                    # both waves dispatched
    assert sorted(s for s, _ in fake.worker_calls) == [1, 2, 3]
    assert fake.verifier_subtasks == []                  # verification never started
    assert out["status"] == "partial"
    assert "Partial result warning" in fake.writer_subtask
    assert "wall-clock deadline exceeded" in fake.writer_subtask


def test_e2e_replan_feedback_reaches_the_planner_state(monkeypatch):
    """On a replan pass the planner's state carries the verifier's structured
    notes — the production orchestrator injects them into its prompt."""
    seen_states = []

    def recording_orchestrator(state):
        seen_states.append(state)
        return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}

    fake = FakeLLMCall(verdicts=[list(REPLAN_VERDICTS), list(ALL_PASSED)])
    graph = _build_yotta(monkeypatch, fake, orchestrator=recording_orchestrator)
    _invoke(graph, effort="unlimited", execution_policy=_policy("unlimited"))

    assert len(seen_states) == 2
    assert seen_states[0].get("verification_notes", "") == ""   # first pass: no notes
    assert "REPLAN" in seen_states[1].get("verification_notes", "")


def test_e2e_verification_attempts_counter_increments_per_pass(monkeypatch):
    fake = FakeLLMCall(verdicts=[list(FAILED_STEP2), list(ALL_PASSED)])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="unlimited", execution_policy=_policy("unlimited"))
    assert out["verification_attempts"] == 2


def test_effort_policy_round_trips_through_state_and_entry(monkeypatch):
    """The entry router stamps the policy into state; the returned state is
    serializable (it went through graph channels/checkpointing)."""
    fake = FakeLLMCall(verdicts=[list(ALL_PASSED)])
    graph = _build_yotta(monkeypatch, fake)
    out = _invoke(graph, effort="thorough", execution_policy=_policy("thorough"))
    assert out["effort"] == "thorough"
    again = ExecutionPolicy.from_dict(out["execution_policy"])
    assert again.preset == "thorough"
    assert again.max_replans == 2
