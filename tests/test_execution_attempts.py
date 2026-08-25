"""Slice 3 regression suite: retry before containment.

Pins the worker-owned bounded attempt loop introduced by Slice 3 of
``TIER_1_CLOSURE_PLAN.md``:

  * a permanently failing step executes exactly ``execution.max_attempts``
    times — no graph-level worker ``RetryPolicy`` multiplies the count;
  * failure markers and skip propagation appear only after exhaustion;
  * exactly one final ``StepStats`` row per planned step, with token and
    tool metadata taken from the successful final attempt;
  * ``asyncio.CancelledError``, ``KeyboardInterrupt`` and ``SystemExit``
    escape instead of being contained as step failures.

``max_attempts`` is read through ``agents.sub_agents_nodes.get_max_attempts``,
so tests monkeypatch that one name; the config-side validation and defaulting
is covered by ``tests/test_config_loader.py``.
"""

import asyncio
from collections import Counter
from typing import ClassVar

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult

from agents import sub_agents_nodes as worker_mod
from agents.sub_agents_nodes import (
    ToolBudgetExceededError,
    _ToolBudgetGuard,
    make_parallel_sub_agent_node,
    make_sub_agent_node,
    run_step_with_attempts,
    run_sub_agent_async,
)
from execution_policy import resolve_execution_policy
from tests.plans import DIAMOND_PLAN, LINEAR_PLAN, step

CONFIG = {"configurable": {"thread_id": "test-execution-attempts"}}


def _build_parallel(plan, worker):
    """The real parallel graph, with the two LLM-bearing nodes faked out."""
    from graphs.parallel_pipeline_graph import build

    def stub_orchestrator(state):
        return {"plan": plan, "results": {}, "current_step": 0}

    return build(orchestrator=stub_orchestrator, sub_agent=worker)


def _build_sequential(plan, worker):
    """The real sequential graph, with the two LLM-bearing nodes faked out."""
    from graphs.sequential_pipeline_graph import build

    def stub_orchestrator(state):
        return {"plan": plan, "results": {}, "current_step": 0}

    return build(orchestrator=stub_orchestrator, sub_agent=worker)


def _stats(tokens=0, tool_calls=0):
    return {"input_tokens": tokens, "output_tokens": tokens, "tool_calls": tool_calls}


# ---------------------------------------------------------------------------
# run_step_with_attempts — the shared helper
# ---------------------------------------------------------------------------

def test_helper_returns_first_success_immediately():
    """A successful first attempt makes exactly one call."""
    calls: list[int] = []
    result = (7, "out-7", _stats(tokens=5))

    async def attempt():
        calls.append(1)
        return result

    out = asyncio.run(run_step_with_attempts(step(7), attempt, max_attempts=2))
    assert out == result
    assert len(calls) == 1


def test_helper_retries_until_success():
    """max_attempts=2: first call fails, second succeeds."""
    calls: list[int] = []
    result = (7, "out-7", _stats(tokens=99))

    async def attempt():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return result

    out = asyncio.run(run_step_with_attempts(step(7), attempt, max_attempts=2))
    assert out == result
    assert len(calls) == 2


def test_helper_exhaustion_raises_final_exception():
    """max_attempts=3: exactly three calls, then the final error is re-raised."""
    calls: list[int] = []

    async def attempt():
        calls.append(1)
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        asyncio.run(run_step_with_attempts(step(3), attempt, max_attempts=3))
    assert len(calls) == 3


def test_helper_logs_every_failed_nonfinal_attempt(monkeypatch):
    """Each failed non-final attempt logs diagnostics; the final one does not."""
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        worker_mod, "log_event", lambda event, **kw: events.append((event, kw))
    )

    async def attempt():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        asyncio.run(
            run_step_with_attempts(
                step(5, agent="researcher"), attempt, max_attempts=3, graph_name="parallel"
            )
        )

    assert [event for event, _ in events] == ["sub_agent_attempt_failed"] * 2
    for (event, kw), attempt_no in zip(events, [1, 2]):
        assert event == "sub_agent_attempt_failed"
        assert kw["step_num"] == 5
        assert kw["agent_name"] == "researcher"
        assert kw["attempt"] == attempt_no
        assert kw["max_attempts"] == 3
        assert kw["exception_type"] == "RuntimeError"
        assert kw["error"] == "boom"
        assert kw["graph"] == "parallel"


@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError, KeyboardInterrupt, SystemExit]
)
def test_helper_process_level_exceptions_escape(exc):
    """Cancellation and process-level signals are never contained."""

    async def attempt():
        raise exc()

    with pytest.raises(exc):
        asyncio.run(run_step_with_attempts(step(1), attempt, max_attempts=3))


# ---------------------------------------------------------------------------
# Worker nodes — containment only after exhaustion
# ---------------------------------------------------------------------------

def test_worker_node_contains_after_exhaustion(monkeypatch):
    """max_attempts=1: one failure call, then contained failure state."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 1)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        raise RuntimeError("boom")

    node = make_sub_agent_node(run_step=fake_run)
    result = asyncio.run(
        node.ainvoke({"step": step(1), "results": {}, "current_datetime": ""})
    )

    assert calls == [1]
    assert "[STEP FAILED]" in result["results"][1]
    assert result["failed_steps"] == [1]
    (row,) = result["step_stats"]
    assert row["status"] == "failed"
    assert row["input_tokens"] == 0


def test_worker_node_retries_then_completes(monkeypatch):
    """max_attempts=2: first attempt fails, second succeeds → completed.

    Token metadata comes from the successful final attempt and the emitted
    row is the only row — a retried step never produces multiple stats rows.
    """
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 2)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        if len(calls) == 1:
            await asyncio.sleep(0.02)
            raise RuntimeError("transient")
        return s["step"], "out-1", _stats(tokens=42, tool_calls=2)

    node = make_sub_agent_node(run_step=fake_run)
    result = asyncio.run(
        node.ainvoke({"step": step(1), "results": {}, "current_datetime": ""})
    )

    assert len(calls) == 2
    assert result["results"] == {1: "out-1"}
    assert "failed_steps" not in result
    (row,) = result["step_stats"]
    assert row["status"] == "completed"
    assert row["input_tokens"] == 42
    assert row["tool_calls"] == 2
    # duration covers the failed attempt's wall-clock time too
    assert row["duration_s"] >= 0.02


def test_worker_node_sync_invoke_drives_attempt_loop(monkeypatch):
    """The factory node is dual-mode: plain ``.invoke`` runs the same
    bounded attempt loop as ``.ainvoke`` (Slice 2.3 / Slice 3)."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 2)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        if len(calls) == 1:
            raise RuntimeError("transient")
        return s["step"], "out-1", _stats(tokens=7)

    node = make_sub_agent_node(run_step=fake_run)
    result = node.invoke({"step": step(1), "results": {}, "current_datetime": ""})

    assert len(calls) == 2
    assert result["results"] == {1: "out-1"}
    assert "failed_steps" not in result
    (row,) = result["step_stats"]
    assert row["status"] == "completed"
    assert row["input_tokens"] == 7


def test_worker_node_propagates_cancellation(monkeypatch):
    """Cancellation escapes the worker — never contained as a failed step."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 3)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        raise asyncio.CancelledError()

    node = make_parallel_sub_agent_node(run_step=fake_run)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(node.ainvoke({"step": step(1), "results": {}, "current_datetime": ""}))
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Compiled graphs — retry, containment, and exactly-one stats row
# ---------------------------------------------------------------------------

def test_parallel_single_attempt_then_contained(monkeypatch):
    """max_attempts=1: step 1 runs once, is contained, dependents are skipped."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 1)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        raise RuntimeError("boom")

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_parallel(LINEAR_PLAN, worker)
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=CONFIG))

    assert calls == [1]  # one attempt, then containment + skips
    assert 1 in out["failed_steps"]
    assert "[SKIPPED — dependency failed]" in out["final_output"]

    stats = {s["step"]: s for s in out["step_stats"]}
    assert len(out["step_stats"]) == 3
    assert stats[1]["status"] == "failed"
    assert stats[2]["status"] == "skipped"
    assert stats[3]["status"] == "skipped"


def test_parallel_retry_then_success_completes(monkeypatch):
    """max_attempts=2: first call fails, second succeeds, final status completed."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 2)
    per_step: Counter = Counter()

    async def fake_run(s, results, current_datetime=""):
        per_step[s["step"]] += 1
        if s["step"] == 1 and per_step[1] == 1:
            await asyncio.sleep(0.02)
            raise RuntimeError("transient")
        return s["step"], f"out-{s['step']}", _stats(tokens=100 + s["step"])

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_parallel(LINEAR_PLAN, worker)
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=CONFIG))

    assert per_step == Counter({1: 2, 2: 1, 3: 1})
    assert out.get("failed_steps", []) == []

    stats = {s["step"]: s for s in out["step_stats"]}
    assert len(out["step_stats"]) == 3
    assert stats[1]["status"] == "completed"
    assert stats[1]["input_tokens"] == 101  # from the successful final attempt
    assert stats[1]["duration_s"] >= 0.02   # includes the failed attempt


def test_parallel_permanent_failure_exact_attempts_no_nested_retry(monkeypatch):
    """max_attempts=3: exactly three calls, one failed row, dependent skipped.

    Runs through the compiled graph, so a reintroduced graph-level worker
    ``RetryPolicy`` would multiply the count and fail the call-count
    assertion. The independent branch must still complete.
    """
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 3)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        if s["step"] == 1:
            raise RuntimeError("boom")
        return s["step"], f"out-{s['step']}", _stats(tokens=10)

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_parallel(DIAMOND_PLAN, worker)
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=CONFIG))

    assert calls.count(1) == 3  # exactly max_attempts — no doubling
    assert calls.count(2) == 1  # independent branch still completes
    assert 3 not in calls       # blocked step never dispatched

    stats = {s["step"]: s for s in out["step_stats"]}
    assert len(out["step_stats"]) == 3
    assert stats[1]["status"] == "failed"
    assert stats[2]["status"] == "completed"
    assert stats[3]["status"] == "skipped"
    assert 1 in out["failed_steps"]


@pytest.mark.parametrize("mode", ["invoke", "ainvoke"])
def test_sequential_honors_attempt_count(monkeypatch, mode):
    """The sequential topology honors the same worker-owned attempt count
    under both ``invoke`` and ``ainvoke`` (Slice 2.3 / Slice 3)."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 2)
    per_step: Counter = Counter()

    async def fake_run(s, results, current_datetime=""):
        per_step[s["step"]] += 1
        if s["step"] == 1 and per_step[1] == 1:
            raise RuntimeError("transient")
        return s["step"], f"out-{s['step']}", _stats(tokens=10)

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_sequential(LINEAR_PLAN, worker)
    payload = {"task": "t", "current_datetime": ""}
    if mode == "invoke":
        out = graph.invoke(payload, config=CONFIG)
    else:
        out = asyncio.run(graph.ainvoke(payload, config=CONFIG))

    assert per_step == Counter({1: 2, 2: 1, 3: 1})
    stats = {s["step"]: s for s in out["step_stats"]}
    assert all(s["status"] == "completed" for s in stats.values())


def test_sequential_permanent_failure_skips_dependents(monkeypatch):
    """Sequential graph: exhaustion of step 1 skips its dependents."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 2)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        if s["step"] == 1:
            raise RuntimeError("boom")
        return s["step"], f"out-{s['step']}", _stats(tokens=10)

    worker = make_parallel_sub_agent_node(run_step=fake_run)
    graph = _build_sequential(LINEAR_PLAN, worker)
    out = asyncio.run(graph.ainvoke({"task": "t", "current_datetime": ""}, config=CONFIG))

    assert calls == [1, 1]  # two attempts, then containment + skips

    stats = {s["step"]: s for s in out["step_stats"]}
    assert stats[1]["status"] == "failed"
    assert stats[2]["status"] == "skipped"
    assert stats[3]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Effort slider — policy/static attempt intersection, tool budget, recursion
# ---------------------------------------------------------------------------

def _light_policy():
    return resolve_execution_policy("light", now=0.0).as_dict()


def test_worker_attempt_cap_is_min_of_policy_and_static_config(monkeypatch):
    """configured=3, policy light=1 -> exactly one execution. The policy and
    the static agent config intersect; neither wins outright."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 3)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        return s["step"], f"out-{s['step']}", _stats(tokens=1)

    node = make_parallel_sub_agent_node(run_step=fake_run)
    result = asyncio.run(node.ainvoke({
        "step": step(1),
        "results": {},
        "current_datetime": "",
        "execution_policy": _light_policy(),
    }))
    assert calls == [1]
    assert result["results"] == {1: "out-1"}


def test_worker_without_policy_keeps_configured_attempts(monkeypatch):
    """A legacy payload (no execution_policy) passes the configured count
    through unchanged — parallel/sequential behavior is preserved."""
    monkeypatch.setattr(worker_mod, "get_max_attempts", lambda agent: 2)
    calls: list[int] = []

    async def fake_run(s, results, current_datetime=""):
        calls.append(s["step"])
        if len(calls) == 1:
            raise RuntimeError("transient")
        return s["step"], f"out-{s['step']}", _stats(tokens=1)

    node = make_parallel_sub_agent_node(run_step=fake_run)
    result = asyncio.run(node.ainvoke({"step": step(1), "results": {}, "current_datetime": ""}))
    assert len(calls) == 2
    assert result["results"] == {1: "out-1"}


def _generation(tool_calls=0):
    calls = [
        {"name": "t", "args": {}, "id": f"i{i}", "type": "tool_call"}
        for i in range(tool_calls)
    ]
    return ChatGeneration(message=AIMessage(content="", tool_calls=calls))


def test_tool_budget_guard_blocks_before_cap_plus_one():
    """Cap 2: two requested calls pass, the third raises — in flight."""
    guard = _ToolBudgetGuard(2)
    guard.on_llm_end(LLMResult(generations=[[_generation(1)]]))
    guard.on_llm_end(LLMResult(generations=[[_generation(1)]]))
    assert guard.count == 2
    with pytest.raises(ToolBudgetExceededError, match="Tool budget exceeded"):
        guard.on_llm_end(LLMResult(generations=[[_generation(1)]]))


def test_tool_budget_guard_raises_within_a_single_response():
    """A response requesting more calls than the cap fails on that response."""
    guard = _ToolBudgetGuard(1)
    with pytest.raises(ToolBudgetExceededError):
        guard.on_llm_end(LLMResult(generations=[[_generation(2)]]))


def test_tool_budget_guard_without_tool_calls_never_raises():
    guard = _ToolBudgetGuard(0)  # instant's cap: zero tools allowed
    guard.on_llm_end(LLMResult(generations=[[_generation(0)]]))
    assert guard.count == 0


def test_tool_budget_guard_raise_error_flag_is_set():
    """``raise_error = True`` is what makes the exception escape the langchain
    callback manager instead of being logged and swallowed."""
    assert _ToolBudgetGuard(1).raise_error is True


class _ExplodingAgent:
    """create_react_agent stand-in: records every invoke, raises a scripted
    number of times, then returns a canned final answer."""

    last_config = None
    invoke_configs: list = []
    invoke_payloads: list = []
    raise_times = float("inf")   # class-level script; set before each test
    answer = "finalized"

    def __init__(self, *, model, tools, prompt, checkpointer=None):
        pass

    async def ainvoke(self, payload, config=None):
        _ExplodingAgent.last_config = config
        _ExplodingAgent.invoke_configs.append(config)
        _ExplodingAgent.invoke_payloads.append(payload)
        if len(_ExplodingAgent.invoke_configs) <= _ExplodingAgent.raise_times:
            raise ToolBudgetExceededError("Tool budget exceeded: simulated")
        return {"messages": [AIMessage(content=_ExplodingAgent.answer)]}


def _script_exploding_agent(raise_times=float("inf"), answer="finalized"):
    """Reset the class-level fake script before a test uses it."""
    _ExplodingAgent.last_config = None
    _ExplodingAgent.invoke_configs = []
    _ExplodingAgent.invoke_payloads = []
    _ExplodingAgent.raise_times = raise_times
    _ExplodingAgent.answer = answer


def test_run_sub_agent_finalize_failure_escapes_to_attempt_loop(monkeypatch):
    """When the finalize pass itself requests tools, the in-flight budget
    error escapes the agent invoke — the bounded attempt loop then decides
    whether to retry (fresh per-attempt budget) or contain the step."""
    _script_exploding_agent(raise_times=float("inf"))
    monkeypatch.setattr(worker_mod, "create_react_agent", _ExplodingAgent)
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        worker_mod, "log_event", lambda event, **kw: events.append((event, kw))
    )
    policy = {"max_tool_calls_per_attempt": 2, "react_recursion_limit": 7}

    with pytest.raises(ToolBudgetExceededError):
        asyncio.run(run_sub_agent_async(step(1), {}, policy=policy))

    assert len(_ExplodingAgent.invoke_configs) == 2
    assert [event for event, _ in events] == [
        "run_sub_agent_start",
        "tool_budget_exhausted",
        "tool_budget_finalize_failed",
    ]
    _, kw = events[1]
    assert kw["agent_name"] == "researcher"
    assert kw["max_tool_calls_per_attempt"] == 2
    assert events[2][1]["max_tool_calls_per_attempt"] == 2


def test_run_sub_agent_merges_recursion_limit_and_guard_into_config(monkeypatch):
    """The policy's ``react_recursion_limit`` rides into the inner invoke
    config alongside the tool-budget guard — and the existing configurable
    (thread/task ids) is preserved on both the attempt and the finalize pass."""
    _script_exploding_agent(raise_times=float("inf"))
    monkeypatch.setattr(worker_mod, "create_react_agent", _ExplodingAgent)
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)
    policy = {"max_tool_calls_per_attempt": 5, "react_recursion_limit": 9}
    config = {"configurable": {"thread_id": "t-1", "task_id": "art-1"}}

    with pytest.raises(ToolBudgetExceededError):
        asyncio.run(run_sub_agent_async(step(1), {}, config=config, policy=policy))

    attempt_config, finalize_config = _ExplodingAgent.invoke_configs
    assert attempt_config["recursion_limit"] == 9
    assert finalize_config["recursion_limit"] == 9
    for invoke_config in (attempt_config, finalize_config):
        assert invoke_config["configurable"]["thread_id"] == "t-1"
        assert invoke_config["configurable"]["task_id"] == "art-1"
    assert any(
        isinstance(h, _ToolBudgetGuard) for h in attempt_config["callbacks"]
    )
    finalize_guards = [
        h for h in finalize_config["callbacks"] if isinstance(h, _ToolBudgetGuard)
    ]
    assert [g.cap for g in finalize_guards] == [0]


def test_run_sub_agent_without_policy_adds_no_budget_machinery(monkeypatch):
    """No policy -> no guard, no recursion limit: legacy callers unchanged
    (a single invoke — no saver means a budget error is re-raised directly)."""
    _script_exploding_agent(raise_times=float("inf"))
    monkeypatch.setattr(worker_mod, "create_react_agent", _ExplodingAgent)
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)

    with pytest.raises(ToolBudgetExceededError):
        asyncio.run(run_sub_agent_async(step(1), {}))

    assert len(_ExplodingAgent.invoke_configs) == 1
    invoke_config = _ExplodingAgent.invoke_configs[0]
    assert invoke_config is None or (
        "recursion_limit" not in invoke_config and "callbacks" not in invoke_config
    )


def test_run_sub_agent_finalizes_after_budget_exhaustion(monkeypatch):
    """On budget exhaustion the agent is re-invoked on the same thread with a
    strict finalize instruction, and the step succeeds with its final answer."""
    _script_exploding_agent(raise_times=1, answer="FINAL")
    monkeypatch.setattr(worker_mod, "create_react_agent", _ExplodingAgent)
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        worker_mod, "log_event", lambda event, **kw: events.append((event, kw))
    )
    policy = {"max_tool_calls_per_attempt": 2, "react_recursion_limit": 7}

    step_num, output, stats = asyncio.run(
        run_sub_agent_async(step(1), {}, policy=policy)
    )

    assert step_num == 1
    assert output == "FINAL"
    assert stats["budget_exhausted"] is True
    assert stats["tool_calls"] == 0  # the fake never counts; guard count is 0
    assert [event for event, _ in events] == [
        "run_sub_agent_start",
        "tool_budget_exhausted",
        "tool_budget_finalized",
        "run_sub_agent_end",
    ]
    attempt_config, finalize_config = _ExplodingAgent.invoke_configs
    assert (
        attempt_config["configurable"]["thread_id"]
        == finalize_config["configurable"]["thread_id"]
    )
    finalize_guards = [
        h for h in finalize_config["callbacks"] if isinstance(h, _ToolBudgetGuard)
    ]
    assert [g.cap for g in finalize_guards] == [0]


def test_run_sub_agent_finalize_seeds_subtask_when_no_checkpoint(monkeypatch):
    """With no checkpoint on the thread (a stand-in agent never checkpoints),
    the finalize pass seeds the conversation with the subtask so the agent
    never finalizes blind."""
    _script_exploding_agent(raise_times=1, answer="SEEDED")
    monkeypatch.setattr(worker_mod, "create_react_agent", _ExplodingAgent)
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)
    policy = {"max_tool_calls_per_attempt": 2, "react_recursion_limit": 7}

    _, output, _ = asyncio.run(run_sub_agent_async(step(1), {}, policy=policy))
    assert output == "SEEDED"

    finalize_payload = _ExplodingAgent.invoke_payloads[1]
    messages = finalize_payload["messages"]
    assert messages[0] == ("user", "subtask 1")
    assert isinstance(messages[1], SystemMessage)
    assert "tool-call budget" in messages[1].content


def test_run_sub_agent_finalize_preserves_caller_configurable(monkeypatch):
    """A caller-provided thread/task id survives both the budgeted attempt and
    the finalize pass untouched."""
    _script_exploding_agent(raise_times=1, answer="OK")
    monkeypatch.setattr(worker_mod, "create_react_agent", _ExplodingAgent)
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)
    policy = {"max_tool_calls_per_attempt": 2, "react_recursion_limit": 7}
    config = {"configurable": {"thread_id": "t-1", "task_id": "art-1"}}

    _, output, _ = asyncio.run(
        run_sub_agent_async(step(1), {}, config=config, policy=policy)
    )
    assert output == "OK"

    attempt_config, finalize_config = _ExplodingAgent.invoke_configs
    assert attempt_config["configurable"] == {"thread_id": "t-1", "task_id": "art-1"}
    assert finalize_config["configurable"] == {"thread_id": "t-1", "task_id": "art-1"}


class _ScriptedModel(BaseChatModel):
    """Real BaseChatModel with a call-by-call script: the first call requests
    two tool calls (trips the cap-1 guard), every later call answers plain."""

    calls: ClassVar[int] = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _ScriptedModel.calls += 1
        if _ScriptedModel.calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "fake", "args": {}, "id": "call-1", "type": "tool_call"},
                    {"name": "fake", "args": {}, "id": "call-2", "type": "tool_call"},
                ],
            )
        else:
            message = AIMessage(content="DONE-FINAL")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_run_sub_agent_real_graph_resume_after_budget_raise(monkeypatch):
    """Hermetic end-to-end of the resume mechanics: the real
    ``create_react_agent`` + ``InMemorySaver`` + guard callback resume the
    conversation after the raise and finalize with the scripted answer."""
    _ScriptedModel.calls = 0
    monkeypatch.setattr(worker_mod, "create_mcp_client", lambda agent: None)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        worker_mod, "log_event", lambda event, **kw: events.append((event, kw))
    )
    policy = {"max_tool_calls_per_attempt": 1, "react_recursion_limit": 6}

    step_num, output, stats = asyncio.run(
        run_sub_agent_async(step(1), {}, llm=_ScriptedModel(), policy=policy)
    )

    assert step_num == 1
    assert output == "DONE-FINAL"
    assert stats["budget_exhausted"] is True
    assert stats["tool_calls"] == 2  # both requested calls, none executed
    assert [event for event, _ in events] == [
        "run_sub_agent_start",
        "tool_budget_exhausted",
        "tool_budget_finalized",
        "run_sub_agent_end",
    ]
