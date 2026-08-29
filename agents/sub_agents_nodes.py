import asyncio
import time
from functools import partial
from typing import Any, Awaitable, Callable

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.callbacks import AsyncCallbackManager, BaseCallbackHandler, CallbackManager
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableConfig
from skill_loader import load_skills, load_skills_body
from tools.agent_tools import AGENT_TOOLS
from utils.validator import validate_step_output
from agent_mcp_tools import create_mcp_client
from dotenv import load_dotenv
from llm_factory import create_llm
from utils.logger import log_event
from pipeline_entry import render_files_block

from agents import AGENT_ROSTER
from agents.agent_states import get_current_datetime_str, WorkerState
from config_loader import AGENT_CONFIG, get_max_attempts
from execution_policy import effective_worker_attempts

load_dotenv()

_SKILL_INDEX, _SKILL_DICTIONARY_PAIRS = load_skills()

# Whitelist for per-agent llm config blocks (Phase 4.3)
_LLM_CONFIG_KEYS = {"model", "url", "api_key_env", "temperature", "max_tokens"}


class ToolBudgetExceededError(RuntimeError):
    """Raised when a ReAct attempt requests more tool calls than its effort
    budget allows. Raised *in flight* (from a callback inside the agent loop,
    before the offending tool ever executes) so ``run_sub_agent_async`` can
    convert the exhaustion into a finalize pass — the agent is re-invoked on
    the same checkpointer thread with a strict no-more-tools instruction and
    finishes with the information already retrieved. Only if that finalize
    pass still requests tools does the error escape to
    ``run_step_with_attempts`` — the pipeline's single retry owner — which
    decides whether to retry with a fresh per-attempt budget or contain the
    step.
    """


class _ToolBudgetGuard(BaseCallbackHandler):
    """Callback that counts requested tool calls during ReAct execution.

    Fires on every model call inside the agent loop (``on_llm_end``) and
    raises ``ToolBudgetExceededError`` the moment the cumulative count would
    exceed the cap — the tool that broke the budget never executes.
    ``raise_error = True`` is what makes the raise propagate out of
    ``agent.ainvoke`` instead of being logged and swallowed by langchain's
    callback manager. The exception is *not* fatal to the step: the runner
    catches it and resumes the conversation with a finalize instruction (see
    ``_finalize_after_budget_exhaustion``).
    """

    raise_error = True

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.count = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        for generation in (response.generations or []):
            for gen in generation:
                message = gen.message if hasattr(gen, "message") else None
                tool_calls = getattr(message, "tool_calls", None) or []
                self.count += len(tool_calls)
                if self.count > self.cap:
                    raise ToolBudgetExceededError(
                        f"Tool budget exceeded: the attempt requested "
                        f"{self.count} tool calls, the effort policy allows "
                        f"{self.cap}. The offending call was not executed."
                    )


_FINALIZE_INSTRUCTION = (
    "Your tool-call budget for this attempt is exhausted — you must not call "
    "any more tools. Finalize your answer now, using the information you have "
    "already retrieved and the context in this conversation. Reply with your "
    "final answer as plain text only."
)


async def _finalize_after_budget_exhaustion(
    agent: Any,
    saver: InMemorySaver,
    invoke_config: dict,
    parent_handlers: list,
    step_num: int,
    agent_name: str,
    subtask: str,
    cap: int,
) -> dict:
    """One strict finalize pass on the same checkpointer thread.

    The failed attempt's checkpoint already holds the conversation up to the
    last committed superstep; invoking again on the same thread appends the
    finalize instruction via the messages reducer and re-runs the agent node
    with the full history (langgraph resume-after-error semantics — the input
    checkpoint is committed before the first model call, and the failed
    node's writes are stored as an ERROR control signal, not merged).

    A fresh, strict cap-0 guard replaces the exhausted one, so any further
    tool request raises ``ToolBudgetExceededError`` again and escapes to the
    bounded attempt loop as before. ``parent_handlers`` are the caller's
    pre-existing callback handlers, kept so the parent run's callbacks fire
    on this pass too.

    Defensive seed: if the thread has no checkpoint (it always has one in
    langgraph 1.2.4 — the input checkpoint is committed before the first
    model call), start a fresh conversation with the subtask so the agent
    never finalizes blind.
    """
    finalize_config = dict(invoke_config)
    finalize_config["callbacks"] = [*parent_handlers, _ToolBudgetGuard(0)]
    if await saver.aget_tuple(finalize_config) is None:
        payload = {
            "messages": [
                ("user", subtask),
                SystemMessage(content=_FINALIZE_INSTRUCTION),
            ]
        }
    else:
        payload = {"messages": [SystemMessage(content=_FINALIZE_INSTRUCTION)]}
    try:
        return await agent.ainvoke(payload, config=finalize_config or None)
    except ToolBudgetExceededError:
        log_event(
            "tool_budget_finalize_failed",
            step_num=step_num,
            agent_name=agent_name,
            max_tool_calls_per_attempt=cap,
        )
        raise


def _llm_kwargs(llm_block: dict, agent_name: str) -> dict:
    """Translate an agent's optional ``llm`` config into ``create_llm`` kwargs.

    Whitelists the five accepted keys and raises ``ValueError`` naming the agent
    and the offending key on a typo.  ``api_key_env`` values are env-var
    **names** — never actual keys.
    """
    if not llm_block:
        return {}
    bad = set(llm_block) - _LLM_CONFIG_KEYS
    if bad:
        raise ValueError(
            f"Unknown key(s) in agent '{agent_name}' llm block: {sorted(bad)}. "
            f"Accepted keys: {sorted(_LLM_CONFIG_KEYS)}."
        )
    return dict(llm_block)

def _build_system_prompt(agent_name: str, agent_description: str,
                          skill_bodies: list[str], context: dict,
                          current_datetime: str = "",
                          documents_block: str = "") -> str:
    skill_block   = "\n\n---\n\n".join(skill_bodies)
    context_block = f"\n\n## Upstream context\n{context}" if context else ""
    datetime_line = f"\n\nCurrent datetime: {current_datetime}" if current_datetime else ""
    docs_block    = f"\n\n{documents_block}" if documents_block else ""
    return f"""You are the {agent_name} specialist agent.
Role: {agent_description}

## Active skills
{skill_block}
{context_block}{datetime_line}{docs_block}

Use tools when needed. Return your final answer as plain text. No meta-commentary."""


async def run_sub_agent_async(
    step: dict,
    results: dict,
    current_datetime: str = "",
    llm=None,
    config: RunnableConfig | None = None,
    streaming: bool = False,
    files: dict[str, str] | None = None,
    feedback: str = "",
    policy: dict | None = None,
) -> tuple[int, str, dict]:
    """Run one sub-agent step. Returns (step_number, output_text, stats_dict).

    ``stats_dict`` carries the token counts and tool-call count that only the
    inner invocation can see: ``input_tokens``, ``output_tokens``, ``tool_calls``.
    The calling worker node adds timing, step identity and status to build the
    full ``StepStats`` entry (Phase 4.9).

    ``llm`` defaults to the env-configured client at the creative default
    temperature (Phase 1.3). It is resolved here rather than at import so that
    importing this module needs no LLM configuration; ``create_llm`` is
    lru_cached, so repeat calls are a dict lookup.

    ``config`` is the run config threaded through from the graph invocation and
    forwarded to the sub-agent, so tools can read ``configurable`` (plan 4.5,
    artifact paths). ``None`` is fine — the sub-agent runs with a fresh config.

    ``policy`` (effort slider) is the run's serialized execution policy, or
    ``None`` for callers without one (legacy graphs, standalone nodes — no
    budget is enforced then, preserving current behavior). When present it
    enforces, *in flight*:

    - ``max_tool_calls_per_attempt`` — a callback guard raises
      ``ToolBudgetExceededError`` before the tool that would exceed the cap
      ever executes; on exhaustion the agent is re-invoked once on the same
      checkpointer thread with a strict finalize instruction (no more tools),
      so the step succeeds with the information already retrieved. The error
      escapes ``agent.ainvoke`` — and the bounded attempt loop decides whether
      to retry (fresh per-attempt budget) or contain the step — only if that
      finalize pass itself requests tools;
    - ``react_recursion_limit`` — merged into the inner invocation config, so
      the ReAct model/tool loop is bounded even when no tool is called.
    """
    cap = policy.get("max_tool_calls_per_attempt") if policy else None
    budget = _ToolBudgetGuard(cap) if cap is not None else None
    invoke_config: dict = dict(config) if config else {}
    if policy:
        recursion_limit = policy.get("react_recursion_limit")
        if recursion_limit is not None:
            invoke_config["recursion_limit"] = recursion_limit
    saver = None
    parent_handlers: list = []
    if budget is not None:
        # The graph runtime stashes its own callback manager under the
        # config's ``callbacks`` key — never splat it into a list. Merge its
        # handlers with the budget guard so the inner invocation keeps the
        # parent run's callbacks.
        existing = invoke_config.get("callbacks")
        if isinstance(existing, list):
            handlers = existing
        elif isinstance(existing, (CallbackManager, AsyncCallbackManager)):
            handlers = existing.handlers
        else:
            handlers = []
        parent_handlers = handlers
        invoke_config["callbacks"] = [*handlers, budget]
        # The checkpointer enables the budget-exhaustion finalize pass below:
        # it must re-invoke the SAME thread, so a thread_id is required —
        # setdefault, never overwrite a caller-provided one. Each attempt
        # builds a fresh saver, so threads never collide across retries or
        # parallel workers.
        saver = InMemorySaver()
        configurable = invoke_config.setdefault("configurable", {})
        configurable.setdefault("thread_id", f"subagent-step-{step['step']}")
    agent_name   = step["agent"]
    step_num     = step["step"]
    llm          = llm or create_llm(**_llm_kwargs(AGENT_CONFIG.get(agent_name, {}).get("llm", {}), agent_name), streaming=streaming)

    # Activate only the skills this step needs
    requested   = step.get("skills_needed", [])
    skill_bodies = [
        load_skills_body(_SKILL_DICTIONARY_PAIRS, s_name)
        for skill_name in requested
        for s_name in _SKILL_INDEX.keys() if s_name == skill_name
    ]

    # Gather upstream context from completed dependency steps
    context = {d: results.get(d, "") for d in step.get("depends_on", [])}

    # Fallback to live datetime if not provided from state
    dt = current_datetime or get_current_datetime_str()

    # Inject full text of the attached documents this step was assigned
    requested_files = step.get("files", [])
    available_files = files or {}
    selected_files = {fn: available_files[fn] for fn in requested_files if fn in available_files}
    documents_block = render_files_block(selected_files)

    system_prompt = _build_system_prompt(
        agent_name, AGENT_ROSTER[agent_name], skill_bodies, context, dt, documents_block
    )

    # Combine native tools + MCP tools for this agent
    native_tools = AGENT_TOOLS.get(agent_name, [])
    mcp_client = create_mcp_client(agent_name)

    print(f"Running Step {step_num} with agent '{agent_name}' using skills {requested} and context from steps {step.get('depends_on', [])}\n-----------\n")
    log_event(
        "run_sub_agent_start",
        step_num=step_num,
        agent_name=agent_name,
        skills=requested,
        dependencies=step.get("depends_on", []),
        subtask=step["subtask"],
    )

    subtask = step["subtask"]
    if feedback:
        subtask = f"{subtask}\n\n## Verifier feedback on the previous attempt (address this)\n{feedback}"

    if mcp_client is not None:
        mcp_tools = await mcp_client.get_tools()
        all_tools = native_tools + mcp_tools

        agent = create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=SystemMessage(content=system_prompt),
            checkpointer=saver,
        )
    else:
        agent = create_react_agent(
                model=llm,
                tools=native_tools,
                prompt=SystemMessage(content=system_prompt),
                checkpointer=saver,
            )

    budget_finalized = False
    try:
        # ``subtask`` here carries the verifier feedback block when present
        # (F2) — the raw step text would silently drop it.
        result = await agent.ainvoke(
            {"messages": [("user", subtask)]}, config=invoke_config or None
        )
    except ToolBudgetExceededError:
        log_event(
            "tool_budget_exhausted",
            step_num=step_num,
            agent_name=agent_name,
            tool_calls=budget.count if budget is not None else None,
            max_tool_calls_per_attempt=cap,
        )
        if saver is None:  # defensive: a guard is always paired with a saver
            raise
        result = await _finalize_after_budget_exhaustion(
            agent, saver, invoke_config, parent_handlers,
            step_num, agent_name, subtask, cap,
        )
        budget_finalized = True
        log_event(
            "tool_budget_finalized",
            step_num=step_num,
            agent_name=agent_name,
            tool_calls=budget.count if budget is not None else None,
        )

    tools_used = [
        call
        for message in result["messages"]
        for call in getattr(message, "tool_calls", None) or []
    ]

    # Collect token usage from every AIMessage that carries usage_metadata
    input_tokens = 0
    output_tokens = 0
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            usage = getattr(message, "usage_metadata", None) or {}
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    stats = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        # With an in-flight budget guard the counter is the authoritative
        # count of requested tool calls; without one, count from the messages.
        "tool_calls": budget.count if budget is not None else len(tools_used),
        # True when the attempt hit the budget and finished via the finalize
        # pass instead of a normal tool-less completion.
        "budget_exhausted": budget_finalized,
    }

    log_event("run_sub_agent_end", step_num=step_num, agent_name=agent_name,
              tools_used=tools_used, **stats)
    output = result["messages"][-1].content
    output = validate_step_output(step_num, agent_name, output)
    return step_num, output, stats


async def run_step_with_attempts(
    step: dict,
    run_attempt: Callable[[], Awaitable[tuple[int, str, dict]]],
    max_attempts: int,
    graph_name: str = "",
) -> tuple[int, str, dict]:
    """Run ``run_attempt`` up to ``max_attempts`` times (Slice 3).

    Returns ``(step_num, output, stats)`` from the first successful attempt.
    On exhaustion, re-raises the final attempt's exception so the calling
    worker can contain it — containment therefore happens **only after the
    final attempt**, never before.

    This helper is the pipeline's single retry owner: graph builders attach
    no worker-node ``RetryPolicy``, so a permanently failing step produces
    exactly ``max_attempts`` executions. The count comes validated from
    ``config_loader.get_max_attempts`` (integer 1–10, default 2).

    Every failed **non-final** attempt logs ``sub_agent_attempt_failed`` with
    step, agent, attempt number, maximum attempts, exception type and error
    text; the final failure is logged by the calling worker. Attempt-level
    token aggregation is not claimed — failed calls expose no reliable usage.

    ``asyncio.CancelledError``, ``KeyboardInterrupt`` and ``SystemExit`` are
    never contained as step failures: they escape immediately.
    """
    step_num = step["step"]
    agent_name = step["agent"]
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await run_attempt()
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                log_event(
                    "sub_agent_attempt_failed",
                    step_num=step_num,
                    agent_name=agent_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    exception_type=type(exc).__name__,
                    error=str(exc),
                    graph=graph_name or None,
                )
    if last_exc is None:  # pragma: no cover - max_attempts is validated >= 1
        raise RuntimeError(
            f"run_step_with_attempts got max_attempts={max_attempts!r} (must be >= 1)"
        )
    raise last_exc


async def _run_guarded_attempt(
    step: dict,
    run_step: Callable[..., Awaitable[tuple[int, str, dict]]],
    results: dict,
    current_datetime: str,
) -> tuple[int, str, dict]:
    """One sequential-worker attempt: satisfy the dependency guard, then run.

    Raises ``RuntimeError`` when the step's dependencies are not all present
    — a permanently blocked step dispatched in error. The guard participates
    in the bounded attempt loop like any other ordinary exception (Slice 3).
    """
    unfinished_deps = [d for d in step.get("depends_on", []) if d not in results]
    if unfinished_deps:
        raise RuntimeError(
            f"Step {step['step']} cannot execute: its dependencies "
            f"{unfinished_deps} are not in results and cannot be satisfied. "
            f"Check that every step's depends_on references valid step numbers."
        )
    return await run_step(step, results, current_datetime)


def _resolve_run_step(run_step, llm):
    """Return ``(run_step, takes_config)``.

    ``run_step`` wins; otherwise the real step runs, optionally on a given LLM.
    ``takes_config`` is True only for the production runner, which accepts the
    run config so tools can read ``configurable`` (plan 4.5); test-injected
    stubs keep their 3-argument signature and never receive it.
    """
    if run_step is not None:
        return run_step, False
    default = partial(run_sub_agent_async, llm=llm) if llm is not None else run_sub_agent_async
    return default, True


def _bind_config(run_step, takes_config, config):
    """Bind the run config into ``run_step`` when it is accepted, else no-op."""
    if takes_config and config is not None:
        return partial(run_step, config=config)
    return run_step


def make_sub_agent_node(run_step=None, llm=None):
    """Factory for the sequential worker node.

    Returns a dual-mode node (``RunnableLambda``): the sync body drives the
    attempt loop on a fresh event loop for ``graph.invoke``, the async body
    awaits it directly for ``graph.ainvoke``.

    ``run_step`` defaults to ``run_sub_agent_async``; tests inject a stub
    coroutine
    ``(step, results, current_datetime) -> (step_num, output, stats_dict)``.
    ``llm`` overrides the client that default ``run_step`` uses.
    """
    run_step, takes_config = _resolve_run_step(run_step, llm)

    async def sub_agent_node_async(state: WorkerState, config: RunnableConfig = None) -> dict:
        """Sequential node: executes the next uncompleted step in the plan.

        Retry before containment (Slice 3): the node owns the bounded attempt
        loop configured by the agent's ``execution.max_attempts``, so a step
        is contained only after its final attempt raises. Per-step stats
        (Phase 4.9): every step — completed, failed, or blocked — emits exactly
        one ``StepStats`` entry, with ``duration_s`` covering all attempts.
        """
        step = state["step"]
        results = state["results"]
        current_datetime = state.get("current_datetime", "")

        # Step already completed — no-op (defensive; the router should
        # never dispatch an already-completed step, but if it does we
        # must not return a scalar string that would break the results
        # reducer ``lambda a, b: {**a, **b}``).
        if step["step"] in results:
            return {}

        t0 = time.monotonic()
        try:
            step_runner = _bind_config(run_step, takes_config, config)
            # Production runner: thread the effort policy (tool budget +
            # recursion limit). Legacy payloads carry none — behavior unchanged.
            if takes_config:
                step_runner = partial(step_runner, policy=state.get("execution_policy"))
            step_num, output, inner_stats = await run_step_with_attempts(
                step,
                partial(_run_guarded_attempt, step, step_runner, results, current_datetime),
                effective_worker_attempts(
                    state.get("execution_policy"), get_max_attempts(step["agent"])
                ),
                graph_name="sequential",
            )
            stats = {
                "step": step_num,
                "agent": step["agent"],
                "status": "completed",
                "duration_s": round(time.monotonic() - t0, 3),
                "input_tokens": inner_stats["input_tokens"],
                "output_tokens": inner_stats["output_tokens"],
                "tool_calls": inner_stats["tool_calls"],
            }
            return {"results": {step_num: output}, "step_stats": [stats]}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Cancellation and process-level signals are never contained as
            # step failures — they escape to the caller (Slice 3).
            raise
        except Exception as e:
            import traceback
            error_string = traceback.format_exc()
            duration = round(time.monotonic() - t0, 3)
            log_event("sub_agent_step_failed", step=state["step"]["step"], error=error_string)
            return {
                "results": {state["step"]["step"]: f"[STEP FAILED] {e}"},
                "failed_steps": [state["step"]["step"]],
                "step_stats": [{
                    "step": state["step"]["step"],
                    "agent": state["step"]["agent"],
                    "status": "failed",
                    "duration_s": duration,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tool_calls": 0,
                }],
            }

    def sub_agent_node(state: WorkerState, config: RunnableConfig = None) -> dict:
        """Sync entry point — LangGraph calls this under ``graph.invoke``."""
        return asyncio.run(sub_agent_node_async(state, config))

    return RunnableLambda(sub_agent_node, afunc=sub_agent_node_async)


def make_parallel_sub_agent_node(run_step=None, llm=None):
    """Factory for the parallel worker node (one ``Send`` task per ready step).

    Returns a dual-mode node (``RunnableLambda``): the sync body drives the
    attempt loop on a fresh event loop for ``graph.invoke``, the async body
    awaits it directly for ``graph.ainvoke``.

    ``run_step`` defaults to ``run_sub_agent_async``; tests inject a stub
    coroutine
    ``(step, results, current_datetime) -> (step_num, output, stats_dict)``.
    ``llm`` overrides the client that default ``run_step`` uses.
    Failure is contained here (Phase 1.4): an exhausted step is recorded as a
    result and flagged in ``failed_steps`` instead of killing the whole graph.
    """
    run_step, takes_config = _resolve_run_step(run_step, llm)

    async def parallel_sub_agent_node_async(state: WorkerState, config: RunnableConfig = None) -> dict:
        step = state["step"]
        t0 = time.monotonic()
        try:
            step_runner = _bind_config(run_step, takes_config, config)
            # Production runner: thread the effort policy (tool budget +
            # recursion limit). Legacy payloads carry none — behavior unchanged.
            if takes_config:
                step_runner = partial(step_runner, policy=state.get("execution_policy"))
            step_num, output, inner_stats = await run_step_with_attempts(
                step,
                partial(step_runner, step, state["results"], state.get("current_datetime", "")),
                effective_worker_attempts(
                    state.get("execution_policy"), get_max_attempts(step["agent"])
                ),
                graph_name="parallel",
            )
            stats = {
                "step": step_num,
                "agent": step["agent"],
                "status": "completed",
                "duration_s": round(time.monotonic() - t0, 3),
                "input_tokens": inner_stats["input_tokens"],
                "output_tokens": inner_stats["output_tokens"],
                "tool_calls": inner_stats["tool_calls"],
            }
            return {"results": {step_num: output}, "step_stats": [stats]}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Cancellation and process-level signals are never contained as
            # step failures — they escape to the caller (Slice 3).
            raise
        except Exception as e:
            duration = round(time.monotonic() - t0, 3)
            log_event("sub_agent_step_failed", step=state["step"]["step"], error=str(e))
            return {"results": {state["step"]["step"]: f"[STEP FAILED] {e}"},
                    "failed_steps": [state["step"]["step"]],
                    "step_stats": [{
                        "step": step["step"],
                        "agent": step["agent"],
                        "status": "failed",
                        "duration_s": duration,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "tool_calls": 0,
                    }]}

    def parallel_sub_agent_node(state: WorkerState, config: RunnableConfig = None) -> dict:
        """Sync entry point — LangGraph calls this under ``graph.invoke``."""
        return asyncio.run(parallel_sub_agent_node_async(state, config))

    return RunnableLambda(parallel_sub_agent_node, afunc=parallel_sub_agent_node_async)