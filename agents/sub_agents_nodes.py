import asyncio
import time
from functools import partial
from typing import Awaitable, Callable

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from skill_loader import load_skills, load_skills_body
from tools.agent_tools import AGENT_TOOLS
from utils.validator import validate_step_output
from agent_mcp_tools import create_mcp_client
from dotenv import load_dotenv
from llm_factory import create_llm
from utils.logger import log_event

from agents import AGENT_ROSTER
from agents.agent_states import get_current_datetime_str, WorkerState
from config_loader import AGENT_CONFIG, get_max_attempts

load_dotenv()

_SKILL_INDEX, _SKILL_DICTIONARY_PAIRS = load_skills()

# Whitelist for per-agent llm config blocks (Phase 4.3)
_LLM_CONFIG_KEYS = {"model", "url", "api_key_env", "temperature", "max_tokens"}


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
                          current_datetime: str = "") -> str:
    skill_block   = "\n\n---\n\n".join(skill_bodies)
    context_block = f"\n\n## Upstream context\n{context}" if context else ""
    datetime_line = f"\n\nCurrent datetime: {current_datetime}" if current_datetime else ""
    return f"""You are the {agent_name} specialist agent.
Role: {agent_description}

## Active skills
{skill_block}
{context_block}{datetime_line}

Use tools when needed. Return your final answer as plain text. No meta-commentary."""


async def run_sub_agent_async(
    step: dict,
    results: dict,
    current_datetime: str = "",
    llm=None,
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
    """
    agent_name   = step["agent"]
    step_num     = step["step"]
    llm          = llm or create_llm(**_llm_kwargs(AGENT_CONFIG.get(agent_name, {}).get("llm", {}), agent_name))

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

    system_prompt = _build_system_prompt(
        agent_name, AGENT_ROSTER[agent_name], skill_bodies, context, dt
    )

    # Combine native tools + MCP tools for this agent
    native_tools = AGENT_TOOLS.get(agent_name, [])
    mcp_client = create_mcp_client(agent_name)


    log_event(
        "run_sub_agent_start",
        step_num=step_num,
        agent_name=agent_name,
        skills=requested,
        dependencies=step.get("depends_on", []),
        subtask=step["subtask"],
    )


    if mcp_client is not None:
        mcp_tools = await mcp_client.get_tools()
        all_tools = native_tools + mcp_tools

        agent = create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=SystemMessage(content=system_prompt),
        )
        result = await agent.ainvoke({"messages": [("user", step["subtask"])]})
    else:
        agent = create_react_agent(
                model=llm,
                tools=native_tools,
                prompt=SystemMessage(content=system_prompt),
            )
        result = await agent.ainvoke({"messages": [("user", step["subtask"])]})


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
        "tool_calls": len(tools_used),
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
    """``run_step`` wins; otherwise run the real step, optionally on a given LLM."""
    if run_step is not None:
        return run_step
    return partial(run_sub_agent_async, llm=llm) if llm is not None else run_sub_agent_async


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
    run_step = _resolve_run_step(run_step, llm)

    async def sub_agent_node_async(state: WorkerState) -> dict:
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
            step_num, output, inner_stats = await run_step_with_attempts(
                step,
                partial(_run_guarded_attempt, step, run_step, results, current_datetime),
                get_max_attempts(step["agent"]),
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
            duration = round(time.monotonic() - t0, 3)
            log_event("sub_agent_step_failed", step=state["step"]["step"], error=str(e))
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

    def sub_agent_node(state: WorkerState) -> dict:
        """Sync entry point — LangGraph calls this under ``graph.invoke``."""
        return asyncio.run(sub_agent_node_async(state))

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
    run_step = _resolve_run_step(run_step, llm)

    async def parallel_sub_agent_node_async(state: WorkerState) -> dict:
        step = state["step"]
        t0 = time.monotonic()
        try:
            step_num, output, inner_stats = await run_step_with_attempts(
                step,
                partial(run_step, step, state["results"], state.get("current_datetime", "")),
                get_max_attempts(step["agent"]),
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

    def parallel_sub_agent_node(state: WorkerState) -> dict:
        """Sync entry point — LangGraph calls this under ``graph.invoke``."""
        return asyncio.run(parallel_sub_agent_node_async(state))

    return RunnableLambda(parallel_sub_agent_node, afunc=parallel_sub_agent_node_async)