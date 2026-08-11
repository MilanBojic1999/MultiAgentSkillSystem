import time
from functools import partial

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from skill_loader import load_skills, load_skills_body
from tools.agent_tools import AGENT_TOOLS
from utils.validator import validate_step_output
from agent_mcp_tools import create_mcp_client
from dotenv import load_dotenv
from llm_factory import create_llm
from utils.logger import log_event

from agents import AGENT_ROSTER
from agents.agent_states import get_current_datetime_str, WorkerState
from config_loader import AGENT_CONFIG

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



def _resolve_run_step(run_step, llm):
    """``run_step`` wins; otherwise run the real step, optionally on a given LLM."""
    if run_step is not None:
        return run_step
    return partial(run_sub_agent_async, llm=llm) if llm is not None else run_sub_agent_async


def make_sub_agent_node(run_step=None, llm=None):
    """Factory for the sequential worker node.

    ``run_step`` defaults to ``run_sub_agent_async``; tests inject a stub
    coroutine
    ``(step, results, current_datetime) -> (step_num, output, stats_dict)``.
    ``llm`` overrides the client that default ``run_step`` uses.
    """
    run_step = _resolve_run_step(run_step, llm)

    async def sub_agent_node(state: WorkerState) -> dict:
        """Sequential node: executes the next uncompleted step in the plan.

        Failure containment (Phase 4.13): a failed step is recorded as a result
        and flagged in ``failed_steps`` instead of killing the whole graph.
        Per-step stats (Phase 4.9): every step — completed, failed, or blocked —
        emits a ``StepStats`` entry.
        """
        step = state["step"]
        results = state["results"]
        current_datetime = state.get("current_datetime", "")

        t0 = time.monotonic()
        try:
            # Find the next step whose dependencies are all resolved
            if step["step"] in results:
                return {"results": results[step["step"]]}
            deps_met = all(d in results for d in step.get("depends_on", []))
            if deps_met:
                step_num, output, inner_stats = await run_step(step, results, current_datetime)
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

            unfinished_deps = [d for d in step.get("depends_on", []) if d not in results]
            raise RuntimeError(
                f"Step {step['step']} cannot execute: its dependencies "
                f"{unfinished_deps} are not in results and cannot be satisfied. "
                f"Check that every step's depends_on references valid step numbers."
            )
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

    return sub_agent_node


def make_parallel_sub_agent_node(run_step=None, llm=None):
    """Factory for the parallel worker node (one ``Send`` task per ready step).

    ``run_step`` defaults to ``run_sub_agent_async``; tests inject a stub
    coroutine
    ``(step, results, current_datetime) -> (step_num, output, stats_dict)``.
    ``llm`` overrides the client that default ``run_step`` uses.
    Failure is contained here (Phase 1.4): an exhausted step is recorded as a
    result and flagged in ``failed_steps`` instead of killing the whole graph.
    """
    run_step = _resolve_run_step(run_step, llm)

    async def parallel_sub_agent_node(state: WorkerState) -> dict:
        step = state["step"]
        t0 = time.monotonic()
        try:
            step_num, output, inner_stats = await run_step(
                step, state["results"], state.get("current_datetime", "")
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

    return parallel_sub_agent_node