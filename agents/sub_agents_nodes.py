from functools import partial
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills, load_skills_body
from tools.agent_tools import AGENT_TOOLS
from utils.validator import validate_step_output
from agent_mcp_tools import create_mcp_client
from dotenv import load_dotenv
from llm_factory import create_llm
from utils.logger import log_event

from agents import AGENT_ROSTER
from agents.agent_states import get_current_datetime_str, WorkerState

load_dotenv()

_SKILL_INDEX, _SKILL_DICTIONARY_PAIRS = load_skills()

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
) -> tuple[int, str]:
    """Run one sub-agent step. Returns (step_number, output_text).

    ``llm`` defaults to the env-configured client at the creative default
    temperature (Phase 1.3). It is resolved here rather than at import so that
    importing this module needs no LLM configuration; ``create_llm`` is
    lru_cached, so repeat calls are a dict lookup.
    """
    llm          = llm or create_llm()
    agent_name   = step["agent"]
    step_num     = step["step"]

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
    log_event("run_sub_agent_end", step_num=step_num, agent_name=agent_name, tools_used=tools_used)
    output = result["messages"][-1].content
    output = validate_step_output(step_num, agent_name, output)
    return step_num, output



def _resolve_run_step(run_step, llm):
    """``run_step`` wins; otherwise run the real step, optionally on a given LLM."""
    if run_step is not None:
        return run_step
    return partial(run_sub_agent_async, llm=llm) if llm is not None else run_sub_agent_async


def make_sub_agent_node(run_step=None, llm=None):
    """Factory for the sequential worker node.

    ``run_step`` defaults to ``run_sub_agent_async``; tests inject a stub
    coroutine ``(step, results, current_datetime) -> (step_num, output)``.
    ``llm`` overrides the client that default ``run_step`` uses.
    """
    run_step = _resolve_run_step(run_step, llm)

    async def sub_agent_node(state: WorkerState) -> dict:
        """
        Sequential node: executes the next uncompleted step in the plan.
        """

        step    = state["step"]
        results = state["results"]
        current_datetime = state.get("current_datetime", "")
        # Find the next step whose dependencies are all resolved
        if step["step"] in results:
            return {"results": results[step["step"]]}
        deps_met = all(d in results for d in step.get("depends_on", []))
        if deps_met:
            step_num, output = await run_step(step, results, current_datetime)

            return {"results": {step_num: output}}

        unfinished_deps = [d for d in step.get("depends_on", []) if d not in results]
        raise RuntimeError(
            f"Step {step['step']} cannot execute: its dependencies "
            f"{unfinished_deps} are not in results and cannot be satisfied. "
            f"Check that every step's depends_on references valid step numbers."
        )

    return sub_agent_node


def make_parallel_sub_agent_node(run_step=None, llm=None):
    """Factory for the parallel worker node (one ``Send`` task per ready step).

    ``run_step`` defaults to ``run_sub_agent_async``; tests inject a stub
    coroutine ``(step, results, current_datetime) -> (step_num, output)``.
    ``llm`` overrides the client that default ``run_step`` uses.
    Failure is contained here (Phase 1.4): an exhausted step is recorded as a
    result and flagged in ``failed_steps`` instead of killing the whole graph.
    """
    run_step = _resolve_run_step(run_step, llm)

    async def parallel_sub_agent_node(state: WorkerState) -> dict:
        try:
            step_num, output = await run_step(
                state["step"], state["results"], state.get("current_datetime", "")
            )
            return {"results": {step_num: output}}
        except Exception as e:
            log_event("sub_agent_step_failed", step=state["step"]["step"], error=str(e))
            return {"results": {state["step"]["step"]: f"[STEP FAILED] {e}"},
                    "failed_steps": [state["step"]["step"]]}

    return parallel_sub_agent_node