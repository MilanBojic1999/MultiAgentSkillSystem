import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills, load_skills_body
from tools.agent_tools import AGENT_TOOLS
from utils.validator import validate_step_output
from agent_mcp_tools import create_mcp_client
from dotenv import load_dotenv
import os
from utils.logger import log_event
from pipeline_entry import render_files_block

from agents import AGENT_ROSTER
from agent_states import get_current_datetime_str


load_dotenv()

LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_KEY = os.getenv("LLM_KEY")


_SKILL_INDEX, _SKILL_DICTIONARY_PAIRS = load_skills()

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
    streaming: bool = False,
    files: dict[str, str] | None = None,
    feedback: str = "",
) -> tuple[int, str]:
    """Run one sub-agent step. Returns (step_number, output_text)."""
    agent_name   = step["agent"]
    agent_cfg    = next(a_name for a_name in AGENT_ROSTER.keys() if a_name == agent_name)
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

    # Inject full text of the attached documents this step was assigned
    requested_files = step.get("files", [])
    available_files = files or {}
    selected_files = {fn: available_files[fn] for fn in requested_files if fn in available_files}
    documents_block = render_files_block(selected_files)

    system_prompt = _build_system_prompt(
        agent_name, AGENT_ROSTER[agent_cfg], skill_bodies, context, dt, documents_block
    )

    # Combine native tools + MCP tools for this agent
    native_tools = AGENT_TOOLS.get(agent_name, [])
    mcp_client = create_mcp_client(agent_name)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=LLM_KEY,
        openai_api_base=LLM_URL,
        max_tokens=8192,
        temperature=0.2,
        streaming=streaming,
    )


    print(f"Running Step {step_num} with agent '{agent_name}' using skills {requested} and context from steps {step.get('depends_on', [])}\n-----------\n")
    log_event("run_sub_agent_start", step_num=step_num, agent_name=agent_name, skills=requested, dependencies=step.get("depends_on", []))

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
        )
        result = await agent.ainvoke({"messages": [("user", subtask)]})
    else:
        agent = create_react_agent(
                model=llm,
                tools=native_tools,
                prompt=SystemMessage(content=system_prompt),
            )
        result = await agent.ainvoke({"messages": [("user", subtask)]})


    log_event("run_sub_agent_end", step_num=step_num, agent_name=agent_name, tools_used=result["messages"][-1].tool_calls)
    # print(agent_name, "-------------", result["messages"][-1])
    # print("+"*50)
    output = result["messages"][-1].content
    output = validate_step_output(step_num, agent_name, output)
    return step_num, output


def sub_agent_node(state: dict) -> dict:
    """
    Sequential node: executes the next uncompleted step in the plan.
    """

    plan    = state["plan"]
    results = state.get("results", {})
    current_datetime = state.get("current_datetime", "")
    streaming = state.get("streaming", False)
    files = state.get("files", {})

    # Find the next step whose dependencies are all resolved
    for step in plan:
        if step["step"] in results:
            continue
        deps_met = all(d in results for d in step.get("depends_on", []))
        if deps_met:
            step_num, output = asyncio.run(
                run_sub_agent_async(step, results, current_datetime, streaming, files)
            )

            return {"results": {step_num: output}}

    return {}