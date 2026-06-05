import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills, load_skills_body
from agent_tools import AGENT_TOOLS
from agent_mcp_tools import create_mcp_client, MCP_ACCESS_AGENT
from dotenv import load_dotenv
import os

load_dotenv()

LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_KEY = os.getenv("LLM_KEY")


AGENT_ROSTER = {
    "mathematician": "Expert in solving complex mathematical problems and plotting functions.",
    "researcher": "Skilled in gathering and synthesizing information from various sources.",
    "writer": "Proficient in crafting clear and engaging written content on a wide range of topics.",
}


llm = ChatOpenAI(
    model=LLM_MODEL, # Must match the --model flag you gave vLLM
    openai_api_key=LLM_KEY,                  # vLLM doesn't require a key by default
    openai_api_base=LLM_URL, 
    max_tokens=4048,
    temperature=0.9
)

def _build_system_prompt(agent_name: str, agent_description: str,
                          skill_bodies: list[str], context: dict) -> str:
    skill_block   = "\n\n---\n\n".join(skill_bodies)
    context_block = f"\n\n## Upstream context\n{context}" if context else ""
    return f"""You are the {agent_name} specialist agent.
Role: {agent_description}

## Active skills
{skill_block}
{context_block}

Use tools when needed. Return your final answer as plain text. No meta-commentary."""


async def run_sub_agent_async(
    step: dict,
    skill_index: list[dict],
    skill_dictionary_pairs: dict[str, str],
    results: dict,
) -> tuple[int, str]:
    """Run one sub-agent step. Returns (step_number, output_text)."""
    agent_name   = step["agent"]
    agent_cfg    = next(a_name for a_name in AGENT_ROSTER.keys() if a_name == agent_name)
    step_num     = step["step"]

    # Activate only the skills this step needs
    requested   = step.get("skills_needed", [])
    skill_bodies = [
        load_skills_body(skill_dictionary_pairs, s_name)
        for skill_name in requested
        for s_name in skill_index.keys() if s_name == skill_name
    ]

    # Gather upstream context from completed dependency steps
    context = {d: results.get(d, "") for d in step.get("depends_on", [])}

    system_prompt = _build_system_prompt(
        agent_name, AGENT_ROSTER[agent_cfg], skill_bodies, context
    )

    # Combine native tools + MCP tools for this agent
    native_tools = AGENT_TOOLS.get(agent_name, [])
    mcp_client, mcp_tools = create_mcp_client(agent_name)
    all_tools = native_tools + mcp_tools

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SystemMessage(content=system_prompt),
    )

    print(f"Running Step {step_num} with agent '{agent_name}' using skills {requested} and context from steps {step.get('depends_on', [])}\n-----------\n{step["subtask"]}")
    if mcp_client is not None:
        async with mcp_client:
            result = await agent.ainvoke({"messages": [("user", step["subtask"])]})
    else:
        result = await agent.ainvoke({"messages": [("user", step["subtask"])]})


    output = result["messages"][-1].content
    return step_num, output


def sub_agent_node(state: dict) -> dict:
    """
    Sequential node: executes the next uncompleted step in the plan.
    For parallel fan-out, see Phase 6 (Send API).
    """
    skill_index, skill_dictionary_pairs = load_skills()

    plan    = state["plan"]
    results = state.get("results", {})

    # Find the next step whose dependencies are all resolved
    for step in plan:
        if step["step"] in results:
            continue
        deps_met = all(d in results for d in step.get("depends_on", []))
        if deps_met:
            step_num, output = asyncio.run(
                run_sub_agent_async(step, skill_index, skill_dictionary_pairs, results)
            )
            return {"results": {step_num: output}}

    return {}