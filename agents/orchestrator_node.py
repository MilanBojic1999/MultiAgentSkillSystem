import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills
from dotenv import load_dotenv

load_dotenv()

LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_KEY = os.getenv("LLM_KEY")

llm = ChatOpenAI(
    model=LLM_MODEL, # Must match the --model flag you gave vLLM
    openai_api_key=LLM_KEY,                  # vLLM doesn't require a key by default
    openai_api_base=LLM_URL, 
    max_tokens=4048,
    temperature=0.9
)

AGENT_ROSTER = {
    "mathematician": "Expert in solving complex mathematical problems and plotting functions.",
    "researcher": "Skilled in gathering and synthesizing information from various sources.",
    "writer": "Proficient in crafting clear and engaging written content on a wide range of topics.",
}

SKILL_INDEX, SKILLS_DICTIONARY_PAIRS = load_skills()

ORCHESTRATOR_SYSTEM = """
You are the Orchestrator in a multi-agent pipeline.

## Your role
1. Analyse the user's task.
2. Decompose it into ordered subtasks.
3. For each subtask, select the best specialist sub-agent from the roster below.
4. Output a JSON plan in the exact format shown.
5. Do NOT execute any subtask yourself.

## Available sub-agents
{agent_roster}

## Available skills (name → description)
{skill_index}

## Output format (JSON only — no prose, no markdown fences)
{{
  "plan": [
    {{
      "step": 1,
      "subtask": "<concise description>",
      "agent": "<agent_name>",
      "skills_needed": ["<skill-name>"],
      "depends_on": []
    }}
  ]
}}
""".strip()

def orchestrator_agent(state: dict):
    user_task = state["task"]
    skill_summery = "\n".join([f"- {name}: {desc['description']}" for name, desc in SKILL_INDEX.items()])
    agent_roster_str = "\n".join([f"- {name}: {desc}" for name, desc in AGENT_ROSTER.items()])

    system_prompt = ORCHESTRATOR_SYSTEM.format(
        agent_roster=agent_roster_str,
        skill_index=skill_summery,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_task),
    ]

    response = llm.invoke(messages)
    try:
        plan = json.loads(response.content)["plan"]
        return {"plan": plan, "results": {}, "current_step": 0}
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}")