import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills
from dotenv import load_dotenv
from utils.logger import log_event
from utils.senitize import sanitize_content
import re

from agents import AGENT_ROSTER
from agent_states import get_current_datetime_str


load_dotenv()

LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_KEY = os.getenv("LLM_KEY")


def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose
    """
    # Try direct parse first (best case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find the outermost JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"Could not extract JSON from response: {text[:500]}...")


SKILL_INDEX, SKILLS_DICTIONARY_PAIRS = load_skills()

ORCHESTRATOR_SYSTEM = """
You are the Orchestrator in a multi-agent pipeline.

## Your role
1. Analyse the user's task.
2. Decompose it into ordered subtasks.
3. For each subtask, select the best specialist sub-agent from the roster below.
4. Output a JSON plan in the exact format shown.
5. Do NOT execute any subtask yourself.

Current datetime: {current_datetime}

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

# Agents that are handled by dedicated pipeline nodes (verify_node, assemble_node)
# and should NOT appear in the orchestrator's plan — they run after the sub-agent loop.
_PIPELINE_RESERVED_AGENTS = {"verifier", "writer"}
_PIPELINE_RESERVED_SKILLS = {"answer-writer", "information-verifier"}

def orchestrator_agent(state: dict):
    user_task = state["task"]
    current_datetime = state.get("current_datetime") or get_current_datetime_str()
    streaming = state.get("streaming", False)

    skill_summery = "\n".join([f"- {name}: {desc['description']}" for name, desc in SKILL_INDEX.items() if name not in _PIPELINE_RESERVED_SKILLS])
    # Exclude pipeline-reserved agents so the orchestrator doesn't put them in the plan
    agent_roster_str = "\n".join(
        [f"- {name}: {desc}" for name, desc in AGENT_ROSTER.items()
         if name not in _PIPELINE_RESERVED_AGENTS]
    )

    system_prompt = ORCHESTRATOR_SYSTEM.format(
        agent_roster=agent_roster_str,
        skill_index=skill_summery,
        current_datetime=current_datetime,
    )
    user_task = sanitize_content(user_task, "user")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_task),
    ]
    
    log_event("orchestrator_agent_start", user_task=user_task)

    llm = ChatOpenAI(
        model=LLM_MODEL, # Must match the --model flag you gave vLLM
        openai_api_key=LLM_KEY,                  # vLLM doesn't require a key by default
        openai_api_base=LLM_URL, 
        max_tokens=4048,
        temperature=0.9,
        streaming=streaming,
    )

    response = llm.invoke(messages)
    try:
        plan_json = _extract_json(response.content)
        plan = plan_json.get("plan", [])
        if not isinstance(plan, list) or len(plan) == 0:
            raise ValueError(f"Orchestrator produced an empty or invalid plan: {plan_json}")
        log_event("orchestrator_agent_plan", pipeline_plan=plan)

        return {"plan": plan, "results": {}, "current_step": 0}
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}")