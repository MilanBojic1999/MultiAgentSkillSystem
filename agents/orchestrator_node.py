import json
import os
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills
from dotenv import load_dotenv
from utils.logger import log_event
from utils.sanitize import sanitize_content
from utils.json_utils import extract_json
from utils.plan_validator import validate_plan
from llm_factory import create_llm
import re

from agents import AGENT_ROSTER
from agents.agent_states import get_current_datetime_str


load_dotenv()


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


def make_orchestrator_agent(llm=None, agent_roster=None, skill_index=None):
    # The orchestrator only emits strict JSON, so it runs at low temperature —
    # creative sampling here is the main source of malformed plans.
    llm = llm or create_llm(temperature=float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.1")))
    roster = agent_roster or AGENT_ROSTER
    index = skill_index or SKILL_INDEX

    def orchestrator_agent(state: dict):
        user_task = state["task"]
        current_datetime = state.get("current_datetime") or get_current_datetime_str()
        skill_summery = "\n".join([f"- {name}: {desc['description']}" for name, desc in index.items()])
        agent_roster_str = "\n".join([f"- {name}: {desc}" for name, desc in roster.items()])

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

        response = llm.invoke(messages)
        try:
            plan_json = extract_json(response.content)
            plan = plan_json.get("plan", [])
            if not isinstance(plan, list) or len(plan) == 0:
                raise ValueError(f"Orchestrator produced an empty or invalid plan: {plan_json}")
            log_event("orchestrator_agent_plan", pipeline_plan=plan)

            plan = validate_plan(plan, set(roster), set(index))

            return {"plan": plan, "results": {}, "current_step": 0}
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    return orchestrator_agent
