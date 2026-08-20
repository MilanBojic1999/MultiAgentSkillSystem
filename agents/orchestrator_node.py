import json
import os
from collections import deque
from typing import Any

from langchain_openai import ChatOpenAI
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

# Maximum replanning passes (keeps a bad LLM plan from looping forever).
# Module-level so graphs can import it (graphs/yotta_graph.py's after_verify).
_MAX_REPLANS = 3


SKILL_INDEX, SKILLS_DICTIONARY_PAIRS = load_skills()

ORCHESTRATOR_SYSTEM = """
You are the Planner in a multi-agent research pipeline.

## Your role
1. Analyse the user's query and initial search for the same query — and, if a verifier_report is present, the gap it flagged.
2. If you find the information from the search are sufficinet to answer the query to the fullest, return empty plan to signal the system good results
2. Decompose the query into the smallest ordered set of subqueries that, once each is answered, fully answers it.
3. For each subquery, mark which earlier subqueries (if any) it depends on, so dependent ones only run once those resolve (dependencis put inside `depends_on` field in the output).
4. For each subquery, select the best specialist agent from the roster below and tag the tool it's expected to need.
5. If any attached documents (see "## Attached documents" below) are relevant to a subquery, list their exact filenames in that step's `files` field — this is the ONLY way a worker gets the document's full text, so assign a document to every step that actually needs to read it. A step reading an attached document should normally use the `document-reader-worker` skill.
6. Output a JSON plan in the exact format shown.
7. Do NOT answer any subquery yourself, and do NOT call any tools.

Current datetime: {current_datetime}

## Replanning
A verifier_report in your input means this is a replanning pass, not a first pass. Revise only the subqueries it flagged — add a missing one, reword an unanswerable one, or fix a wrong dependency — and leave every subquery the verifier already passed untouched. This pipeline caps replanning at three passes; if replan_count is already 2 going in, mark any subquery still unresolved as such rather than requesting a fourth pass.

**Stable step ids**: for every subquery you are keeping unchanged (the verifier already approved it), reuse its EXACT SAME `step` id and EXACT SAME `subtask` text from the previous plan — the pipeline uses an (id, subtask) match to carry its already-verified output forward without re-running it. If you reword a kept subquery even slightly, its research will be discarded and it will re-run from scratch. Only assign new ids (starting above the highest id in the previous plan) to genuinely new or reworded subqueries.

## Available sub-agents
{agent_roster}

## Available skills (name -> description)
{skill_index}

## Output format (JSON only — no prose, no markdown fences)
{{
  "plan": [
    {{
      "step": 1,
      "subtask": "<concise description>",
      "agent": "<agent_name>",
      "skills_needed": ["<skill-name>"],
      "depends_on": [<step_ids_from_which_agent_depends_on>],
      "files": ["<attached filename, only if this step needs one — omit otherwise>"]
    }}
  ]
}}

or if input search results are good:
{{
  "plan": []
}}

""".strip()



def _describe_attached_files(files: dict[str, str]) -> str:
    """Render filename, size, and a short preview per attached document.

    Deliberately NOT full content — the orchestrator only needs enough to
    decide which steps should read which files; the assigned worker gets
    the full text via ``PlanStep.files`` (see ``run_sub_agent_async``).
    """
    if not files:
        return ""
    lines = []
    for filename, content in files.items():
        preview = content[:300].replace("\n", " ").strip()
        if len(content) > 300:
            preview += "..."
        lines.append(f"- **{filename}** ({len(content):,} chars): {preview}")
    return "## Attached documents\n" + "\n".join(lines)

def make_orchestrator_agent(llm=None, agent_roster=None, skill_index=None):
    # The orchestrator only emits strict JSON, so it runs at low temperature —
    # creative sampling here is the main source of malformed plans.
    llm = llm or create_llm(temperature=float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.1")))
    roster = agent_roster or AGENT_ROSTER
    index = skill_index or SKILL_INDEX

    _PIPELINE_RESERVED_AGENTS = {"verifier", "writer"}
    _PIPELINE_RESERVED_SKILLS = {"answer-writer", "information-verifier"}

    _FAILED_OUTPUT_PREVIEW_CHARS = 1500

    def orchestrator_agent(state: dict):
        print(state.keys())
        user_task = state["task"]
        current_datetime = state.get("current_datetime") or get_current_datetime_str()
        skill_summery = "\n".join([f"- {name}: {desc['description']}" for name, desc in SKILL_INDEX.items() if name not in _PIPELINE_RESERVED_SKILLS])
        agent_roster_str = "\n".join([f"- {name}: {desc}" for name, desc in AGENT_ROSTER.items() if name not in _PIPELINE_RESERVED_AGENTS])

        streaming = state.get("streaming", False)
        search_results = state.get("search_results", "")
        files = state.get("files", {})

        system_prompt = ORCHESTRATOR_SYSTEM.format(
            agent_roster=agent_roster_str,
            skill_index=skill_summery,
            current_datetime=current_datetime,
        )

        user_task = sanitize_content(user_task, "user")
        user_parts = [user_task]

        if search_results:
            user_parts.append(f"## Initial search results\n{search_results}")
        
        files_block = _describe_attached_files(files)
        if files_block:
            user_parts.append(files_block)

        user_task = "\n".join(user_parts)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_task),
        ]
        
        log_event("orchestrator_agent_start", user_task=user_task)

        response = llm.invoke(messages)
        try:
            plan_json = extract_json(response.content)
            plan = plan_json.get("plan", [])
            if not isinstance(plan, list):
                raise ValueError(f"Orchestrator produced an empty or invalid plan: {plan_json}")
            log_event("orchestrator_agent_plan", pipeline_plan=plan)

            if len(plan) == 0:
                return {"plan": plan, "results": {}, "current_step": 0}

            plan = validate_plan(plan, set(roster), set(index))

            return {"plan": plan, "results": {}, "current_step": 0}
        except Exception as e:
            raise ValueError(f"Failed to parse JSON response: {e}")

    return orchestrator_agent