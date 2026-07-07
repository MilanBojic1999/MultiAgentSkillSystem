import json
import os
from collections import deque
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from skill_loader import load_skills
from dotenv import load_dotenv
from utils.logger import log_event
from utils.senitize import sanitize_content
from utils.json_utils import extract_json

from agents import AGENT_ROSTER, AGENT_CONFIG
from agent_states import get_current_datetime_str


load_dotenv()

LLM_URL = os.getenv("LLM_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
LLM_KEY = os.getenv("LLM_KEY")


SKILL_INDEX, SKILLS_DICTIONARY_PAIRS = load_skills()

ORCHESTRATOR_SYSTEM = """
You are the Planner in a multi-agent research pipeline.

## Your role
1. Analyse the user's query and initial search for the same query — and, if a verifier_report is present, the gap it flagged.
2. If you find the information from the search are sufficinet to answer the query to the fullest, return empty plan to signal the system good results
2. Decompose the query into the smallest ordered set of subqueries that, once each is answered, fully answers it.
3. For each subquery, mark which earlier subqueries (if any) it depends on, so dependent ones only run once those resolve.
4. For each subquery, select the best specialist agent from the roster below and tag the tool it's expected to need.
5. Output a JSON plan in the exact format shown.
6. Do NOT answer any subquery yourself, and do NOT call any tools.

Current datetime: {current_datetime}

## Replanning
A verifier_report in your input means this is a replanning pass, not a first pass. Revise only the subqueries it flagged — add a missing one, reword an unanswerable one, or fix a wrong dependency — and leave every subquery the verifier already passed untouched. This pipeline caps replanning at three passes; if replan_count is already 2 going in, mark any subquery still unresolved as such rather than requesting a fourth pass.

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
      "depends_on": [<step_ids_from_which_agent_depends_on>]
    }}
  ]
}}

or if input search results are good:
{{
  "plan": []
}}

""".strip()

# Agents that are handled by dedicated pipeline nodes (verify_node, assemble_node)
# and should NOT appear in the orchestrator's plan — they run after the sub-agent loop.
_PIPELINE_RESERVED_AGENTS = {"verifier", "writer"}
_PIPELINE_RESERVED_SKILLS = {"answer-writer", "information-verifier"}

# Maximum replanning passes (keeps a bad LLM plan from looping forever).
_MAX_REPLANS = 3


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def _validate_plan(plan: list[dict[str, Any]]) -> None:
    """Validate a freshly-parsed plan before it enters state.

    Raises ``ValueError`` with a retryable message on any violation so the
    orchestrator can feed the error back and re-prompt the LLM.
    """
    if not isinstance(plan, list):
        raise ValueError("Plan is not a list.")

    # --- empty plan is always valid (signals "search results are sufficient") ---
    if len(plan) == 0:
        return

    step_ids: set[int] = set()

    for s in plan:
        if not isinstance(s, dict):
            raise ValueError(f"Plan step is not a dict: {s!r}")

        sid = s.get("step")
        if not isinstance(sid, int) or sid < 0:
            raise ValueError(f"Plan step has invalid 'step' field: {s!r}")

        if sid in step_ids:
            raise ValueError(f"Duplicate step id {sid} in plan.")
        step_ids.add(sid)

        # --- agent must exist (exclude pipeline-reserved agents) ---
        agent = s.get("agent", "")
        if agent not in AGENT_ROSTER or agent in _PIPELINE_RESERVED_AGENTS:
            valid = sorted(
                a for a in AGENT_ROSTER if a not in _PIPELINE_RESERVED_AGENTS
            )
            raise ValueError(
                f"Step {sid}: agent '{agent}' is unknown or reserved. "
                f"Available agents: {valid}"
            )

        # --- skills must exist (warn on unknown, don't block) ---
        for skill in s.get("skills_needed", []):
            if skill not in SKILL_INDEX and skill not in _PIPELINE_RESERVED_SKILLS:
                raise ValueError(
                    f"Step {sid}: skill '{skill}' is not in the skill index. "
                    f"Available skills: {sorted(SKILL_INDEX.keys())}"
                )

    # --- depends_on references must exist and be acyclic ---
    for s in plan:
        sid = s["step"]
        for dep in s.get("depends_on", []):
            if dep not in step_ids:
                raise ValueError(
                    f"Step {sid}: depends_on {dep} does not reference an existing step."
                )

    # Cycle detection via Kahn's algorithm
    in_degree: dict[int, int] = {s["step"]: 0 for s in plan}
    children: dict[int, list[int]] = {s["step"]: [] for s in plan}
    for s in plan:
        for dep in s.get("depends_on", []):
            children[dep].append(s["step"])
            in_degree[s["step"]] += 1

    queue: deque[int] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if visited != len(plan):
        raise ValueError(
            f"Plan contains a dependency cycle. Steps: {[s['step'] for s in plan]}"
        )


# ---------------------------------------------------------------------------
# Orchestrator agent
# ---------------------------------------------------------------------------

def orchestrator_agent(state: dict) -> dict:
    user_task = state["task"]
    current_datetime = state.get("current_datetime") or get_current_datetime_str()
    streaming = state.get("streaming", False)
    search_results = state.get("search_results", "")
    verifier_report = state.get("verifier_report", "")
    verification_notes = state.get("verification_notes", "")
    replan_count = state.get("replan_count", 0)

    skill_summery = "\n".join(
        f"- {name}: {desc['description']}"
        for name, desc in SKILL_INDEX.items()
        if name not in _PIPELINE_RESERVED_SKILLS
    )
    # Exclude pipeline-reserved agents so the orchestrator doesn't put them in the plan
    agent_roster_str = "\n".join(
        f"- {name}: {desc}"
        for name, desc in AGENT_ROSTER.items()
        if name not in _PIPELINE_RESERVED_AGENTS
    )

    system_prompt = ORCHESTRATOR_SYSTEM.format(
        agent_roster=agent_roster_str,
        skill_index=skill_summery,
        current_datetime=current_datetime,
    )

    # ---- build the user message with all available context -----------------
    user_parts: list[str] = []
    user_parts.append(f"## User query\n{user_task}")

    if search_results:
        user_parts.append(f"## Initial search results\n{search_results}")

    if verifier_report:
        user_parts.append(
            f"## Verifier report (REPLANNING PASS #{replan_count + 1} of {_MAX_REPLANS})\n"
            f"{verifier_report}\n\n"
            f"### Verifier notes\n{verification_notes}\n\n"
            f"Revise the plan to address ONLY the gaps flagged above. "
            f"Keep steps the verifier already approved."
        )

    user_message = "\n\n".join(user_parts)
    user_message = sanitize_content(user_message, "user")

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    log_event("orchestrator_agent_start", user_task=user_task)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=LLM_KEY,
        openai_api_base=LLM_URL,
        max_tokens=8192,
        temperature=0.2,           # strict-JSON role — needs deterministic output
        streaming=streaming,
    )

    response = llm.invoke(messages)
    try:
        plan_json = extract_json(response.content)
        plan = plan_json.get("plan", [])
        if not isinstance(plan, list):
            raise ValueError(
                f"Orchestrator produced an empty or invalid plan: {plan_json}"
            )

        # --- validate before letting the plan into state ---
        _validate_plan(plan)

        if len(plan) == 0:
            return {"plan": plan, "results": {}, "current_step": 0}

        log_event("orchestrator_agent_plan", pipeline_plan=plan)

        # When replanning: clear stale step results from the previous plan
        # so set-based routing sees the new plan's steps as uncompleted.
        # The sentinel key -1 tells the results reducer to replace, not merge.
        if verifier_report:
            return {
                "plan": plan,
                "results": {-1: "", 0: user_task},   # clear sentinel
                "current_step": 0,
                "step_verifications": {},             # new plan → new step IDs
            }

        return {"plan": plan, "results": {0: user_task}, "current_step": 0}

    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}")