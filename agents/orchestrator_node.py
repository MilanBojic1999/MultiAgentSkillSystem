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

# Agents that are handled by dedicated pipeline nodes (verify_node, assemble_node)
# and should NOT appear in the orchestrator's plan — they run after the sub-agent loop.
_PIPELINE_RESERVED_AGENTS = {"verifier", "writer"}
_PIPELINE_RESERVED_SKILLS = {"answer-writer", "information-verifier"}

# Maximum replanning passes (keeps a bad LLM plan from looping forever).
_MAX_REPLANS = 3

# Per-step output preview length in the replanning "failed steps" block —
# enough for the planner to judge what went wrong without blowing the prompt.
_FAILED_OUTPUT_PREVIEW_CHARS = 1500


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

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


def _validate_plan(plan: list[dict[str, Any]], available_files: set[str] | None = None) -> None:
    """Validate a freshly-parsed plan before it enters state.

    Raises ``ValueError`` with a retryable message on any violation so the
    orchestrator can feed the error back and re-prompt the LLM.
    """
    available_files = available_files or set()
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

        # --- referenced files must actually be attached ---
        for filename in s.get("files", []):
            if filename not in available_files:
                raise ValueError(
                    f"Step {sid}: file '{filename}' is not among the attached "
                    f"documents. Available files: {sorted(available_files)}"
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


def _describe_failed_step_outputs(
    old_plan: list[dict[str, Any]],
    old_results: dict[int, str],
    step_verifications: dict[int, dict],
) -> str:
    """Render a preview of each FAILED step's actual output, so the
    orchestrator replans informed by what the worker produced rather than
    the verifier's verdict alone (fixes F3 — "replans partially blind").
    """
    lines = []
    for s in old_plan:
        sid = s["step"]
        verdict = step_verifications.get(sid, {}).get("verdict", "")
        output = old_results.get(sid, "")
        if verdict != "FAILED" or not output:
            continue
        preview = output[:_FAILED_OUTPUT_PREVIEW_CHARS]
        if len(output) > _FAILED_OUTPUT_PREVIEW_CHARS:
            preview += "..."
        lines.append(f"--- Step {sid} ({s.get('subtask', '')}) ---\n{preview}")
    if not lines:
        return ""
    return "### Output of failed steps\n" + "\n\n".join(lines)


def _carry_over_passed_steps(
    new_plan: list[dict[str, Any]],
    old_plan: list[dict[str, Any]],
    old_results: dict[int, str],
    step_verifications: dict[int, dict],
) -> tuple[dict[int, str], dict[int, dict]]:
    """Determine which steps of a revised plan can keep their previous
    output instead of re-running (fixes F3 — replan no longer wipes
    already-approved research).

    A step is carried over only if it kept the SAME id and SAME subtask
    text across the revision AND was previously verified as PASSED or
    PASSED WITH NOTES. Anything else (renumbered, reworded, still FAILED,
    or never verified) is left out and simply re-runs — the safe fallback
    if the LLM doesn't honor the stable-id instruction.
    """
    old_by_id = {s["step"]: s for s in old_plan}
    kept_results: dict[int, str] = {}
    kept_verifications: dict[int, dict] = {}

    for s in new_plan:
        sid = s["step"]
        old_step = old_by_id.get(sid)
        if old_step is None or old_step.get("subtask") != s.get("subtask"):
            continue
        verdict = step_verifications.get(sid, {}).get("verdict", "")
        if verdict not in ("PASSED", "PASSED WITH NOTES"):
            continue
        if sid not in old_results:
            continue
        kept_results[sid] = old_results[sid]
        kept_verifications[sid] = step_verifications[sid]

    return kept_results, kept_verifications


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
    files = state.get("files", {})
    old_plan = state.get("plan", [])
    old_results = state.get("results", {})
    step_verifications = state.get("step_verifications", {})

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

    files_block = _describe_attached_files(files)
    if files_block:
        user_parts.append(files_block)

    if verifier_report:
        failed_outputs_block = _describe_failed_step_outputs(
            old_plan, old_results, step_verifications
        )
        user_parts.append(
            f"## Verifier report (REPLANNING PASS #{replan_count + 1} of {_MAX_REPLANS})\n"
            f"{verifier_report}\n\n"
            f"### Verifier notes\n{verification_notes}\n\n"
            + (f"{failed_outputs_block}\n\n" if failed_outputs_block else "")
            + f"Revise the plan to address ONLY the gaps flagged above. "
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
        _validate_plan(plan, set(files.keys()))

        if len(plan) == 0:
            return {"plan": plan, "results": {}, "current_step": 0}

        log_event("orchestrator_agent_plan", pipeline_plan=plan)

        # When replanning: clear stale step results from the previous plan
        # so set-based routing sees the new plan's steps as uncompleted —
        # except steps the verifier already approved, which carry their
        # output and verification forward if they kept the same id and
        # subtask text (fixes F3). The sentinel key -1 tells the results
        # reducer to replace, not merge.
        if verifier_report:
            kept_results, kept_verifications = _carry_over_passed_steps(
                plan, old_plan, old_results, step_verifications
            )
            return {
                "plan": plan,
                "results": {-1: "", 0: user_task, **kept_results},   # clear sentinel
                "current_step": 0,
                "step_verifications": kept_verifications,
            }

        return {"plan": plan, "results": {0: user_task}, "current_step": 0}

    except Exception as e:
        raise ValueError(f"Failed to parse JSON response: {e}")