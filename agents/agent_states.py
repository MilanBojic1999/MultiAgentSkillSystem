from typing import Annotated
from typing_extensions import TypedDict
import operator
from datetime import datetime


def get_current_datetime_str() -> str:
    """Return the current datetime as a human-friendly long-form string."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y at %H:%M:%S %Z")


class PlanStep(TypedDict):
    step: int
    subtask: str
    agent: str
    skills_needed: list[str]
    depends_on: list[int]

class AgentState(TypedDict):
    # Inputs
    task: str

    # Current datetime (human-friendly) — set at pipeline start, available to all nodes
    current_datetime: str

    # Set by Orchestrator node
    plan: list[PlanStep]

    # Accumulated by sub-agent nodes; reducer merges dicts
    results: Annotated[dict[int, str], lambda a, b: {**a, **b}]

    # Which step is currently executing (used by router)
    current_step: int

    # Steps whose sub-agent failed after retries (Phase 1.4 failure containment)
    failed_steps: Annotated[list[int], operator.add]

    # Final assembled output
    final_output: str


def _transitive_dependents(plan: list[dict], failed: set[int]) -> set[int]:
    """Return every step transitively blocked by a failed step.

    Does not include the *failed* steps themselves — only the steps that depend
    on them, directly or transitively. Used by scheduler nodes to write
    ``[SKIPPED — dependency failed]`` markers before the router runs (Phase 4.13).
    """
    if not failed:
        return set()
    blocked = set(failed)
    changed = True
    while changed:
        changed = False
        for s in plan:
            if s["step"] not in blocked:
                if set(s.get("depends_on", [])) & blocked:
                    blocked.add(s["step"])
                    changed = True
    return blocked - failed


class WorkerState(TypedDict):
    """Input schema for a single parallel worker task (the ``Send`` payload).

    A worker sees only the step it must run plus the results accumulated so far;
    it writes ``results``/``failed_steps`` back into ``AgentState`` via that
    state's reducers, so no reducers are needed here.
    """
    step: PlanStep
    results: dict[int, str]
    current_datetime: str