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


class WorkerState(TypedDict):
    """Input schema for a single parallel worker task (the ``Send`` payload).

    A worker sees only the step it must run plus the results accumulated so far;
    it writes ``results``/``failed_steps`` back into ``AgentState`` via that
    state's reducers, so no reducers are needed here.
    """
    step: PlanStep
    results: dict[int, str]
    current_datetime: str