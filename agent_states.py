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

    # Final assembled output
    final_output: str

    # Set by the verifier node — drives conditional routing to assemble vs re-orchestrate
    verification_result: str
    verification_notes: str

    # Synthetic step used by verify / writer nodes (not part of the plan)
    step: dict