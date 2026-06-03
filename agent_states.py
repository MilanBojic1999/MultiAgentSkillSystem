from typing import Annotated
from typing_extensions import TypedDict
import operator


class PlanStep(TypedDict):
    step: int
    subtask: str
    agent: str
    skills_needed: list[str]
    depends_on: list[int]

class AgentState(TypedDict):
    # Inputs
    task: str

    # Set by Orchestrator node
    plan: list[PlanStep]

    # Accumulated by sub-agent nodes; reducer merges dicts
    results: Annotated[dict[int, str], lambda a, b: {**a, **b}]

    # Which step is currently executing (used by router)
    current_step: int

    # Final assembled output
    final_output: str
