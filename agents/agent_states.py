from typing import Annotated, Literal
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


StepStatus = Literal["completed", "failed", "skipped"]
"""Per-step terminal status (Slice 4)."""

ExecutionStatus = Literal["completed", "partial", "failed", "running"]
"""Terminal pipeline status (Slice 4).

- ``completed`` — every planned step completed successfully;
- ``partial`` — at least one step failed or was skipped, but the graph
  assembled usable output;
- ``failed`` — the planner, graph, configuration or infrastructure failed
  before a usable assembled result existed;
- ``running`` — an asynchronous run has not reached a terminal state.
"""


class StepStats(TypedDict):
    """Per-step execution statistics (Phase 4.9).

    Accumulated by worker nodes and scheduler nodes via ``operator.add`` on the
    ``step_stats`` list in ``AgentState``. Every step in the plan produces exactly
    one entry — either ``"completed"``, ``"failed"``, or ``"skipped"``.
    """
    step: int
    agent: str
    status: StepStatus
    duration_s: float
    input_tokens: int
    output_tokens: int
    tool_calls: int


class PipelineResult(TypedDict):
    """Typed pipeline result returned at presentation boundaries (Slice 4).

    Built by ``assemble_node.pipeline_result`` from a terminal graph state and
    consumed by the CLI and the API response models.
    """
    status: ExecutionStatus
    final_output: str
    failed_steps: list[int]
    skipped_steps: list[int]
    step_stats: list[StepStats]


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

    # Per-step execution statistics (Phase 4.9) — one entry per plan step
    step_stats: Annotated[list[StepStats], operator.add]

    # Final assembled output
    final_output: str

    # Terminal execution status, written by the assemble node (Slice 4)
    status: ExecutionStatus


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