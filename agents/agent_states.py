from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict
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
    # Filenames of attached documents the orchestrator assigned to this step;
    # the worker injects those documents' full text into its system prompt.
    # NotRequired: plans validated before this field existed omit it.
    files: NotRequired[list[str]]


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
    consumed by the CLI and the API response models. The effort-slider
    metadata (effort slider) has safe defaults so graph modules that predate
    it stay compatible.
    """
    status: ExecutionStatus
    final_output: str
    failed_steps: list[int]
    skipped_steps: list[int]
    step_stats: list[StepStats]
    # Effort slider metadata
    effort: str
    verification: str | None
    verification_exhausted: bool
    replan_count: int
    safety_stop_reason: str | None


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

    ``files`` carries the run's attached documents (``{filename: text}``) so the
    worker can inject the full text of the ones its step was assigned — nothing
    outside the payload is visible to it. ``NotRequired`` because only the yotta
    router populates it; parallel/sequential have no ``files`` channel.
    """
    step: PlanStep
    results: dict[int, str]
    current_datetime: str
    files: NotRequired[dict[str, str]]


RESULTS_RESET = "__reset__"  # sentinel: clear results on replan


class YottaState(AgentState):
    """State for the yotta graph (``graphs/yotta_graph.py``).

    The only difference from ``AgentState`` is the ``results`` reducer: the
    shared ``{**a, **b}`` merge can never delete keys, so the orchestrator's
    ``results: {}`` wipe is a no-op and the replan loop would re-verify stale
    results forever. The reset-aware reducer lets ``verify_node`` clear
    ``results`` on the replan route by returning ``{RESULTS_RESET: ""}``
    (parallel/sequential keep the plain merge — untouched).
    """
    results: Annotated[dict[int, str],
                       lambda a, b: {} if RESULTS_RESET in b else {**a, **b}]
    streaming: bool
    files: dict[str, str]
    search_results: str
    step_verifications: dict[int, dict]
    verification_result: str
    verification_notes: str
    verifier_report: str
    replan_count: int
    verification_route: str
    # All workers of one pass write [] concurrently (the last write wins);
    # only verify_node writes a meaningful value, alone in its own superstep
    # after the scheduler barrier — so a plain channel would reject the
    # parallel workers' writes (InvalidUpdateError).
    pending_retries: Annotated[list[int], lambda a, b: b]
    # Effort slider (execution policy contract). ``effort_router`` writes the
    # canonical preset + serialized policy at entry; every budgeted node reads
    # them from here, and Send payloads carry both so workers see the same
    # budgets. All four counters are written by exactly one node each, in its
    # own superstep — plain channels, no reducers needed.
    effort: str
    execution_policy: dict
    dispatch_count: int
    safety_stop_reason: str
    verification_attempts: int
    verification_exhausted: bool