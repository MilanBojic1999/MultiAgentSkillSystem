from typing import Annotated
from typing_extensions import TypedDict, NotRequired
import operator
from datetime import datetime


def get_current_datetime_str() -> str:
    """Return the current datetime as a human-friendly long-form string."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y at %H:%M:%S %Z")


# ---------------------------------------------------------------------------
# Sentinel key used in ``results`` to signal "replace, don't merge."
# When a node returns ``results={-1: "", ...}`` the reducer discards all
# prior results and keeps only the new entries (minus the sentinel).
# ---------------------------------------------------------------------------
_CLEAR_RESULTS = -1


def _results_reducer(current: dict[int, str], update: dict[int, str]) -> dict[int, str]:
    """Merge step results, with support for clearing via the ``-1`` sentinel key.

    Normal path (no sentinel): ``update`` is merged into ``current``
    (newer keys overwrite older ones).

    Clear path (sentinel present): ``current`` is **replaced** by
    ``update`` (minus the sentinel).  Used by the orchestrator when
    replanning so stale step outputs from a previous plan don't leak
    into the new one.
    """
    if _CLEAR_RESULTS in update:
        return {k: v for k, v in update.items() if k != _CLEAR_RESULTS}
    return {**current, **update}


class PlanStep(TypedDict):
    step: int
    subtask: str
    agent: str
    skills_needed: list[str]
    depends_on: list[int]
    # Filenames (keys into AgentState.files) this step needs the full text of.
    # Optional — most steps don't touch attached documents.
    files: NotRequired[list[str]]

class AgentState(TypedDict):
    # Inputs
    task: str

    # Attached documents: filename -> decoded/truncated text content.
    # Set once at pipeline entry (api_server.py / streaming.py) and never
    # mutated by nodes — the planner assigns filenames to steps via
    # PlanStep.files, and run_sub_agent_async injects the matching content.
    files: dict[str, str]

    # Current datetime (human-friendly) — set at pipeline start, available to all nodes
    current_datetime: str

    # Whether the pipeline is running in streaming mode (affects how outputs are handled)
    streaming: bool

    # Set by Orchestrator node
    plan: list[PlanStep]

    # Accumulated by sub-agent nodes; reducer merges dicts.
    # Key ``-1`` is reserved as a clear-sentinel — see ``_results_reducer``.
    results: Annotated[dict[int, str], _results_reducer]

    # Which step is currently executing (used by router)
    current_step: int

    # Final assembled output
    final_output: str

    # Set by the verifier node — drives conditional routing to assemble vs re-orchestrate
    verification_result: str
    verification_notes: str
    verifier_report: str

    # Single routing decision computed by verify_node: "retry" | "replan" | "proceed".
    # after_verify reads this directly instead of re-parsing verdicts/notes.
    verification_route: str

    replan_count: int

    # Per-step verification tracking, accumulated across verify cycles.
    # Maps plan step number → {"verdict": str, "notes": str, "retries": int}
    # Cleared on replan since new plan = new step IDs.
    step_verifications: dict[int, dict]

    # Initial search / grounding results (e.g. yotta) — injected before orchestration,
    # consumed by orchestrator and writer nodes without embedding them in ``task``.
    search_results: str

    # Synthetic step used by verify / writer nodes (not part of the plan)
    step: dict