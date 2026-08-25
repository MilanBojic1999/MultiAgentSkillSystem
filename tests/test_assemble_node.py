"""assemble_node.assemble_node — output assembly formatting (pure function).

Also covers the Slice 4 result-derivation helpers:
``derive_status``, ``skipped_steps_from_stats`` and ``pipeline_result``.
"""

from assemble_node import (
    assemble_node,
    derive_status,
    pipeline_result,
    skipped_steps_from_stats,
)
from tests.plans import LINEAR_PLAN


def test_sections_are_ordered_per_plan_step():
    results = {1: "one", 2: "two", 3: "three"}
    out = assemble_node({"plan": LINEAR_PLAN, "results": results})["final_output"]
    assert out == (
        "## Step 1: subtask 1\none\n\n"
        "## Step 2: subtask 2\ntwo\n\n"
        "## Step 3: subtask 3\nthree"
    )


def test_missing_result_renders_as_empty_body():
    out = assemble_node({"plan": LINEAR_PLAN, "results": {1: "one"}})["final_output"]
    assert "## Step 2: subtask 2\n\n" in out
    assert out.endswith("## Step 3: subtask 3\n")


def test_empty_plan_yields_empty_output():
    assert assemble_node({"plan": [], "results": {}})["final_output"] == ""


def test_failed_steps_prepend_a_warning_header():
    """Plan 4.13: assemble_node emits a warning when ``failed_steps`` is non-empty."""
    state = {
        "plan": LINEAR_PLAN,
        "results": {1: "one", 2: "[STEP FAILED] boom", 3: "three"},
        "failed_steps": [2],
    }
    out = assemble_node(state)["final_output"]
    assert out.startswith("> ⚠️")
    assert "1 of 3 steps failed" in out.split("\n")[0]
    assert "steps 2" in out.split("\n")[0]
    assert "output below is partial" in out.split("\n")[0]
    # Step bodies still follow the warning
    assert "## Step 1: subtask 1\none" in out


def test_failed_steps_warning_not_emitted_when_none_failed():
    """No warning when failed_steps is missing or empty."""
    state = {"plan": LINEAR_PLAN, "results": {1: "one", 2: "two", 3: "three"}}
    out = assemble_node(state)["final_output"]
    assert "⚠️" not in out
    assert out.startswith("## Step 1")


# ---------------------------------------------------------------------------
# Slice 4 — status derivation and the typed pipeline result
# ---------------------------------------------------------------------------

def _row(step, status):
    return {
        "step": step,
        "agent": "researcher",
        "status": status,
        "duration_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": 0,
    }


def test_derive_status_completed_when_nothing_failed_or_skipped():
    stats = [_row(1, "completed"), _row(2, "completed")]
    assert derive_status([], stats) == "completed"


def test_derive_status_partial_on_failed_steps():
    stats = [_row(1, "completed"), _row(2, "failed")]
    assert derive_status([2], stats) == "partial"


def test_derive_status_partial_on_skipped_steps():
    """A skipped (transitively blocked) step also makes the run partial."""
    stats = [_row(1, "completed"), _row(2, "skipped")]
    assert derive_status([], stats) == "partial"


def test_skipped_steps_from_stats_sorted():
    stats = [_row(3, "skipped"), _row(1, "completed"), _row(2, "skipped")]
    assert skipped_steps_from_stats(stats) == [2, 3]


def test_assemble_writes_completed_status():
    out = assemble_node({"plan": LINEAR_PLAN, "results": {1: "one", 2: "two", 3: "three"}})
    assert out["status"] == "completed"


def test_assemble_writes_partial_status_on_failure():
    state = {
        "plan": LINEAR_PLAN,
        "results": {1: "one", 2: "[STEP FAILED] boom", 3: "three"},
        "failed_steps": [2],
        "step_stats": [_row(1, "completed"), _row(2, "failed"), _row(3, "skipped")],
    }
    out = assemble_node(state)
    assert out["status"] == "partial"
    # Warning text stays human-facing; the machine-readable status is separate.
    assert "⚠️" in out["final_output"]


def test_pipeline_result_normalizes_terminal_state():
    """``pipeline_result`` sorts steps, derives skipped steps, and reads status."""
    state = {
        "final_output": "done",
        "status": "partial",
        "failed_steps": [2, 1],  # unsorted — presentation sorts
        "step_stats": [_row(3, "skipped"), _row(1, "completed"), _row(2, "failed")],
    }
    result = pipeline_result(state)
    assert result["status"] == "partial"
    assert result["final_output"] == "done"
    assert result["failed_steps"] == [1, 2]
    assert result["skipped_steps"] == [3]
    assert [s["step"] for s in result["step_stats"]] == [1, 2, 3]


def test_pipeline_result_falls_back_to_derived_status():
    """Graph modules that do not write ``status`` still get a derived one."""
    state = {"final_output": "x", "failed_steps": [1], "step_stats": [_row(1, "failed")]}
    assert pipeline_result(state)["status"] == "partial"
    state = {"final_output": "x", "failed_steps": [], "step_stats": [_row(1, "completed")]}
    assert pipeline_result(state)["status"] == "completed"


# ---------------------------------------------------------------------------
# Effort slider — verification exhaustion and response metadata
# ---------------------------------------------------------------------------

def test_derive_status_partial_on_verification_exhaustion():
    """An exhausted verification budget is a partial result even when no step
    failed or was skipped — the writer still synthesized the best available
    output, but it was never fully verified."""
    stats = [_row(1, "completed")]
    assert derive_status([], stats) == "completed"
    assert derive_status([], stats, verification_exhausted=True) == "partial"
    assert derive_status([2], stats, verification_exhausted=True) == "partial"


def test_assemble_writes_partial_on_verification_exhaustion():
    out = assemble_node({
        "plan": LINEAR_PLAN,
        "results": {1: "one", 2: "two", 3: "three"},
        "verification_exhausted": True,
    })
    assert out["status"] == "partial"


def test_pipeline_result_carries_effort_metadata_with_safe_defaults():
    """Graph modules that predate the effort slider still normalize cleanly."""
    state = {"final_output": "done", "failed_steps": [], "step_stats": []}
    result = pipeline_result(state)
    assert result["effort"] == "unlimited"
    assert result["verification"] is None
    assert result["verification_exhausted"] is False
    assert result["replan_count"] == 0
    assert result["safety_stop_reason"] is None


def test_pipeline_result_normalizes_effort_metadata_from_state():
    state = {
        "final_output": "done",
        "status": "partial",
        "failed_steps": [],
        "step_stats": [],
        "effort": "standard",
        "verification_result": "PASSED WITH NOTES",
        "verification_exhausted": True,
        "replan_count": 1,
        "safety_stop_reason": None,
    }
    result = pipeline_result(state)
    assert result["effort"] == "standard"
    assert result["verification"] == "PASSED WITH NOTES"
    assert result["verification_exhausted"] is True
    assert result["replan_count"] == 1
    assert result["safety_stop_reason"] is None


def test_pipeline_result_surfaces_safety_stop_reason():
    state = {
        "final_output": "best available",
        "status": "partial",
        "failed_steps": [],
        "step_stats": [],
        "safety_stop_reason": "max_graph_dispatches=16 exceeded",
    }
    assert pipeline_result(state)["safety_stop_reason"] == "max_graph_dispatches=16 exceeded"
