"""assemble_node.assemble_node — output assembly formatting (pure function)."""

import pytest

from assemble_node import assemble_node
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
