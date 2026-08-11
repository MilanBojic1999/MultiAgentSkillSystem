"""Unit tests for the 4.10 minimal evaluation harness.

No network, no LLM — plan-shape assertions and task-loading are pure
functions exercised against the canonical plan fixtures in ``tests/plans.py``.
"""

import pytest
import yaml

from evals.runner import (
    ASSERTIONS,
    check_plan,
    load_tasks,
    run_evals,
    stub_worker,
)
from tests.plans import LINEAR_PLAN, WIDE_PLAN


# ---------------------------------------------------------------------------
# check_plan
# ---------------------------------------------------------------------------

def test_check_plan_all_assertions_pass():
    """A well-formed plan passing every assertion."""
    expect = {
        "min_steps": 2,
        "max_steps": 4,
        "agents_include": ["researcher"],
        "has_dependency": True,
    }
    res = check_plan(LINEAR_PLAN, expect)
    assert res["passed"] is True
    details = res["details"]

    # min_steps: 3 >= 2
    assert details["min_steps"] == (True, 3, 2)
    # max_steps: 3 <= 4
    assert details["max_steps"] == (True, 3, 4)
    # agents_include: researcher is in every step
    assert details["agents_include"] == (True, ["researcher"], ["researcher"])
    # has_dependency: steps 2 and 3 have depends_on
    assert details["has_dependency"] == (True, True, True)


def test_check_plan_min_steps_fails():
    """Plan too short → min_steps fails."""
    expect = {"min_steps": 10}
    res = check_plan(LINEAR_PLAN, expect)
    assert res["passed"] is False
    ok, actual, expected = res["details"]["min_steps"]
    assert ok is False
    assert actual == 3
    assert expected == 10


def test_check_plan_max_steps_fails():
    """Plan too long → max_steps fails."""
    expect = {"max_steps": 1}
    res = check_plan(LINEAR_PLAN, expect)
    assert res["passed"] is False
    ok, actual, expected = res["details"]["max_steps"]
    assert ok is False
    assert actual == 3
    assert expected == 1


def test_check_plan_agents_include_fails():
    """Missing agent → agents_include fails."""
    expect = {"agents_include": ["mathematician"]}
    res = check_plan(LINEAR_PLAN, expect)  # LINEAR_PLAN uses "researcher" only
    assert res["passed"] is False
    ok, actual, expected = res["details"]["agents_include"]
    assert ok is False
    assert "mathematician" not in actual
    assert expected == ["mathematician"]


def test_check_plan_has_dependency_true_passes():
    """Plan with dependencies, expect True → pass."""
    expect = {"has_dependency": True}
    res = check_plan(LINEAR_PLAN, expect)  # steps 2 and 3 depend on others
    assert res["passed"] is True
    assert res["details"]["has_dependency"] == (True, True, True)


def test_check_plan_has_dependency_false_passes():
    """Plan with no dependencies, expect False → pass."""
    # Build a plan where no step has depends_on
    no_dep_plan = [
        {"step": 1, "subtask": "a", "agent": "researcher",
         "skills_needed": [], "depends_on": []},
        {"step": 2, "subtask": "b", "agent": "writer",
         "skills_needed": [], "depends_on": []},
    ]
    expect = {"has_dependency": False}
    res = check_plan(no_dep_plan, expect)
    assert res["passed"] is True
    assert res["details"]["has_dependency"] == (True, False, False)


def test_check_plan_has_dependency_false_fails():
    """Plan with dependencies, expect False → fail."""
    expect = {"has_dependency": False}
    res = check_plan(LINEAR_PLAN, expect)  # LINEAR_PLAN has dependencies
    assert res["passed"] is False
    ok, actual, expected = res["details"]["has_dependency"]
    assert ok is False
    assert actual is True
    assert expected is False


def test_check_plan_unknown_key_ignored():
    """Unknown expect keys are silently ignored."""
    expect = {"bogus_key": 99, "min_steps": 1}
    res = check_plan(LINEAR_PLAN, expect)
    assert res["passed"] is True
    assert "bogus_key" not in res["details"]
    assert "min_steps" in res["details"]


def test_check_plan_none_plan_degraded_to_empty():
    """None plan → treated as [], assertions fail naturally."""
    expect = {"min_steps": 1}
    res = check_plan(None, expect)
    assert res["passed"] is False
    ok, actual, _expected = res["details"]["min_steps"]
    assert actual == 0


def test_check_plan_empty_expect_all_pass():
    """Empty expect dict → vacuously passed."""
    res = check_plan(LINEAR_PLAN, {})
    assert res["passed"] is True
    assert res["details"] == {}


# ---------------------------------------------------------------------------
# stub_worker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 5, 99])
def test_stub_worker_contract(n):
    """stub_worker returns the expected dict-merge payload."""
    state = {"step": {"step": n}, "results": {}, "current_datetime": ""}
    out = stub_worker(state)
    assert out == {"results": {n: f"eval-out-{n}"}}


# ---------------------------------------------------------------------------
# ASSERTIONS registry
# ---------------------------------------------------------------------------

def test_all_assertion_keys_are_registered():
    """Every key the YAML files can declare has a predicate."""
    assert "min_steps" in ASSERTIONS
    assert "max_steps" in ASSERTIONS
    assert "agents_include" in ASSERTIONS
    assert "has_dependency" in ASSERTIONS


# ---------------------------------------------------------------------------
# load_tasks
# ---------------------------------------------------------------------------

_VALID_YAML = """\
task: "explain something"
expect:
  min_steps: 2
  max_steps: 5
  agents_include: [researcher]
  has_dependency: true
"""


def test_load_tasks_parses_valid_file(tmp_path):
    (tmp_path / "good.yaml").write_text(_VALID_YAML)
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    name, task = tasks[0]
    assert name == "good"
    assert task["task"] == "explain something"
    assert task["expect"] == {
        "min_steps": 2,
        "max_steps": 5,
        "agents_include": ["researcher"],
        "has_dependency": True,
    }


def test_load_tasks_skips_broken_yaml(tmp_path):
    (tmp_path / "good.yaml").write_text(_VALID_YAML)
    (tmp_path / "bad.yaml").write_text("not: [valid: yaml: oops")
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0][0] == "good"


def test_load_tasks_skips_non_dict(tmp_path):
    (tmp_path / "list.yaml").write_text("- item1\n- item2\n")
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 0


def test_load_tasks_skips_missing_task(tmp_path):
    (tmp_path / "no_task.yaml").write_text("expect:\n  min_steps: 1\n")
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 0


def test_load_tasks_skips_empty_task(tmp_path):
    (tmp_path / "empty_task.yaml").write_text("task: ''\nexpect: {}\n")
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 0


def test_load_tasks_defaults_missing_expect(tmp_path):
    (tmp_path / "no_expect.yaml").write_text("task: 'smoke test'\n")
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    _name, task = tasks[0]
    assert task["task"] == "smoke test"
    assert task["expect"] == {}


def test_load_tasks_skips_bad_expect_type(tmp_path):
    (tmp_path / "bad_expect.yaml").write_text(
        "task: 'test'\nexpect: [1, 2, 3]\n"
    )
    tasks = load_tasks(tmp_path)
    assert len(tasks) == 0


def test_load_tasks_sorted_by_stem(tmp_path):
    (tmp_path / "c.yaml").write_text("task: 'c'\n")
    (tmp_path / "a.yaml").write_text("task: 'a'\n")
    (tmp_path / "b.yaml").write_text("task: 'b'\n")
    tasks = load_tasks(tmp_path)
    names = [n for n, _ in tasks]
    assert names == ["a", "b", "c"]


def test_load_tasks_empty_dir(tmp_path):
    assert load_tasks(tmp_path) == []


def test_load_tasks_missing_dir():
    from pathlib import Path
    with pytest.raises(ValueError, match="Tasks directory not found"):
        load_tasks(Path("/nonexistent/path/evals/tasks"))


# ---------------------------------------------------------------------------
# run_evals — smoke (returns before building graph)
# ---------------------------------------------------------------------------

def test_run_evals_empty_dir_returns_false(tmp_path, capsys):
    """Empty tasks dir → False + stderr message."""
    ok = run_evals("parallel", tmp_path)
    assert ok is False
    captured = capsys.readouterr()
    assert "no tasks found" in captured.err
