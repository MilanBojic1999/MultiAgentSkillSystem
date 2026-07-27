"""Plan 1.2 acceptance criteria for utils.plan_validator.validate_plan.

Pure function, no mocking. Error *messages* are part of the contract (they are
what the orchestrator's re-planning loop and a debugging developer see), so we
assert on them with ``match=``.
"""

import pytest

from utils.plan_validator import PlanValidationError, validate_plan
from tests.plans import CYCLIC_PLAN, KNOWN_AGENTS, KNOWN_SKILLS, LINEAR_PLAN, step


def _validate(plan):
    return validate_plan(plan, KNOWN_AGENTS, KNOWN_SKILLS)


def test_valid_linear_plan_returns_sorted_plain_dicts():
    out = _validate(LINEAR_PLAN)
    assert isinstance(out, list)
    assert all(isinstance(d, dict) for d in out)
    assert [d["step"] for d in out] == [1, 2, 3]


def test_out_of_order_plan_is_sorted_and_optional_keys_default():
    raw = [
        {"step": 2, "subtask": "b", "agent": "writer", "depends_on": [1]},
        {"step": 1, "subtask": "a", "agent": "researcher"},
    ]
    out = _validate(raw)
    assert [d["step"] for d in out] == [1, 2]
    assert out[0]["skills_needed"] == []
    assert out[0]["depends_on"] == []


def test_plan_validation_error_is_a_value_error():
    # The orchestrator's RetryPolicy retries on ValueError, so this subclassing
    # is what makes bad plans trigger a re-plan.
    assert issubclass(PlanValidationError, ValueError)


@pytest.mark.parametrize("bad", [{}, [], "not a list"])
def test_non_list_or_empty_raises(bad):
    with pytest.raises(PlanValidationError, match="non-empty list"):
        _validate(bad)


@pytest.mark.parametrize(
    "bad_step",
    [
        {"step": 1, "agent": "researcher"},                       # missing subtask
        {"step": 0, "subtask": "x", "agent": "researcher"},       # step < 1
        {"step": 1, "subtask": "", "agent": "researcher"},        # empty subtask
    ],
)
def test_schema_violation_names_the_index(bad_step):
    with pytest.raises(PlanValidationError, match="index 0"):
        _validate([bad_step])


def test_duplicate_step_numbers_raise():
    plan = [step(1), step(1, agent="writer")]
    with pytest.raises(PlanValidationError, match="Duplicate step numbers"):
        _validate(plan)


def test_unknown_agent_raises_and_lists_known_agents():
    with pytest.raises(PlanValidationError, match="unknown agent 'nonexistent'"):
        _validate([step(1, agent="nonexistent")])


def test_unknown_skill_is_dropped_not_raised():
    plan = [step(1, skills=["roll-dice", "bogus-skill"])]
    out = _validate(plan)
    assert out[0]["skills_needed"] == ["roll-dice"]


def test_self_dependency_raises():
    with pytest.raises(PlanValidationError, match="depends on itself"):
        _validate([step(1), step(2, deps=[2])])


def test_dangling_dependency_names_the_missing_step():
    with pytest.raises(PlanValidationError, match="99"):
        _validate([step(1, deps=[99])])


def test_cycle_raises_plan_validation_error_listing_steps():
    with pytest.raises(PlanValidationError, match=r"cycle detected.*\[1, 2\]"):
        _validate(CYCLIC_PLAN)
