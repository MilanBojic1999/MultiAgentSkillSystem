"""execution_policy — the effort-slider contract, pinned.

These tests are hermetic: the policy module imports nothing project-local, so
no LLM, env or CWD concerns apply. Every preset budget is pinned exactly —
tuning the table requires updating this file deliberately, which is the point
(effort_plan.md step 1: "explicit, documented, test-pinned defaults").
"""

import pytest

from execution_policy import (
    DEFAULT_EFFORT,
    EFFORT_PRESETS,
    PRESET_BUDGETS,
    ExecutionPolicy,
    VERIFICATION_EFFORTS,
    deadline_exceeded,
    effective_verification_attempts,
    effective_worker_attempts,
    graph_effort_compat_error,
    normalize_effort,
    policy_from_config,
    policy_from_state,
    resolve_execution_policy,
    stamp_deadline,
)

# Every preset budget, pinned field by field (the effort_plan.md table).
PINNED_BUDGETS = {
    "instant": {
        "plan_enabled": False,
        "instant_writer_only": True,
        "max_plan_steps": 1,
        "max_worker_attempts": 1,
        "max_tool_calls_per_attempt": 0,
        "max_verification_attempts": 0,
        "max_step_verification_retries": 0,
        "max_replans": 0,
        "max_graph_dispatches": 1,
        "react_recursion_limit": 4,
        "timeout_seconds": 120,
    },
    "light": {
        "plan_enabled": True,
        "instant_writer_only": False,
        "max_plan_steps": 3,
        "max_worker_attempts": 1,
        "max_tool_calls_per_attempt": 2,
        "max_verification_attempts": 1,
        "max_step_verification_retries": 0,
        "max_replans": 0,
        "max_graph_dispatches": 16,
        "react_recursion_limit": 10,
        "timeout_seconds": 300,
    },
    "standard": {
        "plan_enabled": True,
        "instant_writer_only": False,
        "max_plan_steps": 8,
        "max_worker_attempts": 2,
        "max_tool_calls_per_attempt": 5,
        "max_verification_attempts": 1,
        "max_step_verification_retries": 1,
        "max_replans": 1,
        "max_graph_dispatches": 64,
        "react_recursion_limit": 25,
        "timeout_seconds": 900,
    },
    "thorough": {
        "plan_enabled": True,
        "instant_writer_only": False,
        "max_plan_steps": 16,
        "max_worker_attempts": 3,
        "max_tool_calls_per_attempt": 10,
        "max_verification_attempts": 2,
        "max_step_verification_retries": 2,
        "max_replans": 2,
        "max_graph_dispatches": 128,
        "react_recursion_limit": 50,
        "timeout_seconds": 1800,
    },
    "unlimited": {
        "plan_enabled": True,
        "instant_writer_only": False,
        "max_plan_steps": 64,
        "max_worker_attempts": 10,
        "max_tool_calls_per_attempt": 100,
        "max_verification_attempts": 10,
        "max_step_verification_retries": 2,
        # 2 actual replan passes = the historical three-planner-pass ceiling
        "max_replans": 2,
        "max_graph_dispatches": 256,
        "react_recursion_limit": 100,
        "timeout_seconds": 3600,
    },
}


def _policy(effort="standard"):
    return resolve_execution_policy(effort, now=0.0)


# ---------------------------------------------------------------------------
# Normalization and default resolution
# ---------------------------------------------------------------------------

def test_preset_names_are_canonical_lowercase():
    assert EFFORT_PRESETS == ("instant", "light", "standard", "thorough", "unlimited")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("instant", "instant"),
        ("Instant", "instant"),
        ("INSTANT", "instant"),
        ("  standard  ", "standard"),
        ("", DEFAULT_EFFORT),
        (None, DEFAULT_EFFORT),
    ],
)
def test_effort_input_is_normalized_case_insensitively(raw, expected):
    assert normalize_effort(raw) == expected


@pytest.mark.parametrize("raw", ["extreme", "SUPER", "lightning", 3, ["light"]])
def test_invalid_effort_raises_naming_valid_presets(raw):
    with pytest.raises(ValueError, match="instant"):
        normalize_effort(raw)


def test_omitted_effort_resolves_to_unlimited():
    policy = resolve_execution_policy(None, now=0.0)
    assert policy.preset == "unlimited"
    # compatibility contract: the default must preserve current yotta caps
    assert policy.max_step_verification_retries == 2
    assert policy.max_replans == 2
    # and current per-agent attempts pass through (policy ceiling = config bound)
    assert policy.max_worker_attempts == 10


@pytest.mark.parametrize("preset", EFFORT_PRESETS)
def test_every_preset_budget_is_exactly_pinned(preset):
    """Every field of every preset is pinned — tuning the table must edit
    this test deliberately."""
    policy = resolve_execution_policy(preset, now=0.0)
    pinned = PINNED_BUDGETS[preset]
    assert policy.preset == preset
    for field, expected in pinned.items():
        assert getattr(policy, field) == expected, f"{preset}.{field}"
    # the public table carries the same values
    for field, expected in pinned.items():
        assert PRESET_BUDGETS[preset][field] == expected, f"PRESET_BUDGETS[{preset}][{field}]"


def test_instant_guarantees_single_writer_no_verification():
    p = _policy("instant")
    assert not p.plan_enabled
    assert p.instant_writer_only
    assert p.max_worker_attempts == 1
    assert p.max_tool_calls_per_attempt == 0
    assert p.max_verification_attempts == 0
    assert p.max_step_verification_retries == 0
    assert p.max_replans == 0


def test_worker_verification_and_replan_budgets_are_separate_counters():
    """The worker, verification and replan budgets are separate fields with
    independent counters (effort_plan.md: 'Worker retries and full replans are
    separate budgets'). Values may coincide across presets; what never happens
    is two budgets sharing one field or one consumption path."""
    # separate fields: every budget travels under its own key in the
    # serialized policy (the state/config transport)
    p = _policy("thorough")
    data = p.as_dict()
    for budget in ("max_worker_attempts", "max_verification_attempts",
                   "max_step_verification_retries", "max_replans"):
        assert budget in data, f"{budget} must be its own policy field"

    # independent counters: each effective-count helper reads only its own
    # field (thorough: worker 3, verifier 2, each capped independently)
    assert effective_worker_attempts(p, 5) == 3          # worker budget only
    assert effective_verification_attempts(p, 5) == 2    # verifier budget only

    # meaningful inequalities that hold keep their pin
    for preset in ("standard", "thorough"):
        p = _policy(preset)
        assert p.max_worker_attempts != p.max_verification_attempts  # 2!=1, 3!=2
    p = _policy("thorough")
    assert p.max_worker_attempts == 3
    assert p.max_verification_attempts == 2
    assert p.max_step_verification_retries == 2
    assert p.max_replans == 2


def test_normal_modes_keep_a_real_verification_loop():
    for preset in VERIFICATION_EFFORTS:
        p = _policy(preset)
        assert p.plan_enabled
        assert not p.instant_writer_only
        assert p.max_verification_attempts >= 1


def test_unlimited_has_only_finite_safety_bounds():
    """Unlimited means compatibility, not unboundedness: every ceiling is a
    finite int that a pathological run will eventually hit."""
    p = _policy("unlimited")
    for field in PINNED_BUDGETS["unlimited"]:
        value = getattr(p, field)
        if isinstance(value, int) and not isinstance(value, bool):
            assert 0 < value < 10_000, f"unlimited.{field} is not a finite ceiling"
    assert p.max_plan_steps == 64
    assert p.max_worker_attempts == 10  # the validated config bound
    assert p.max_replans == 2           # historical yotta: planner runs at most 3x
    assert p.max_graph_dispatches == 256


def test_no_preset_ever_claims_literal_unbounded_execution():
    for preset in EFFORT_PRESETS:
        p = _policy(preset)
        assert isinstance(p.max_graph_dispatches, int)
        assert isinstance(p.max_plan_steps, int)
        assert isinstance(p.react_recursion_limit, int)


# ---------------------------------------------------------------------------
# Static agent configuration intersection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "effort,configured,expected",
    [
        ("instant", 2, 1),      # Instant always one writer attempt
        ("light", 3, 1),        # researcher's configured 3 is clamped by light
        ("standard", 1, 1),     # configured below the policy cap wins
        ("standard", 2, 2),
        ("standard", 10, 2),
        ("unlimited", 3, 3),    # existing per-agent behavior passes through
        ("unlimited", 2, 2),
        ("unlimited", 10, 10),
    ],
)
def test_effective_attempts_is_min_of_policy_and_static_config(effort, configured, expected):
    policy = _policy(effort)
    assert effective_worker_attempts(policy, configured) == expected


def test_effective_attempts_accepts_serialized_policy_and_none():
    # serialized dict (checkpoint round-trip) behaves identically
    policy = _policy("standard")
    assert effective_worker_attempts(policy.as_dict(), 3) == 2
    # no policy (legacy parallel/sequential state) passes the configured count
    assert effective_worker_attempts(None, 3) == 3


def test_effective_verification_attempts_intersect_the_same_way():
    """The verifier's LLM call uses the same min() contract as workers, but
    against its own budget field."""
    from execution_policy import effective_verification_attempts

    assert effective_verification_attempts(_policy("light"), 2) == 1
    assert effective_verification_attempts(_policy("thorough"), 2) == 2
    assert effective_verification_attempts(_policy("unlimited"), 3) == 3
    assert effective_verification_attempts(_policy("instant"), 2) == 0


# ---------------------------------------------------------------------------
# Serialization and extraction
# ---------------------------------------------------------------------------

def test_policy_dict_round_trip_is_exact():
    p = _policy("thorough")
    again = ExecutionPolicy.from_dict(p.as_dict())
    assert again == p
    assert again.as_dict() == p.as_dict()


def test_serialized_policy_is_plain_json_safe():
    import json

    data = json.loads(json.dumps(_policy("standard").as_dict()))
    assert data["max_plan_steps"] == 8
    assert data["deadline"] == 900.0  # now=0 + timeout_seconds


def test_from_dict_rejects_missing_or_mistyped_fields():
    good = _policy("standard").as_dict()
    broken = dict(good)
    del broken["max_plan_steps"]
    with pytest.raises(ValueError, match="max_plan_steps"):
        ExecutionPolicy.from_dict(broken)
    broken = dict(good)
    broken["max_replans"] = "many"
    with pytest.raises(ValueError, match="max_replans"):
        ExecutionPolicy.from_dict(broken)
    broken = dict(good)
    broken["plan_enabled"] = 1  # bool field must not accept an int
    with pytest.raises(ValueError, match="plan_enabled"):
        ExecutionPolicy.from_dict(broken)


def test_from_dict_ignores_unknown_keys_for_forward_compat():
    data = {**_policy("standard").as_dict(), "future_field": 1}
    assert ExecutionPolicy.from_dict(data).preset == "standard"


def test_policy_from_config_reads_configurable_and_defaults_to_unlimited():
    p = _policy("light")
    config = {"configurable": {"thread_id": "t", "execution_policy": p.as_dict()}}
    assert policy_from_config(config).preset == "light"

    # configurable with only the effort name resolves through the resolver
    config = {"configurable": {"thread_id": "t", "effort": "Thorough"}}
    assert policy_from_config(config).preset == "thorough"

    # plain old clients (no policy at all) get the backwards-compatible default
    assert policy_from_config(None).preset == "unlimited"
    assert policy_from_config({"configurable": {"thread_id": "t"}}).preset == "unlimited"


def test_policy_from_state_reads_state_and_defaults_to_unlimited():
    p = _policy("standard")
    assert policy_from_state({"execution_policy": p.as_dict()}).preset == "standard"
    assert policy_from_state({"effort": "instant"}).preset == "instant"
    assert policy_from_state({"task": "t"}).preset == "unlimited"
    assert policy_from_state(None).preset == "unlimited"
    # a dataclass in state (tests, same-process injection) also works
    assert policy_from_state({"execution_policy": p}).preset == "standard"


# ---------------------------------------------------------------------------
# Wall-clock deadline
# ---------------------------------------------------------------------------

def test_resolve_stamps_deadline_from_timeout_budget():
    p = resolve_execution_policy("standard", now=100.0)
    assert p.deadline == 100.0 + p.timeout_seconds


def test_deadline_not_exceeded_before_and_exceeded_after():
    p = resolve_execution_policy("light", now=0.0)
    assert not deadline_exceeded(p, now=p.deadline)
    assert deadline_exceeded(p, now=p.deadline + 0.001)


def test_policy_without_deadline_is_never_exceeded_until_stamped():
    p = ExecutionPolicy.from_dict(
        {k: v for k, v in _policy("light").as_dict().items() if k != "deadline"}
    )
    assert p.deadline is None
    assert not deadline_exceeded(p, now=1e12)
    stamped = stamp_deadline(p, now=10.0)
    assert stamped.deadline == 10.0 + p.timeout_seconds
    assert deadline_exceeded(stamped, now=stamped.deadline + 1)


def test_stamp_deadline_never_mutates_the_original():
    p = _policy("standard")
    stamped = stamp_deadline(p, now=0.0)
    assert stamped is not p
    assert p.deadline == 900.0  # unchanged from resolve(now=0)


# ---------------------------------------------------------------------------
# Compatibility topology rule
# ---------------------------------------------------------------------------

def test_legacy_graphs_accept_only_unlimited_and_instant():
    assert graph_effort_compat_error("parallel", "unlimited") is None
    assert graph_effort_compat_error("sequential", "unlimited") is None
    assert graph_effort_compat_error("parallel", "instant") is None
    assert graph_effort_compat_error("sequential", "instant") is None
    assert graph_effort_compat_error("parallel", None) is None  # default unlimited


@pytest.mark.parametrize("effort", VERIFICATION_EFFORTS)
def test_legacy_graphs_reject_verification_promising_efforts(effort):
    err = graph_effort_compat_error("parallel", effort)
    assert err is not None
    assert "verification" in err
    assert "yotta" in err
    assert graph_effort_compat_error("sequential", effort) is not None


def test_yotta_and_unknown_graphs_have_no_compat_error():
    for effort in EFFORT_PRESETS:
        assert graph_effort_compat_error("yotta", effort) is None
    assert graph_effort_compat_error(None, "standard") is None
    assert graph_effort_compat_error("", "standard") is None
