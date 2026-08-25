"""Execution effort policy — the single source of per-run effort budgets.

The pipeline exposes a user-facing **effort slider** with five named presets::

    Instant -> Light -> Standard -> Thorough -> Unlimited

This module is the execution-policy contract, not an LLM-provider
``reasoning_effort`` parameter: ``llm_factory.create_llm`` targets arbitrary
OpenAI-compatible endpoints, so a provider-specific reasoning control would be
invalid or silently ignored. Effort is enforced by pipeline orchestration and
budgets (plan size, worker attempts, tool calls, verification, replans,
dispatch waves, ReAct recursion, wall-clock deadline).

Contract
--------
- The selection is **per run**; it is never stored as a server/user preference.
- ``instant`` bypasses ordinary planning and makes exactly one configured
  writer-worker invocation (no tools, no verifier, no replan).
- Normal modes include a real verification and correction loop.
- Worker retries and full replans are separate budgets.
- ``unlimited`` means compatibility with the current high-effort behavior —
  **not** literal unbounded execution. Every preset keeps finite hard safety
  ceilings so a pathological plan can always terminate; cancellation still
  propagates.
- Static per-agent configuration (``agents/agent_config.json`` →
  ``execution.max_attempts``) remains an upper limit: the effective attempt
  count is ``min(agent_configured_max_attempts, policy.max_worker_attempts)``.

The resolver is deliberately independent of FastAPI, LangGraph and
``config_loader`` so the direct CLI, the API server and hermetic tests all
share exactly the same behavior. Policies travel as plain serializable dicts
(``as_dict``) inside ``RunnableConfig["configurable"]`` and graph state, which
is what makes them checkpointable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any

# Canonical preset names, cheapest first. Transport input is normalized
# case-insensitively at the boundary; internally only these names exist.
EFFORT_PRESETS: tuple[str, ...] = (
    "instant",
    "light",
    "standard",
    "thorough",
    "unlimited",
)

DEFAULT_EFFORT: str = "unlimited"
"""Resolved when a request omits the effort field (backwards compatible)."""

# Effort levels whose contract includes verification and correction. Used by
# the compatibility-topology rule: parallel/sequential have no verifier, so
# they may only serve ``unlimited`` (legacy) or ``instant`` (routed through
# yotta) runs.
VERIFICATION_EFFORTS: tuple[str, ...] = ("light", "standard", "thorough")

# ---------------------------------------------------------------------------
# Hard safety ceilings for ``unlimited``
# ---------------------------------------------------------------------------
# These bound availability without altering ordinary current runs. They are
# deliberately separate from the preset table below so no preset can ever
# claim an unbounded value.
#
# ``_UNLIMITED_WORKER_ATTEMPTS`` mirrors the validated 1-10 bound of
# ``config_loader._MAX_ATTEMPTS_BOUND``; keep the two in sync if that bound
# ever changes. ``_UNLIMITED_REPLANS`` = 2 keeps the historical yotta
# behavior exactly: the planner runs at most three times (initial plan plus
# two replan passes) before the writer finalizes
# (``agents.orchestrator_node._MAX_REPLANS`` gated the same loop).

_UNLIMITED_PLAN_STEPS = 64
_UNLIMITED_WORKER_ATTEMPTS = 10
_UNLIMITED_TOOL_CALLS_PER_ATTEMPT = 100
_UNLIMITED_VERIFICATION_ATTEMPTS = 10
_UNLIMITED_STEP_VERIFICATION_RETRIES = 2
_UNLIMITED_REPLANS = 2
_UNLIMITED_GRAPH_DISPATCHES = 256
_UNLIMITED_REACT_RECURSION_LIMIT = 100
_UNLIMITED_TIMEOUT_SECONDS = 3600

# ---------------------------------------------------------------------------
# Preset budget table
# ---------------------------------------------------------------------------
# Explicit, documented, test-pinned defaults; deliberately conservative and
# tunable later from observed ``StepStats``. Every field is a *per-run*
# budget:
#
# plan_enabled                    orchestrator may plan (instant skips it)
# instant_writer_only             guaranteed single writer-worker route
# max_plan_steps                  steps allowed in a validated plan
# max_worker_attempts             total worker executions per step (incl. first)
# max_tool_calls_per_attempt      in-flight guard, enforced during ReAct
# max_verification_attempts       attempts of the verifier's single LLM call
# max_step_verification_retries   re-dispatch a deficient step with feedback
# max_replans                     return to the orchestrator with feedback
# max_graph_dispatches            scheduler waves — terminates pathological
#                                 re-entry even when every other budget allows it
# react_recursion_limit           bounds the ReAct model/tool loop per attempt
# timeout_seconds                 wall-clock budget; the entry stamps a deadline

_PRESET_BUDGETS: dict[str, dict[str, Any]] = {
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
        # Compatibility with the current high-effort behavior, subject to the
        # hard safety ceilings above — not literal unbounded execution.
        "plan_enabled": True,
        "instant_writer_only": False,
        "max_plan_steps": _UNLIMITED_PLAN_STEPS,
        "max_worker_attempts": _UNLIMITED_WORKER_ATTEMPTS,
        "max_tool_calls_per_attempt": _UNLIMITED_TOOL_CALLS_PER_ATTEMPT,
        "max_verification_attempts": _UNLIMITED_VERIFICATION_ATTEMPTS,
        "max_step_verification_retries": _UNLIMITED_STEP_VERIFICATION_RETRIES,
        "max_replans": _UNLIMITED_REPLANS,
        "max_graph_dispatches": _UNLIMITED_GRAPH_DISPATCHES,
        "react_recursion_limit": _UNLIMITED_REACT_RECURSION_LIMIT,
        "timeout_seconds": _UNLIMITED_TIMEOUT_SECONDS,
    },
}

PRESET_BUDGETS: dict[str, dict[str, Any]] = {
    name: dict(budgets) for name, budgets in _PRESET_BUDGETS.items()
}
"""Public read-only copy of the preset table (documentation and tests)."""

_BUDGET_FIELDS: tuple[str, ...] = (
    "plan_enabled",
    "instant_writer_only",
    "max_plan_steps",
    "max_worker_attempts",
    "max_tool_calls_per_attempt",
    "max_verification_attempts",
    "max_step_verification_retries",
    "max_replans",
    "max_graph_dispatches",
    "react_recursion_limit",
    "timeout_seconds",
)


@dataclass(frozen=True)
class ExecutionPolicy:
    """One resolved, immutable effort policy.

    Frozen so a policy shared across concurrent graph branches can never be
    mutated mid-run; serialization goes through ``as_dict``/``from_dict``
    (plain JSON-safe values only — checkpointable, transportable).
    """

    preset: str
    plan_enabled: bool
    instant_writer_only: bool
    max_plan_steps: int
    max_worker_attempts: int
    max_tool_calls_per_attempt: int
    max_verification_attempts: int
    max_step_verification_retries: int
    max_replans: int
    max_graph_dispatches: int
    react_recursion_limit: int
    timeout_seconds: int
    # Absolute wall-clock deadline (epoch seconds) stamped once per run.
    # ``None`` until stamped — the graph entry re-stamps if a caller resolved
    # the policy without one (e.g. hand-built test state).
    deadline: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The policy as a plain serializable mapping for ``configurable``."""
        data = {field: getattr(self, field) for field in _BUDGET_FIELDS}
        data["preset"] = self.preset
        data["deadline"] = self.deadline
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionPolicy":
        """Rebuild a policy from a serialized mapping (checkpoints, state).

        Raises ``ValueError`` for a missing or mistyped budget field so a
        corrupted policy fails loudly at the boundary instead of silently
        running with a wrong cap. Unknown keys are ignored — policies from a
        future version must not break old checkpoints.
        """
        for field in _BUDGET_FIELDS:
            if field not in data:
                raise ValueError(
                    f"Execution policy is missing required field '{field}': {data!r}"
                )
            value = data[field]
            expected = bool if field in ("plan_enabled", "instant_writer_only") else int
            if type(value) is not expected:
                raise ValueError(
                    f"Execution policy field '{field}' must be "
                    f"{expected.__name__}, got {type(value).__name__} ({value!r})"
                )
        kwargs = {field: data[field] for field in _BUDGET_FIELDS}
        kwargs["preset"] = data.get("preset", DEFAULT_EFFORT)
        deadline = data.get("deadline")
        if deadline is not None and type(deadline) not in (int, float):
            raise ValueError(
                f"Execution policy 'deadline' must be a number or None, "
                f"got {type(deadline).__name__} ({deadline!r})"
            )
        kwargs["deadline"] = deadline
        return cls(**kwargs)


def normalize_effort(effort: str | None) -> str:
    """Canonicalize transport input: ``"Instant"``, ``" instant "`` -> ``"instant"``.

    ``None``/``""`` resolve to ``DEFAULT_EFFORT`` (``unlimited``) — an omitted
    effort is backwards compatible with the current high-effort behavior.
    Invalid names raise ``ValueError`` naming the valid presets, which the API
    surfaces as its normal 422 validation.
    """
    if effort is None:
        return DEFAULT_EFFORT
    if not isinstance(effort, str):
        raise ValueError(
            f"effort must be a string or None naming a valid preset "
            f"({', '.join(EFFORT_PRESETS)}), got {type(effort).__name__}"
        )
    normalized = effort.strip().casefold()
    if not normalized:
        return DEFAULT_EFFORT
    if normalized not in _PRESET_BUDGETS:
        raise ValueError(
            f"Unknown effort preset '{effort}'. Valid presets: "
            f"{', '.join(EFFORT_PRESETS)}."
        )
    return normalized


def resolve_execution_policy(
    effort: str | None = None, *, now: float | None = None
) -> ExecutionPolicy:
    """Resolve a preset name into an immutable, deadline-stamped policy.

    The one resolver for defaults and numeric budgets: API, CLI and tests all
    go through here, so no call site invents its own caps. ``now`` exists only
    for tests; production stamps ``time.time() + timeout_seconds``.
    """
    preset = normalize_effort(effort)
    budgets = _PRESET_BUDGETS[preset]
    return ExecutionPolicy(
        preset=preset,
        **budgets,
        deadline=(now if now is not None else time.time()) + budgets["timeout_seconds"],
    )


def stamp_deadline(policy: ExecutionPolicy, now: float | None = None) -> ExecutionPolicy:
    """Return a copy of ``policy`` with a fresh absolute deadline.

    Used by the graph entry so hand-built state without a deadline still gets
    wall-clock protection for the actual run.
    """
    base = now if now is not None else time.time()
    return replace(policy, deadline=base + policy.timeout_seconds)


def deadline_exceeded(policy: ExecutionPolicy, now: float | None = None) -> bool:
    """True when the policy's wall-clock budget has been spent."""
    if policy.deadline is None:
        return False
    return (now if now is not None else time.time()) > policy.deadline


def effective_worker_attempts(
    policy: ExecutionPolicy | dict[str, Any] | None, agent_configured_attempts: int
) -> int:
    """The effective total-execution count for one worker/writer step.

    Static agent configuration (``execution.max_attempts``, validated 1-10 by
    ``config_loader``) remains the upper limit and the policy another one:
    ``min(agent_configured_max_attempts, policy.max_worker_attempts)``. A
    missing policy (parallel/sequential legacy state, standalone nodes) means
    the configured count passes through unchanged.
    """
    cap = _coerce(policy).max_worker_attempts if policy is not None else None
    if cap is None:
        return agent_configured_attempts
    return min(agent_configured_attempts, cap)


def effective_verification_attempts(
    policy: ExecutionPolicy | dict[str, Any] | None, agent_configured_attempts: int
) -> int:
    """The effective attempt count for the verifier's single LLM call.

    Same min() contract as :func:`effective_worker_attempts` but against the
    verifier's own budget field (``max_verification_attempts``) — the worker,
    verifier and replan counters stay separate budgets. For ``unlimited`` the
    policy ceiling equals the config bound, so configured values pass through
    and current behavior is preserved exactly.
    """
    cap = _coerce(policy).max_verification_attempts if policy is not None else None
    if cap is None:
        return agent_configured_attempts
    return min(agent_configured_attempts, cap)


def _coerce(policy: ExecutionPolicy | dict[str, Any] | None) -> ExecutionPolicy | None:
    """Accept either representation (dataclass or serialized dict)."""
    if policy is None or isinstance(policy, ExecutionPolicy):
        return policy
    return ExecutionPolicy.from_dict(policy)


def policy_from_config(config: dict[str, Any] | None) -> ExecutionPolicy:
    """Extract the resolved policy from ``RunnableConfig``, or the default.

    Reads ``config["configurable"]["execution_policy"]`` (a serialized dict
    written by the API/CLI boundary). No policy present — plain old clients,
    or tests that never set one — resolves to ``unlimited`` so current
    behavior is preserved exactly.
    """
    if not config:
        return resolve_execution_policy(DEFAULT_EFFORT)
    configurable = config.get("configurable") or {}
    raw = configurable.get("execution_policy")
    if raw is None:
        return resolve_execution_policy(configurable.get("effort") or DEFAULT_EFFORT)
    return _coerce(raw) or resolve_execution_policy(DEFAULT_EFFORT)


def policy_from_state(state: dict[str, Any] | None) -> ExecutionPolicy:
    """Extract the policy from graph state (the checkpointed copy).

    Yotta's entry node writes ``execution_policy`` into state before any
    budgeted node runs; nodes that can run without it (parallel/sequential
    workers, standalone tests) fall back to ``unlimited``.
    """
    if not state:
        return resolve_execution_policy(DEFAULT_EFFORT)
    raw = state.get("execution_policy")
    if raw is None:
        return resolve_execution_policy(state.get("effort") or DEFAULT_EFFORT)
    return _coerce(raw) or resolve_execution_policy(DEFAULT_EFFORT)


# ---------------------------------------------------------------------------
# Compatibility topology rule
# ---------------------------------------------------------------------------

# Graphs that assemble without verification. They remain selectable legacy
# topologies, but they must never silently claim the verification guarantees
# that light/standard/thorough promise.
_LEGACY_NON_VERIFYING_GRAPHS = frozenset({"parallel", "sequential"})


def graph_effort_compat_error(
    graph_name: str | None, effort: str | None = None
) -> str | None:
    """Compatibility error for an explicitly selected legacy graph, or ``None``.

    - ``unlimited``: permitted — backwards-compatible legacy mode.
    - ``instant``: permitted — always executed on the yotta graph regardless
      of the requested topology (Instant has exactly one implementation).
    - ``light``/``standard``/``thorough``: rejected — these effort levels
      promise verification, which parallel/sequential do not provide.
    """
    if not graph_name:
        return None
    if graph_name.strip() not in _LEGACY_NON_VERIFYING_GRAPHS:
        return None
    preset = normalize_effort(effort)
    if preset in VERIFICATION_EFFORTS:
        return (
            f"Effort '{preset}' requires verification and correction, which the "
            f"'{graph_name}' graph does not provide. Omit 'graph' to run on the "
            f"effort-aware default graph, or select 'yotta' explicitly. "
            f"'{graph_name}' serves 'unlimited' (legacy) and 'instant' "
            f"(executed on yotta) runs."
        )
    return None
