"""Yotta pipeline: Send fan-out with per-step verification, bounded retries,
replan, and LLM writer synthesis — the effort-aware default topology.

Registered automatically as the graph named ``"yotta"`` (see
``graphs/__init__.py``) — the module name minus its ``_graph`` suffix.

Capabilities (moved from the old root ``yotta_graph.py``):
- parallel fan-out of ready steps via ``Send`` through a scheduler barrier
  (same topology as ``graphs/parallel_pipeline_graph.py``);
- per-step verification with retry / replan routing (``verify_node``);
- LLM writer synthesis, including the direct (empty-plan) route where yotta
  search results were already sufficient;
- **effort slider** (``execution_policy``): an entry router resolves the
  per-run policy from config/state and stamps it; every cap below (plan
  steps, worker/verifier attempts, step retries, replans, dispatch waves,
  wall-clock deadline) is policy-derived. ``instant`` is a graph route that
  guarantees exactly one writer invocation with zero tools, zero verifier
  and zero replan passes;
- ``citatitaion_node`` — Phase 4 placeholder, defined but not wired.
"""

import asyncio
import json
import re
import time
from functools import partial

from langgraph.graph import StateGraph, END
from langgraph.types import Send, RetryPolicy
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig, RunnableLambda

from agents.agent_states import YottaState, RESULTS_RESET
from agents.orchestrator_node import make_orchestrator_agent
from agents.sub_agents_nodes import (
    _bind_config,
    _resolve_run_step,
    run_step_with_attempts,
    run_sub_agent_async,  # module-global name — tests monkeypatch this
)
from config_loader import get_max_attempts
from execution_policy import (
    deadline_exceeded,
    effective_verification_attempts,
    effective_worker_attempts,
    policy_from_config,
    policy_from_state,
    stamp_deadline,
)
from graphs.parallel_pipeline_graph import scheduler_node
from pipeline_entry import render_files_block
from utils.json_utils import extract_json
from utils.logger import log_event

GRAPH_DESCRIPTION = (
    "Parallel fan-out with per-step verification, bounded retries, "
    "replan, and LLM writer synthesis (effort-aware default)"
)


# ---------------------------------------------------------------------------
# Effort policy — entry routing and dispatch accounting
# ---------------------------------------------------------------------------

def effort_router(state: dict, config: RunnableConfig = None) -> dict:
    """Graph entry: resolve and stamp the per-run effort policy into state.

    The API/CLI boundary already resolved the policy into
    ``config["configurable"]``; when present it wins (it is the transport
    authority). Otherwise the policy comes from hand-seeded state, or the
    ``unlimited`` default — exactly what plain old callers got before the
    slider existed. A policy without a deadline (hand-built test state) is
    stamped here so every real run has wall-clock protection.

    For ``instant`` the graph routes straight to the writer node — planning,
    verification, retries and replans are all skipped by routing, and the
    writer's direct-route branch renders the task and any attached documents
    itself.
    """
    configurable = (config or {}).get("configurable") or {}
    if configurable.get("execution_policy") is not None or configurable.get("effort") is not None:
        policy = policy_from_config(config)
    else:
        policy = policy_from_state(state)
        log_event("execution_policy_resolved", effort=policy.preset,
                  execution_policy=policy.as_dict())
    if policy.deadline is None:
        policy = stamp_deadline(policy)

    updates: dict = {
        "effort": policy.preset,
        "execution_policy": policy.as_dict(),
        "dispatch_count": state.get("dispatch_count", 0),
    }
    if policy.instant_writer_only:
        log_event("instant_route_selected", effort=policy.preset)
        # Instant skips the orchestrator, workers and verifier: the writer
        # node runs the single LLM call itself, and its empty-plan branch
        # renders any attached documents from the raw ``files`` channel
        # (the same handling the direct route gets). Normalize never-written
        # channels and zero the counters the skipped nodes own.
        updates["results"] = state.get("results") or {}
        updates["plan"] = []
        updates["replan_count"] = state.get("replan_count", 0)
        updates["verification_attempts"] = state.get("verification_attempts", 0)
    return updates


def route_from_entry(state: dict):
    """After the entry router: instant routes straight to the writer node —
    the same "writer" target ``fan_out_router`` uses for its direct route;
    everything else plans normally."""
    policy = policy_from_state(state)
    if policy.instant_writer_only:
        return "writer"
    return "orchestrator"


def _wave_or_route(state: dict) -> str:
    """fan_out_router's decision as a node name: a ``Send`` list means a
    worker dispatch wave ("sub_agent"). Pure — safe to call twice."""
    route = fan_out_router(state)
    return "sub_agent" if isinstance(route, list) else route


def _yotta_scheduler(state: dict) -> dict:
    """Scheduler barrier plus effort dispatch/deadline accounting.

    Runs the shared skip-marker barrier, then decides — on the projected
    state the router will actually see — whether the next move is a worker
    dispatch wave. Only dispatch waves consume the ``max_graph_dispatches``
    and wall-clock deadline budgets, so a pathological replan/retry cycle
    always terminates with a structured ``safety_stop_reason`` (never an
    unbounded loop) while a legitimate run's verification passes never
    consume wave budget.
    """
    updates = scheduler_node(state)

    projected_results = dict(state.get("results", {}))
    projected_results.update(updates.get("results") or {})
    projected = {**state, **updates, "results": projected_results}

    dispatch_count = state.get("dispatch_count", 0)
    safety = state.get("safety_stop_reason", "") or ""

    if _wave_or_route(projected) == "sub_agent":
        dispatch_count += 1
        policy = policy_from_state(state)
        if dispatch_count > policy.max_graph_dispatches:
            safety = (
                f"max_graph_dispatches={policy.max_graph_dispatches} exceeded"
            )
        elif deadline_exceeded(policy):
            safety = "wall-clock deadline exceeded"
        if safety:
            log_event("effort_safety_stop", reason=safety,
                      effort=policy.preset, dispatch_count=dispatch_count)

    out = dict(updates)
    out["dispatch_count"] = dispatch_count
    if safety:
        out["safety_stop_reason"] = safety
    return out


def _latest_retry_note(step_verifications: dict[int, dict], step_num: int) -> str:
    """Extract the most recent ``[Retry k/N] ...`` note for a step (N is the
    policy-derived retry cap), so a retried worker sees only the latest
    actionable feedback, not the whole accumulated history (fixes F2).
    """
    entry = step_verifications.get(step_num)
    if not entry:
        return ""
    notes = entry.get("notes", "")
    if not isinstance(notes, str) or not notes:
        return ""
    retry_lines = [line for line in notes.split("\n") if line.startswith("[Retry ")]
    if not retry_lines:
        return ""
    return retry_lines[-1]


def fan_out_router(state: dict):
    """After orchestration, dispatch ALL ready steps in parallel via Send.

    Mirrors ``graphs.parallel_pipeline_graph.fan_out_router`` plus the yotta
    routes:

    - an effort **safety stop** (dispatch ceiling or deadline exceeded)
      → ``"writer"`` — finalize the best result available instead of
      dispatching another wave;
    - empty plan → ``"writer"`` (direct route — yotta search results were
      already sufficient, no sub-agents needed; distinct from Instant, which
      never reaches this router);
    - ``ready`` = steps not in results with satisfied deps **plus**
      ``pending_retries`` — a retried step is dispatched even though its old
      result is still in ``results`` (intentional re-execution, overwritten
      via the merge);
    - all done → ``"verify"``;
    - deadlock → ``RuntimeError`` naming every blocked step (same detail
      message as the parallel graph).
    """
    if state.get("safety_stop_reason"):
        return "writer"

    plan = state["plan"]
    results = state.get("results", {})
    current_datetime = state.get("current_datetime", "")
    pending_retries = state.get("pending_retries", [])

    if not plan:
        return "writer"

    # Find all steps whose dependencies are satisfied, plus retried steps
    # (already in results, but intentionally re-executed).
    ready = [
        s for s in plan
        if s["step"] in pending_retries
        or (
            s["step"] not in results
            and all(d in results for d in s.get("depends_on", []))
        )
    ]

    if ready:
        # Send each ready step to the sub_agent node in parallel. Base payload
        # shape matches the parallel graph; step_verifications + streaming are
        # carried as well so the production worker can bind verifier feedback
        # (F2) — feedback is NOT the payload itself (see the worker factory).
        # ``files`` (the run's attached documents) rides along too: a Send
        # payload is the ONLY state a worker sees, so without it a step the
        # orchestrator assigned documents to would receive filenames but no
        # text. The effort policy rides along for the same reason — budget
        # enforcement must mean the same thing in every worker wave, retry
        # pass and replan execution.
        return [
            Send("sub_agent", {
                "step": s,
                "results": results,
                "current_datetime": current_datetime,
                "step_verifications": state.get("step_verifications", {}),
                "streaming": state.get("streaming", False),
                "files": state.get("files", {}),
                "effort": state.get("effort", ""),
                "execution_policy": state.get("execution_policy", {}),
            })
            for s in ready
        ]

    # No ready steps — distinguish "all done" from "permanently blocked"
    unfinished = [s["step"] for s in plan if s["step"] not in results]
    if unfinished:
        detail_parts = []
        for s in plan:
            if s["step"] in unfinished:
                unmet = [d for d in s.get("depends_on", []) if d not in results]
                detail_parts.append(
                    f"step {s['step']} (unmet dependencies: {unmet})"
                )
        raise RuntimeError(
            f"No step is ready to execute, but {len(unfinished)} step(s) "
            f"remain unfinished and are permanently blocked: {unfinished}. "
            f"Details: {'; '.join(detail_parts)}. "
            f"Check that every step's depends_on references valid step numbers."
        )

    return "verify"


def make_yotta_sub_agent_node(run_step=None, llm=None):
    """Factory for the yotta worker node (one ``Send`` task per ready step).

    Mirrors ``make_parallel_sub_agent_node`` — dual-mode ``RunnableLambda``
    (sync body = ``asyncio.run(async body)``), ``_resolve_run_step`` injection,
    the bounded ``run_step_with_attempts`` loop
    (``graph_name="yotta"``), identical
    stats/containment returns (``results`` / ``step_stats`` / ``failed_steps``)
    — with two differences:

    - **no "already done" guard** — a retried step is dispatched even though
      its old result is still in ``results`` (intentional re-execution);
    - the **production runner** reads ``step_verifications`` from its input to
      bind ``feedback=`` (F2 — the latest ``[Retry k/N]`` note) plus
      ``streaming=``, ``files=`` (the run's attached documents, keyed by
      filename — the runner injects the ones this step was assigned) and the
      run's effort ``policy`` via ``partial``; injected 3-arg test stubs keep
      their signature and never receive any of those keywords.

    The attempt cap is policy-aware: the effective count is
    ``min(agent-configured max_attempts, policy.max_worker_attempts)`` — the
    Send payload carries the policy, and a payload without one (legacy
    callers, older tests) resolves to ``unlimited``, preserving today's
    per-agent behavior exactly.
    """
    run_step, takes_config = _resolve_run_step(run_step, llm)

    async def yotta_sub_agent_node_async(state: dict, config: RunnableConfig = None) -> dict:
        step = state["step"]
        t0 = time.monotonic()
        try:
            step_runner = _bind_config(run_step, takes_config, config)
            if takes_config:
                # production runner — bind verifier feedback + streaming +
                # the run's effort policy (tool budget, recursion limit)
                feedback = _latest_retry_note(
                    state.get("step_verifications", {}), step["step"]
                )
                run = partial(
                    step_runner,
                    step,
                    state["results"],
                    state.get("current_datetime", ""),
                    streaming=state.get("streaming", False),
                    files=state.get("files", {}),
                    feedback=feedback,
                    policy=state.get("execution_policy"),
                )
            else:
                run = partial(
                    step_runner,
                    step,
                    state["results"],
                    state.get("current_datetime", ""),
                )
            step_num, output, inner_stats = await run_step_with_attempts(
                step,
                run,
                effective_worker_attempts(
                    state.get("execution_policy"), get_max_attempts(step["agent"])
                ),
                graph_name="yotta",
            )
            stats = {
                "step": step_num,
                "agent": step["agent"],
                "status": "completed",
                "duration_s": round(time.monotonic() - t0, 3),
                "input_tokens": inner_stats["input_tokens"],
                "output_tokens": inner_stats["output_tokens"],
                "tool_calls": inner_stats["tool_calls"],
            }
            # pending_retries is cleared here: the router re-reads it after
            # the scheduler barrier, so without this clear a retried step
            # (whose old result is still in results) would be re-dispatched
            # forever. All workers of one pass write the same value — the
            # last write wins.
            return {"results": {step_num: output}, "step_stats": [stats],
                    "pending_retries": []}
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Cancellation and process-level signals are never contained as
            # step failures — they escape to the caller (Slice 3).
            raise
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            duration = round(time.monotonic() - t0, 3)
            log_event("sub_agent_step_failed", step=step["step"], error=str(e))
            return {
                "results": {step["step"]: f"[STEP FAILED] {e}"},
                "failed_steps": [step["step"]],
                "step_stats": [{
                    "step": step["step"],
                    "agent": step["agent"],
                    "status": "failed",
                    "duration_s": duration,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "tool_calls": 0,
                }],
                "pending_retries": [],
            }

    def yotta_sub_agent_node(state: dict, config: RunnableConfig = None) -> dict:
        """Sync entry point — LangGraph calls this under ``graph.invoke``."""
        return asyncio.run(yotta_sub_agent_node_async(state, config))

    return RunnableLambda(yotta_sub_agent_node, afunc=yotta_sub_agent_node_async)


_MIN_SUBSTRING_MATCH_LEN = 15


def _match_subquery_to_step(subquery: str, plan: list[dict]) -> int | None:
    """Map a verifier's legacy ``subquery`` reference back to a plan step
    number. Only used as a fallback when the verifier didn't send a numeric
    ``step`` field (see S1 — the parser in ``verify_node`` prefers that).

    Tries to extract "Step N" from the subquery string first, then a bare
    integer id, then falls back to substring matching against plan step
    subtasks. Returns ``None`` when no mapping can be determined.
    """
    if not subquery:
        return None

    # ---- "Step N" pattern ------------------------------------------------
    m = re.match(r'^[Ss]tep\s+(\d+)', subquery)
    if m:
        step_num = int(m.group(1))
        if any(s["step"] == step_num for s in plan):
            return step_num

    # ---- bare integer id, e.g. "2" ----------------------------------------
    m = re.match(r'^\s*(\d+)\s*$', subquery)
    if m:
        step_num = int(m.group(1))
        if any(s["step"] == step_num for s in plan):
            return step_num
        return None

    # ---- substring match against subtask text ------------------------------
    # Guard against short strings ("2", "AI") fuzzy-matching an unrelated
    # subtask that merely contains that character sequence.
    if len(subquery) < _MIN_SUBSTRING_MATCH_LEN:
        return None
    for s in plan:
        subtask = s.get("subtask", "")
        if subtask and len(subtask) >= _MIN_SUBSTRING_MATCH_LEN and (subtask in subquery or subquery in subtask):
            return s["step"]

    return None


def _build_verifier_context(step_verifications: dict[int, dict],
                            max_step_verification_retries: int) -> str:
    """Build a context block showing previous verification results per step.

    The verifier uses this to respect the retry cap (policy-derived) and to
    avoid repeating notes it already raised in a prior pass.
    """
    if not step_verifications:
        return ""
    lines = ["## Previous verification history"]
    for sid in sorted(step_verifications.keys()):
        v = step_verifications[sid]
        verdict = v.get("verdict", "?")
        retries = v.get("retries", 0)
        notes = v.get("notes", "")
        notes_short = (notes[:120] + "..") if len(notes) > 120 else notes
        lines.append(
            f"- Step {sid}: {verdict}"
            + (f" (retries used: {retries}/{max_step_verification_retries})" if retries > 0 else "")
            + f" — {notes_short}"
        )
    return "\n".join(lines)


async def verify_node(state: dict) -> dict:
    """Run the verifier agent against all sub-agent results, parse the verdict,
    and persist it so the conditional edge can route accordingly.

    The verifier's raw output is stored in ``verifier_report`` (a dedicated
    state field) — NOT in ``results``, so it doesn't leak into the final
    document via the writer as if it were a research finding.

    Routing (fixes F4/F5/F6): a single ``verification_route`` decision;
    on ``"retry"`` the failed steps are re-dispatched via ``pending_retries``
    (their old result stays in ``results`` — overwritten on re-execution);
    on ``"replan"`` results are wiped via the ``RESULTS_RESET`` sentinel so
    the orchestrator's new plan replaces the research wholesale.

    Effort policy: the verifier's LLM call attempts, the per-step retry cap
    and the replan cap all come from the policy. When the replan budget is
    spent, ``verification_exhausted`` is set deterministically (no LLM
    involved) so the writer produces an explicitly partial result instead of
    looping or raising a transport error for an otherwise usable result.
    """
    plan = state.get("plan", [])
    results = state.get("results", {})
    replan_count = state.get("replan_count", 0)
    step_verifications = state.get("step_verifications", {})
    policy = policy_from_state(state)

    # Build a synthetic step that describes the verification task
    plan_summary = "\n".join(
        f"Step {s['step']} ({s['agent']}): {s['subtask']}"
        for s in plan
    )

    log_event("verification_started", plan_steps=[s["step"] for s in plan],
              result_steps=sorted(k for k in results.keys() if isinstance(k, int)),
              verification_attempts=state.get("verification_attempts", 0) + 1,
              effort=policy.preset)

    results_summary = "\n\n".join(
        f"--- Step {step_num} output ---\n{output}"
        for step_num, output in sorted(results.items())
        if isinstance(step_num, int) and step_num >= 1
    )

    verify_step = {
        "step": "verify",
        "agent": "verifier",
        "subtask": (
            "Verify the accuracy, completeness, and consistency of the sub-agent results below.\n\n"
            "In your output, reference each result by its numeric `step` "
            "(the N in `--- Step N output ---`) — not by a paraphrase of the subquery.\n\n"
            f"## Original plan\n{plan_summary}\n\n"
            + _build_verifier_context(
                step_verifications, policy.max_step_verification_retries
            )
            + f"\n\n## Sub-agent results\n{results_summary}\n\n"
        ),
        "skills_needed": ["information-verifier"],
        "depends_on": [s["step"] for s in plan],
    }

    # Bounded attempt loop (Slice 3): the verifier's single LLM call runs
    # through the same helper as the workers — no graph-level RetryPolicy.
    # The attempt cap is the policy's verification budget intersected with
    # the static config (min()), so ``unlimited`` keeps today's behavior.
    _, output, _ = await run_step_with_attempts(
        verify_step,
        partial(
            run_sub_agent_async,
            verify_step,
            results,
            state.get("current_datetime", ""),
            streaming=state.get("streaming", False),
            policy=state.get("execution_policy"),
        ),
        effective_verification_attempts(policy, get_max_attempts("verifier")),
        graph_name="yotta",
    )

    # Seed from ALL existing entries so retry counts and prior notes survive
    # for steps that are being re-verified in this pass (fixes Bug 2).
    accumulated: dict[int, dict] = {
        sid: dict(entry) for sid, entry in step_verifications.items()
    }
    any_replan = False
    any_retry = False
    steps_to_remove: set[int] = set()
    unmapped_failed_notes: list[str] = []
    aggregate_verdict = "PASSED"
    verification_notes = ""
    route = "proceed"

    try:
        # Parse the verifier's JSON output — expect a list of per-subquery
        # verdicts (the SKILL.md spec) with a single-object fallback.
        parsed = extract_json(output)
        if not isinstance(parsed, list):
            # print(type(parsed))
            # print('-'*50)
            # print(parsed)
            # print('-'*50)

            parsed = [parsed]
        log_event("verify_node_parsed", result_count=len(results),
                  verdict_count=len(parsed))

        # ---- map each verifier entry to a plan step ---------------------------
        # Prefer the numeric `step` field (S1 contract); fall back to the
        # legacy subquery-text matcher for older-format verifier output.
        plan_step_ids = {s["step"] for s in plan}
        per_step: dict[int, dict] = {}
        for item in parsed:
            raw_step = item.get("step")
            sid: int | None = None
            if isinstance(raw_step, int) and raw_step in plan_step_ids:
                sid = raw_step
            elif isinstance(raw_step, str) and raw_step.strip().isdigit() and int(raw_step.strip()) in plan_step_ids:
                sid = int(raw_step.strip())
            else:
                sid = _match_subquery_to_step(item.get("subquery", ""), plan)

            verdict = item.get("verification_result", "PASSED").upper()
            notes = item.get("notes", "")

            if sid is None:
                log_event("verify_node_unmapped_verdict",
                          raw_step=raw_step,
                          subquery=item.get("subquery", ""))
                if verdict == "FAILED":
                    # Never drop a FAILED verdict silently just because it
                    # couldn't be mapped — surface it as a replan signal so
                    # the orchestrator sees it instead of it vanishing (F1).
                    unmapped_failed_notes.append(notes or "(no notes provided)")
                continue

            per_step[sid] = {"verdict": verdict, "notes": notes}

        # ---- accumulate with existing step_verifications ----------------------
        for sid, v in per_step.items():
            verdict = v["verdict"]
            notes = v["notes"]

            acc = accumulated.get(sid)
            if acc is None:
                acc = {"verdict": "", "notes": "", "retries": 0}
                accumulated[sid] = acc

            # Normalise existing notes to a list for append-only accumulation.
            # Notes may be a newline-joined string (from a previous cycle) or a
            # list (fresh initialisation above).  (fixes Bug 3)
            raw_notes: str = acc.get("notes", "") if isinstance(acc.get("notes"), str) else ""
            acc_notes: list[str] = [raw_notes] if raw_notes else []

            current_retries: int = acc.get("retries", 0)

            # A scheduler-skipped step (dependency failed) gets verdict
            # "SKIPPED" and never triggers retry/replan — retrying it would
            # loop forever since its dependencies can never be satisfied.
            prev_output = results.get(sid, "")
            if isinstance(prev_output, str) and prev_output.startswith("[SKIPPED"):
                acc["verdict"] = "SKIPPED"
                if notes:
                    acc_notes.append(notes)
                acc["notes"] = "\n".join(acc_notes)
                continue

            if verdict == "FAILED":
                if notes.upper().startswith("REPLAN:"):
                    # ---- REPLAN — needs orchestrator --------------------------
                    any_replan = True
                    acc_notes.append(f"[Replan] {notes}")
                else:
                    # ---- RETRY — re-run the worker, bounded by the policy ----
                    if current_retries < policy.max_step_verification_retries:
                        any_retry = True
                        steps_to_remove.add(sid)   # re-dispatched via pending_retries
                        current_retries += 1
                        acc_notes.append(
                            f"[Retry {current_retries}/"
                            f"{policy.max_step_verification_retries}] {notes}"
                        )
                    else:
                        # At retry cap: treat as PASSED WITH NOTES
                        verdict = "PASSED WITH NOTES"
                        acc_notes.append(f"[Retry cap reached] {notes}")
            else:
                # PASSED or PASSED WITH NOTES — record as-is
                if notes:
                    acc_notes.append(notes)

            acc["verdict"] = verdict
            acc["notes"] = "\n".join(acc_notes)
            acc["retries"] = current_retries

        # An unmapped FAILED verdict has no step to attach to, but it must
        # still count toward the aggregate and force a replan (fixes F1).
        if unmapped_failed_notes:
            any_replan = True

        # ---- build aggregate verdict from ACCUMULATED verdicts, not per-pass
        # flags — a step whose accumulated verdict is still FAILED (e.g. left
        # over from an earlier pass) must keep the aggregate FAILED even if
        # this pass didn't touch it (fixes F6).
        if unmapped_failed_notes or any(v["verdict"] == "FAILED" for v in accumulated.values()):
            aggregate_verdict = "FAILED"
        elif any(v["verdict"] == "PASSED WITH NOTES" for v in accumulated.values()):
            aggregate_verdict = "PASSED WITH NOTES"
        else:
            aggregate_verdict = "PASSED"

        # ---- compute the single routing decision (fixes F5) -------------------
        # Mixed pass (replan + retry) dominates as "replan" — the orchestrator
        # wipes state on replan anyway, so retry-clearing steps here would just
        # lose research for nothing (fixes F4's mixed-pass case).
        if any_retry and not any_replan:
            route = "retry"
        elif any_replan:
            route = "replan"
        elif aggregate_verdict == "FAILED":
            # Safety net: an accumulated FAILED entry survived this pass
            # untouched (unmapped verdict, or otherwise not re-reported) —
            # never fall through to the writer with an unresolved failure.
            route = "replan"
        else:
            route = "proceed"

        # ---- build accumulated notes string (for writer + orchestrator) -----
        notes_parts: list[str] = []
        for sid in sorted(accumulated.keys()):
            v = accumulated[sid]
            plan_label = next(
                (f"Step {s['step']} ({s['agent']}): {s['subtask']}"
                 for s in plan if s["step"] == sid),
                f"Step {sid}"
            )
            notes_parts.append(
                f"### {plan_label}  [{v['verdict']}]\n{v['notes']}"
            )
        if unmapped_failed_notes:
            notes_parts.append(
                "### Unmapped verification failures  [FAILED]\n"
                "The verifier reported these FAILED verdicts without a "
                "step id the pipeline could resolve — review and address "
                "them manually:\n"
                + "\n".join(f"- {n}" for n in unmapped_failed_notes)
            )
        verification_notes = "\n\n".join(notes_parts)

    except (ValueError, json.JSONDecodeError):
        # If parsing fails, check for keywords in the raw output.
        # We can't distinguish RETRY from REPLAN in this path, so treat
        # any FAILED as needing a replan (safe default).
        aggregate_verdict = "PASSED"
        upper_output = output.upper()
        if "FAILED" in upper_output:
            aggregate_verdict = "FAILED"
        elif "PASSED WITH NOTES" in upper_output:
            aggregate_verdict = "PASSED WITH NOTES"
        verification_notes = output
        accumulated = state.get("step_verifications", {})
        route = "replan" if aggregate_verdict == "FAILED" else "proceed"

    # Increment replan_count only when we're actually routing to the
    # orchestrator — pure retries have their own per-step cap and don't
    # count against it (fixes F5's un-counted-replan bug).
    new_replan_count = replan_count
    if route == "replan":
        new_replan_count = replan_count + 1
        if new_replan_count <= policy.max_replans:
            log_event("replan_scheduled", replan_count=new_replan_count,
                      max_replans=policy.max_replans, effort=policy.preset)
    if route == "retry":
        log_event("step_retry_scheduled", steps=sorted(steps_to_remove),
                  effort=policy.preset)

    # Deterministic verification exhaustion: when the replan budget is spent,
    # ``after_verify`` routes to the writer — and this flag (no LLM involved)
    # makes that result explicitly partial instead of pretending verification
    # succeeded. Never an infinite loop, never a 500 for a usable result.
    verification_exhausted = bool(state.get("verification_exhausted"))
    if route == "replan" and new_replan_count > policy.max_replans:
        verification_exhausted = True
        log_event("replan_exhausted", replan_count=new_replan_count,
                  max_replans=policy.max_replans, effort=policy.preset)

    log_event("verification_finished", verdict=aggregate_verdict, route=route,
              effort=policy.preset,
              verification_exhausted=verification_exhausted)

    # ---- routing payload --------------------------------------------------
    # retry:  re-dispatch the failed steps via pending_retries (their old
    #         result stays in results — overwritten on re-execution);
    # replan: wipe results via the reset sentinel so the orchestrator's new
    #         plan replaces the research wholesale;
    # else:   nothing pending.
    updates: dict = {"pending_retries": []}
    if route == "retry" and steps_to_remove:
        updates["pending_retries"] = sorted(steps_to_remove)
    elif route == "replan":
        updates["results"] = {RESULTS_RESET: ""}

    return {
        "verification_result": aggregate_verdict,
        "verification_notes": verification_notes,
        "verifier_report": output,
        "replan_count": new_replan_count,
        "step_verifications": accumulated,
        "verification_route": route,
        "verification_attempts": state.get("verification_attempts", 0) + 1,
        "verification_exhausted": verification_exhausted,
        **updates,
    }


def after_verify(state: dict) -> str:
    """Thin reader: route purely on ``verification_route``, computed once in
    ``verify_node``. No note re-parsing here (fixes F5).

    safety stop                             → writer (finalize best result)
    route == "retry"                        → scheduler (never gated by the
                                               replan cap — fixes F4)
    route == "replan" and replan <= cap     → orchestrator
    otherwise                               → writer (proceed, replan cap
                                               exhausted, or safety stop)
    """
    if state.get("safety_stop_reason"):
        return "writer"
    route = state.get("verification_route", "proceed")
    replan_count = state.get("replan_count", 0)

    if route == "retry":
        return "scheduler"
    # ``max_replans`` counts actual replan passes: replan_count is already the
    # number of replan requests made, so pass N (1-based) is allowed while
    # N <= max. Unlimited (2) reproduces the historical three-planner-pass
    # ceiling exactly.
    if route == "replan" and replan_count <= policy_from_state(state).max_replans:
        return "orchestrator"
    return "writer"


async def writer_node(state: dict) -> dict:
    """Assemble sub-agent results (or direct yotta findings) plus verifier notes
    into one comprehensive artefact.

    Three routes reach this node:
    1. **Normal** — orchestrator → sub_agents → verify → writer.
       ``plan`` has steps, ``results`` has sub-agent outputs, and
       ``verification_notes`` may be present.
    2. **Direct** — orchestrator returned an empty plan because the yotta
       search results were already sufficient.
       ``plan`` is ``[]``, ``results`` is ``{}``, ``search_results`` has the
       findings. If files were attached, this route sees their raw content
       directly (there are no document-reader steps to have read them).
    3. **Instant** — the effort entry router sends the run here with an empty
       plan (``route_from_entry`` returns ``"writer"``, the same target the
       fan-out router uses for the direct route), so handling is exactly the
       direct route's: original task + any attached files, one writer
       invocation.

    Effort policy: the attempt cap is policy-derived; a safety stop or
    exhausted verification budget is surfaced to the reader as an explicit
    human-readable warning (never raw hidden state, prompts or credentials)
    and the run's status becomes ``partial``.
    """
    results = state.get("results", {})
    verification_notes = state.get("verification_notes", "")
    plan = state.get("plan", [])
    task = state.get("task", "")
    search_results = state.get("search_results", "")
    files = state.get("files", {})
    safety_stop_reason = state.get("safety_stop_reason", "") or ""
    verification_exhausted = bool(state.get("verification_exhausted"))
    policy = policy_from_state(state)

    log_event("writer_node_start", plan_steps=[s["step"] for s in plan],
              result_steps=sorted(k for k in results.keys() if isinstance(k, int)))

    # ---- build the writer's prompt blocks -----------------------------------
    blocks: list[str] = []

    # Original query — strip the "Query: " prefix if present
    clean_query = task
    if clean_query.startswith("Query: "):
        clean_query = clean_query[len("Query: "):]
    blocks.append(f"## Original query\n{clean_query}")

    # Direct route only: no document-reader steps ran, so the writer needs
    # the raw attached documents itself. On the normal route, the writer
    # relies on document-reader-worker findings in ``results`` instead.
    if not plan and files:
        files_block = render_files_block(files)
        if files_block:
            blocks.append(files_block)

    # Original plan (normal route only)
    if plan:
        plan_summary = "\n".join(
            f"Step {s['step']} ({s['agent']}): {s['subtask']}"
            for s in plan
        )
        blocks.append(f"## Original plan\n{plan_summary}")

    # Sub-agent results — only real sub-agent outputs (step >= 1).
    # The verifier's output is NOT in results (it lives in verifier_report).
    if plan:
        results_summary = "\n\n".join(
            f"--- Step {step_num} output ---\n{output}"
            for step_num, output in sorted(results.items())
            if isinstance(step_num, int) and step_num >= 1
        )
        if results_summary:
            blocks.append(f"## Sub-agent results\n{results_summary}")

    # Search / grounding results — read from the dedicated state field
    # instead of parsing the task string.
    if search_results:
        blocks.append(f"## Search results (initial grounding)\n{search_results}")

    # Verifier notes (normal route)
    if verification_notes:
        blocks.append(
            f"## Verifier notes (incorporate these)\n{verification_notes}"
        )

    # Effort warnings: the writer must tell the reader when the result is
    # partial *because of a budget decision*, in plain words — never raw
    # hidden state.
    partial_reasons: list[str] = []
    if safety_stop_reason:
        partial_reasons.append(
            f"Execution stopped by the selected effort budget: "
            f"{safety_stop_reason}. The document below is the best result "
            f"available within that budget and is partial."
        )
    if verification_exhausted:
        partial_reasons.append(
            "The verification budget was exhausted before every finding "
            "could be fully confirmed. The document below incorporates all "
            "available results and verification notes, and is partial."
        )
    if partial_reasons:
        blocks.append("## Partial result warning\n" + "\n\n".join(partial_reasons))

    subtask = (
        "Combine the following information into one comprehensive, "
        "well-structured artefact. Resolve any contradictions and synthesise "
        "the information into a cohesive final document.\n\n"
        + "\n\n".join(blocks)
    )

    print('&'*50)
    print(subtask)
    print('&'*50)

    write_step = {
        "step": "assemble",
        "agent": "writer",
        "subtask": subtask,
        "skills_needed": ["answer-writer"],
        "depends_on": list(results.keys()),
    }

    # Bounded attempt loop (Slice 3), same as the workers and verifier —
    # policy-derived effective cap.
    _, output, _ = await run_step_with_attempts(
        write_step,
        partial(
            run_sub_agent_async,
            write_step,
            results,
            state.get("current_datetime", ""),
            streaming=state.get("streaming", False),
            policy=state.get("execution_policy"),
        ),
        effective_worker_attempts(policy, get_max_attempts("writer")),
        graph_name="yotta",
    )

    # The assembled document is the terminal product — results must not be
    # repopulated with it (the old ``{"assemble": ...}`` entry polluted the
    # results channel for no reader).
    updates: dict = {"final_output": output}
    if partial_reasons:
        updates["status"] = "partial"
    return updates


# ---------------------------------------------------------------------------
# Citation node (Phase 4 — citation pipeline)
# NOTE: This node is defined but not wired into the graph yet.  It will be
# properly integrated in Phase 4 once the source_map threading and
# structured worker outputs are in place. When it is wired, give it a branch
# in ``streaming._ProtocolTranslator`` so its result reaches the client too.
# ---------------------------------------------------------------------------

async def citatitaion_node(state: dict) -> dict:
    """Citation & QA gate: runs after the writer, checks the draft against the
    source map and verified findings, then produces the final answer."""
    log_event("citation_node_start", state_keys=sorted(state.keys()))
    results = state.get("results", {})
    step = state.get("step", {})
    draft = step.get("draft", "")
    source_map = step.get("source_map", [])

    source_map_str = (
        json.dumps(source_map, indent=2)
        if source_map
        else "No source map available"
    )

    cite_step = {
        "step": "citation",
        "agent": "writer",
        "subtask": (
            f"## Draft\n{draft}\n\n"
            f"## Source Map\n{source_map_str}\n\n"
            f"## Verified Findings\n"
            + "\n\n".join(
                f"--- Step {n} ---\n{out}"
                for n, out in results.items()
            )
        ),
        "skills_needed": ["citation-qa-agent"],
        "depends_on": list(results.keys()),
    }

    _, output, _ = await run_sub_agent_async(
        cite_step,
        results,
        state.get("current_datetime", ""),
        streaming=state.get("streaming", False),
        policy=state.get("execution_policy"),
    )

    try:
        parsed = extract_json(output)
        final_answer = parsed.get("final_answer", output)
    except (ValueError, json.JSONDecodeError):
        final_answer = output

    return {"final_output": final_answer}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
    """Compile the yotta pipeline.

    Every argument defaults to the production wiring, so ``build()`` takes no
    arguments in production; tests inject fake nodes and their own checkpointer.
    """
    orchestrator = orchestrator or make_orchestrator_agent()
    sub_agent = sub_agent or make_yotta_sub_agent_node()

    builder = StateGraph(YottaState)
    # ValueError from plan validation (1.2) should re-plan, not kill the run —
    # same as the parallel graph. No worker/verify RetryPolicy (Slice 3): the
    # worker owns its bounded attempt loop; verify/writer wrap their single
    # LLM call in ``run_step_with_attempts`` too.
    builder.add_node("orchestrator", orchestrator,
                     retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
    builder.add_node("sub_agent", sub_agent)
    builder.add_node("scheduler", _yotta_scheduler)
    builder.add_node("verify", verify_node)
    builder.add_node("writer", writer_node)
    builder.add_node("effort_router", effort_router)

    # Entry: resolve the effort policy first; instant goes straight to the
    # writer (its direct-route branch), everything else plans normally.
    builder.set_entry_point("effort_router")
    builder.add_conditional_edges("effort_router", route_from_entry,
                                  ["orchestrator", "writer"])
    # Workers re-enter the scheduler barrier (no-op skip-marker pass).
    builder.add_edge("sub_agent", "scheduler")
    builder.add_edge("orchestrator", "scheduler")
    builder.add_conditional_edges("scheduler", fan_out_router,
                                  ["writer", "verify", "sub_agent"])
    builder.add_conditional_edges("verify", after_verify,
                                  ["scheduler", "orchestrator", "writer"])
    builder.add_edge("writer", END)

    # Compiled with MemorySaver for interactive use (run_pipeline.py,
    # streaming.py); the API server compiles without a checkpointer.
    return builder.compile(checkpointer=checkpointer or MemorySaver())
