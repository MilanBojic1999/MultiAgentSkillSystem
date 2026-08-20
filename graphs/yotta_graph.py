"""Yotta pipeline: Send fan-out with per-step verification, bounded retries,
replan, and LLM writer synthesis.

Registered automatically as the graph named ``"yotta"`` (see
``graphs/__init__.py``) — the module name minus its ``_graph`` suffix.

Capabilities (moved from the old root ``yotta_graph.py``):
- parallel fan-out of ready steps via ``Send`` through a scheduler barrier
  (same topology as ``graphs/parallel_pipeline_graph.py``);
- per-step verification with retry / replan routing (``verify_node``);
- LLM writer synthesis, including the direct (empty-plan) route where yotta
  search results were already sufficient;
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
from agents.orchestrator_node import make_orchestrator_agent, _MAX_REPLANS
from agents.sub_agents_nodes import (
    _bind_config,
    _resolve_run_step,
    run_step_with_attempts,
    run_sub_agent_async,  # module-global name — tests monkeypatch this
)
from config_loader import get_max_attempts
from graphs.parallel_pipeline_graph import scheduler_node
from pipeline_entry import render_files_block
from utils.json_utils import extract_json
from utils.logger import log_event

GRAPH_DESCRIPTION = (
    "Parallel fan-out with per-step verification, bounded retries, "
    "replan, and LLM writer synthesis"
)


def _latest_retry_note(step_verifications: dict[int, dict], step_num: int) -> str:
    """Extract the most recent ``[Retry k/2] ...`` note for a step, so a
    retried worker sees only the latest actionable feedback, not the whole
    accumulated history (fixes F2).
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

    - empty plan → ``"writer"`` (direct route — yotta search results were
      already sufficient, no sub-agents needed);
    - ``ready`` = steps not in results with satisfied deps **plus**
      ``pending_retries`` — a retried step is dispatched even though its old
      result is still in ``results`` (intentional re-execution, overwritten
      via the merge);
    - all done → ``"verify"``;
    - deadlock → ``RuntimeError`` naming every blocked step (same detail
      message as the parallel graph).
    """
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
        # orchestrator assigned documents to would receive filenames but no text.
        return [
            Send("sub_agent", {
                "step": s,
                "results": results,
                "current_datetime": current_datetime,
                "step_verifications": state.get("step_verifications", {}),
                "streaming": state.get("streaming", False),
                "files": state.get("files", {}),
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
    (``get_max_attempts(step["agent"])``, ``graph_name="yotta"``), identical
    stats/containment returns (``results`` / ``step_stats`` / ``failed_steps``)
    — with two differences:

    - **no "already done" guard** — a retried step is dispatched even though
      its old result is still in ``results`` (intentional re-execution);
    - the **production runner** reads ``step_verifications`` from its input to
      bind ``feedback=`` (F2 — the latest ``[Retry k/2]`` note) plus
      ``streaming=`` and ``files=`` (the run's attached documents, keyed by
      filename — the runner injects the ones this step was assigned) via
      ``partial``; injected 3-arg test stubs keep their signature and never
      receive any of those keywords.
    """
    run_step, takes_config = _resolve_run_step(run_step, llm)

    async def yotta_sub_agent_node_async(state: dict, config: RunnableConfig = None) -> dict:
        step = state["step"]
        t0 = time.monotonic()
        try:
            step_runner = _bind_config(run_step, takes_config, config)
            if takes_config:
                # production runner — bind verifier feedback + streaming
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
                get_max_attempts(step["agent"]),
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


def _build_verifier_context(step_verifications: dict[int, dict]) -> str:
    """Build a context block showing previous verification results per step.

    The verifier uses this to respect the retry cap defined in SKILL.md
    and to avoid repeating notes it already raised in a prior pass.
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
            + (f" (retries used: {retries}/2)" if retries > 0 else "")
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
    """
    plan = state.get("plan", [])
    results = state.get("results", {})
    replan_count = state.get("replan_count", 0)
    step_verifications = state.get("step_verifications", {})

    # Build a synthetic step that describes the verification task
    plan_summary = "\n".join(
        f"Step {s['step']} ({s['agent']}): {s['subtask']}"
        for s in plan
    )

    log_event("verify_node_start", plan_steps=[s["step"] for s in plan],
              result_steps=sorted(k for k in results.keys() if isinstance(k, int)))

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
            + _build_verifier_context(step_verifications)
            + f"\n\n## Sub-agent results\n{results_summary}\n\n"
        ),
        "skills_needed": ["information-verifier"],
        "depends_on": [s["step"] for s in plan],
    }

    # Bounded attempt loop (Slice 3): the verifier's single LLM call runs
    # through the same helper as the workers — no graph-level RetryPolicy.
    _, output, _ = await run_step_with_attempts(
        verify_step,
        partial(
            run_sub_agent_async,
            verify_step,
            results,
            state.get("current_datetime", ""),
            streaming=state.get("streaming", False),
        ),
        get_max_attempts("verifier"),
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
                    # ---- RETRY — re-run the worker (cap at 2) ----------------
                    if current_retries < 2:
                        any_retry = True
                        steps_to_remove.add(sid)   # re-dispatched via pending_retries
                        current_retries += 1
                        acc_notes.append(f"[Retry {current_retries}/2] {notes}")
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
        **updates,
    }


def after_verify(state: dict) -> str:
    """Thin reader: route purely on ``verification_route``, computed once in
    ``verify_node``. No note re-parsing here (fixes F5).

    route == "retry"                          → scheduler (never gated by replan cap — fixes F4)
    route == "replan" and replan_count < cap  → orchestrator
    otherwise                                  → writer (proceed, or replan cap exhausted)
    """
    route = state.get("verification_route", "proceed")
    replan_count = state.get("replan_count", 0)

    if route == "retry":
        return "scheduler"
    if route == "replan" and replan_count < _MAX_REPLANS:
        return "orchestrator"
    return "writer"


async def writer_node(state: dict) -> dict:
    """Assemble sub-agent results (or direct yotta findings) plus verifier notes
    into one comprehensive artefact.

    Two routes reach this node:
    1. **Normal** — orchestrator → sub_agents → verify → writer.
       ``plan`` has steps, ``results`` has sub-agent outputs, and
       ``verification_notes`` may be present.
    2. **Direct** — orchestrator returned an empty plan because the yotta
       search results were already sufficient.
       ``plan`` is ``[]``, ``results`` is ``{}``, ``search_results`` has the
       findings. If files were attached, this route sees their raw content
       directly (there are no document-reader steps to have read them).
    """
    results = state.get("results", {})
    verification_notes = state.get("verification_notes", "")
    plan = state.get("plan", [])
    task = state.get("task", "")
    search_results = state.get("search_results", "")
    files = state.get("files", {})

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

    subtask = (
        "Combine the following information into one comprehensive, "
        "well-structured artefact. Resolve any contradictions and synthesise "
        "the information into a cohesive final document.\n\n"
        + "\n\n".join(blocks)
    )

    write_step = {
        "step": "assemble",
        "agent": "writer",
        "subtask": subtask,
        "skills_needed": ["answer-writer"],
        "depends_on": list(results.keys()),
    }

    # Bounded attempt loop (Slice 3), same as the workers and verifier.
    _, output, _ = await run_step_with_attempts(
        write_step,
        partial(
            run_sub_agent_async,
            write_step,
            results,
            state.get("current_datetime", ""),
            streaming=state.get("streaming", False),
        ),
        get_max_attempts("writer"),
        graph_name="yotta",
    )

    # The assembled document is the terminal product — results must not be
    # repopulated with it (the old ``{"assemble": ...}`` entry polluted the
    # results channel for no reader).
    return {"final_output": output}


# ---------------------------------------------------------------------------
# Citation node (Phase 4 — citation pipeline)
# NOTE: This node is defined but not wired into the graph yet.  It will be
# properly integrated in Phase 4 once the source_map threading and
# structured worker outputs are in place. ``streaming.py``'s ``_AGENT_NODES``
# still references the name.
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

    print("inside yotta_graph")
    builder = StateGraph(YottaState)
    # ValueError from plan validation (1.2) should re-plan, not kill the run —
    # same as the parallel graph. No worker/verify RetryPolicy (Slice 3): the
    # worker owns its bounded attempt loop; verify/writer wrap their single
    # LLM call in ``run_step_with_attempts`` too.
    builder.add_node("orchestrator", orchestrator,
                     retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
    builder.add_node("sub_agent", sub_agent)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("verify", verify_node)
    builder.add_node("writer", writer_node)

    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")
    builder.add_edge("sub_agent", "scheduler")
    builder.add_conditional_edges("scheduler", fan_out_router,
                                  ["writer", "verify", "sub_agent"])
    builder.add_conditional_edges("verify", after_verify,
                                  ["scheduler", "orchestrator", "writer"])
    builder.add_edge("writer", END)

    # Compiled with MemorySaver for interactive use (run_pipeline.py,
    # streaming.py); the API server compiles without a checkpointer.
    return builder.compile(checkpointer=checkpointer or MemorySaver())
