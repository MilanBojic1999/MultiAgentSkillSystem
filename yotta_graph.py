from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent, _MAX_REPLANS
from agents.sub_agents_nodes import run_sub_agent_async

from utils.json_utils import extract_json
from pipeline_entry import render_files_block

import json
import re


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


async def sub_agent_node(state: dict) -> dict:
    """
    Async sequential node: executes the next uncompleted step in the plan
    whose dependencies are all satisfied.  Mirrors the pattern in
    ``paralel_pipeline_graph.py`` — await the LLM call directly instead of
    wrapping it in ``asyncio.run()``.
    """
    plan = state["plan"]
    results = state.get("results", {})
    current_datetime = state.get("current_datetime", "")
    streaming = state.get("streaming", False)
    files = state.get("files", {})
    step_verifications = state.get("step_verifications", {})

    plan_step_ids = {s["step"] for s in plan}
    any_ready = False

    for step in plan:
        if step["step"] in results:
            continue
        if all(d in results for d in step.get("depends_on", [])):
            any_ready = True
            feedback = _latest_retry_note(step_verifications, step["step"])
            step_num, output = await run_sub_agent_async(
                step, results, current_datetime, streaming, files, feedback
            )
            return {"results": {step_num: output}}

    # ---- deadlock detection -------------------------------------------------
    # If we get here, every remaining step has an unmet dependency.
    # This is a bug in the plan — fail loudly instead of spinning to the
    # recursion limit.
    done = set(results.keys())
    remaining = plan_step_ids - done
    if remaining:
        blocker_info = []
        for s in plan:
            if s["step"] in remaining:
                unmet = [d for d in s.get("depends_on", []) if d not in done]
                blocker_info.append(
                    f"  Step {s['step']} ('{s['subtask']}') "
                    f"waiting on unmet deps: {unmet}"
                )
        raise RuntimeError(
            f"Deadlock: {len(remaining)} step(s) can never execute because "
            f"their dependencies will never be satisfied.\n"
            + "\n".join(blocker_info)
        )

    return {}


def should_continue(state: dict) -> str:
    """Set-based routing: route on which plan steps still need results.

    Returns:
        ``"writer"``   — plan is empty (search results were sufficient)
        ``"sub_agent"`` — at least one plan step is ready or pending
        ``"verify"``   — all plan steps have results
    """
    plan = state.get("plan", [])
    results = state.get("results", {})

    if len(plan) == 0:
        return "writer"

    plan_step_ids = {s["step"] for s in plan}
    done_ids = set(results.keys())
    remaining = plan_step_ids - done_ids

    if not remaining:
        return "verify"
    return "sub_agent"



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

    print("Verifier input:", results)
    print("-" * 50)
    print("+" * 50)

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

    step_num, output = await run_sub_agent_async(
        verify_step, results, state.get("current_datetime", ""),
        streaming=state.get("streaming", False),
    )

    # Parse the verifier's JSON output — expect a list of per-subquery
    # verdicts (the SKILL.md spec) with a single-object fallback.
    # Accumulate into existing step_verifications so notes survive across
    # multiple verify cycles (issue #2).
    try:
        parsed = extract_json(output)
        print(f"Parsed verifier output: {len(results)} : {parsed}")
        if not isinstance(parsed, list):
            parsed = [parsed]

        # ---- map each verifier entry to a plan step ---------------------------
        # Prefer the numeric `step` field (S1 contract); fall back to the
        # legacy subquery-text matcher for older-format verifier output.
        plan_step_ids = {s["step"] for s in plan}
        per_step: dict[int, dict] = {}
        unmapped_failed_notes: list[str] = []
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
                print(
                    f"  [verifier] could not map verdict to any plan step: "
                    f"step={raw_step!r} subquery={item.get('subquery', '')!r}"
                )
                if verdict == "FAILED":
                    # Never drop a FAILED verdict silently just because it
                    # couldn't be mapped — surface it as a replan signal so
                    # the orchestrator sees it instead of it vanishing
                    # (fixes F1).
                    unmapped_failed_notes.append(notes or "(no notes provided)")
                continue

            per_step[sid] = {"verdict": verdict, "notes": notes}

        # ---- accumulate with existing step_verifications ----------------------
        # Seed from ALL existing entries so retry counts and prior notes survive
        # for steps that are being re-verified in this pass (fixes Bug 2).
        accumulated: dict[int, dict] = {
            sid: dict(entry) for sid, entry in step_verifications.items()
        }

        any_replan = False
        any_retry = False
        steps_to_remove: set[int] = set()

        for sid, v in per_step.items():
            verdict = v["verdict"]
            notes   = v["notes"]

            # fetch or initialise accumulated entry
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

            if verdict == "FAILED":
                if notes.upper().startswith("REPLAN:"):
                    # ---- REPLAN — needs orchestrator --------------------------
                    any_replan = True
                    acc_notes.append(f"[Replan] {notes}")
                else:
                    # ---- RETRY — re-run the worker (cap at 2) ----------------
                    if current_retries < 2:
                        any_retry = True
                        steps_to_remove.add(sid)    # remove so it gets re-run
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
            acc["notes"]   = "\n".join(acc_notes)
            acc["retries"] = current_retries
            accumulated[sid] = acc

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

        # ---- remove retry-cleared steps from results via clear sentinel ----
        # Only when the route is actually "retry" — on a mixed/replan pass the
        # orchestrator wipes results itself, so deleting them here would just
        # discard research the writer might otherwise fall back to (fixes F4).
        if steps_to_remove and route == "retry":
            kept_results = {k: v for k, v in results.items() if k not in steps_to_remove}
            results_update: dict[int, str] = {-1: "", **kept_results}  # type: ignore[dict-item]
        else:
            results_update = {}

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
        results_update = {}
        route = "replan" if aggregate_verdict == "FAILED" else "proceed"

    # Increment replan_count only when we're actually routing to the
    # orchestrator — pure retries have their own per-step cap and don't
    # count against it (fixes F5's un-counted-replan bug).
    new_replan_count = replan_count
    if route == "replan":
        new_replan_count = replan_count + 1

    return {
        "verification_result": aggregate_verdict,
        "verification_notes": verification_notes,
        "verifier_report": output,
        "replan_count": new_replan_count,
        "step_verifications": accumulated,
        "results": results_update,
        "verification_route": route,
    }


def after_verify(state: dict) -> str:
    """Thin reader: route purely on ``verification_route``, computed once in
    ``verify_node``. No note re-parsing here (fixes F5).

    route == "retry"                          → sub_agent  (never gated by replan cap — fixes F4)
    route == "replan" and replan_count < cap  → orchestrator
    otherwise                                  → writer (proceed, or replan cap exhausted)
    """
    route = state.get("verification_route", "proceed")
    replan_count = state.get("replan_count", 0)

    if route == "retry":
        return "sub_agent"
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

    print("Writer input:", results)
    print("-"*50)
    print("+"*50, flush=True)


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
    print("Writer results:")
    for k, v in results.items():
        print(k,"___",v)
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

    print(subtask)
    print("_" * 50)
    print("_" * 50)

    write_step = {
        "step": "assemble",
        "agent": "writer",
        "subtask": subtask,
        "skills_needed": ["answer-writer"],
        "depends_on": list(results.keys()),
    }

    step_num, output = await run_sub_agent_async(
        write_step, results, state.get("current_datetime", ""),
        streaming=state.get("streaming", False),
    )

    return {"final_output": output, "results": {step_num: output}}

# ---------------------------------------------------------------------------
# Citation node (Phase 4 — citation pipeline)
# NOTE: This node is defined but not wired into the graph yet.  It will be
# properly integrated in Phase 4 once the source_map threading and
# structured worker outputs are in place.
# ---------------------------------------------------------------------------

async def citatitaion_node(state: dict) -> dict:
    """Citation & QA gate: runs after the writer, checks the draft against the
    source map and verified findings, then produces the final answer."""
    print("Citatitaion input:", state)
    print("+"*50)
    print("+"*50)
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

    print("CItatitaion intput:", cite_step)
    print("&"*50)

    step_num, output = await run_sub_agent_async(
        cite_step, results, state.get("current_datetime", ""),
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

# The builder is exported so callers can compile with or without a checkpointer
# (the API server compiles without one to avoid unbounded MemorySaver growth).
builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent)
builder.add_node("sub_agent", sub_agent_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("verify", verify_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("writer", writer_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", should_continue)
builder.add_conditional_edges("sub_agent", should_continue)
builder.add_conditional_edges("verify", after_verify)
builder.add_edge("writer", END)

# Compiled with MemorySaver for interactive use (run_pipeline.py, streaming.py).
# The API server should compile without a checkpointer to prevent unbounded
# checkpoint growth (see api_server.py).
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)