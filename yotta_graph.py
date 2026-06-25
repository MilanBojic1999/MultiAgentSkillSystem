from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agent_states import AgentState
from agents.orchestrator_node import orchestrator_agent
from agents.sub_agents_nodes import run_sub_agent_async

import json
import re


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

    for step in plan:
        if step["step"] in results:
            continue
        if all(d in results for d in step.get("depends_on", [])):
            step_num, output = await run_sub_agent_async(
                step, results, current_datetime, streaming
            )
            return {"results": {step_num: output}}

    return {}


def should_continue(state: dict) -> str:
    plan = state.get("plan", [])
    results = state.get("results", {})
    if len(plan) == 0:
        return "writer"
    if len(results) < len(plan):
        return "sub_agent"
    return "verify"


def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose
    """
    # Try direct parse first (best case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find the outermost JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"Could not extract JSON from response: {text[:500]}...")


async def verify_node(state: dict) -> dict:
    """Run the verifier agent against all sub-agent results, parse the verdict,
    and persist it so the conditional edge can route accordingly."""
    plan = state.get("plan", [])
    results = state.get("results", {})

    # Build a synthetic step that describes the verification task
    plan_summary = "\n".join(
        f"Step {s['step']} ({s['agent']}): {s['subtask']}"
        for s in plan
    )
    results_summary = "\n\n".join(
        f"--- Step {step_num} output ---\n{output}"
        for step_num, output in results.items()
    )

    verify_step = {
        "step": "verify",
        "agent": "verifier",
        "subtask": (
            "Verify the accuracy, completeness, and consistency of the sub-agent results below.\n\n"
            f"## Original plan\n{plan_summary}\n\n"
            f"## Sub-agent results\n{results_summary}\n\n"
            "Return your verdict as JSON with 'verification_result' (PASSED / PASSED WITH NOTES / FAILED) and 'notes'."
        ),
        "skills_needed": ["information-verifier"],
        "depends_on": [s["step"] for s in plan],
    }

    step_num, output = await run_sub_agent_async(
        verify_step, results, state.get("current_datetime", ""),
        streaming=state.get("streaming", False),
    )

    # Parse the verifier's JSON output to extract the routing decision
    try:
        parsed = _extract_json(output)
        print(f"Parsed verifier output: {parsed}")
        if isinstance(parsed, list):
            parsed = parsed[0]
        verification_result = parsed.get("verification_result", "PASSED").upper()
        verification_notes = parsed.get("notes", "")
    except (ValueError, json.JSONDecodeError):
        # If parsing fails, check for keywords in the raw output
        verification_result = "PASSED"  # default
        upper_output = output.upper()
        if "FAILED" in upper_output:
            verification_result = "FAILED"
        elif "PASSED WITH NOTES" in upper_output:
            verification_result = "PASSED WITH NOTES"
        verification_notes = output

    return {
        "verification_result": verification_result,
        "verification_notes": verification_notes,
        "final_output": output,
    }


def after_verify(state: dict) -> str:
    """Route based on the verifier's verdict.
    FAILED → re-orchestrate with feedback;
    PASSED / PASSED WITH NOTES → proceed to assembly."""
    verdict = state.get("verification_result", "PASSED").upper()
    if verdict == "FAILED":
        return "orchestrator"
    return "writer"


def assemble_node(state: dict) -> dict:
    plan = state.get("plan", [])
    results = state.get("results", {})
    parts = [
        f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
        for s in plan
    ]
    return {"final_output": "\n\n".join(parts)}

async def writer_node(state: dict) -> dict:
    """Assemble sub-agent results (or direct yotta findings) plus verifier notes
    into one comprehensive artefact.

    Two routes reach this node:
    1. **Normal** — orchestrator → sub_agents → verify → writer.
       ``plan`` has steps, ``results`` has sub-agent outputs, and
       ``verification_notes`` may be present.
    2. **Direct** — orchestrator returned an empty plan because the yotta
       search results were already sufficient.
       ``plan`` is ``[]``, ``results`` is ``{0: <task with yotta findings>}``.
    """
    results = state.get("results", {})
    verification_notes = state.get("verification_notes", "")
    plan = state.get("plan", [])
    task = state.get("task", "")

    # ---- build the writer's prompt blocks -----------------------------------
    blocks: list[str] = []

    # Original query — strip the "Query: ..." prefix and search-results suffix
    # so the writer sees the user's actual question.
    clean_query = task
    if "\n\n## Search results" in task:
        clean_query = task.split("\n\n## Search results")[0].strip()
        clean_query = re.sub(r"^Query:\s*", "", clean_query)
    blocks.append(f"## Original query\n{clean_query}")

    # Original plan (normal route only)
    if plan:
        plan_summary = "\n".join(
            f"Step {s['step']} ({s['agent']}): {s['subtask']}"
            for s in plan
        )
        blocks.append(f"## Original plan\n{plan_summary}")

    # Sub-agent results — skip step 0 (it's just the task string stored by the
    # orchestrator); real sub-agent outputs have step numbers >= 1.
    if plan:
        results_summary = "\n\n".join(
            f"--- Step {step_num} output ---\n{output}"
            for step_num, output in sorted(results.items())
            if isinstance(step_num, int) and step_num >= 1
        )
        if results_summary:
            blocks.append(f"## Sub-agent results\n{results_summary}")

    # Yotta / search findings — always include if the task embeds them (both
    # routes benefit: in the direct route they are the primary material; in the
    # normal route they are supplementary grounding).
    if "\n\n## Search results" in task:
        search_block = task.split("\n\n## Search results", 1)[1].strip()
        if search_block:
            blocks.append(f"## Search results (initial grounding)\n{search_block}")

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

    return {"final_output": output}

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
        parsed = _extract_json(output)
        final_answer = parsed.get("final_answer", output)
    except (ValueError, json.JSONDecodeError):
        final_answer = output

    return {"final_output": final_answer}


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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)