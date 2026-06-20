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
        verify_step, results, state.get("current_datetime", "")
    )

    # Parse the verifier's JSON output to extract the routing decision
    try:
        parsed = _extract_json(output)
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
    return "assemble"


def assemble_node(state: dict) -> dict:
    plan = state.get("plan", [])
    results = state.get("results", {})
    parts = [
        f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
        for s in plan
    ]
    return {"final_output": "\n\n".join(parts)}

async def writer_node(state: dict) -> dict:
    """Assemble all sub-agent results, plus verifier notes, into one comprehensive artefact."""
    results = state.get("results", {})
    verification_notes = state.get("verification_notes", "")

    results_summary = "\n\n".join(
        f"--- Step {step_num} output ---\n{output}"
        for step_num, output in results.items()
    )

    verifier_block = ""
    if verification_notes:
        verifier_block = f"\n\n## Verifier notes\n{verification_notes}"

    write_step = {
        "step": "assemble",
        "agent": "writer",
        "subtask": (
            "Combine the following sub-agent results into one comprehensive, "
            "well-structured artefact. Resolve any contradictions and synthesise "
            "the information into a cohesive final document.\n\n"
            f"## Sub-agent results\n{results_summary}"
        ),
        "skills_needed": ["answer-writer"],
        "depends_on": list(results.keys()),
    }

    step_num, output = await run_sub_agent_async(
        write_step, results, state.get("current_datetime", "")
    )

    return {"final_output": output}


builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent)
builder.add_node("sub_agent", sub_agent_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("verify", verify_node, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("assemble", writer_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", should_continue)
builder.add_conditional_edges("sub_agent", should_continue)
builder.add_conditional_edges("verify", after_verify)
builder.add_edge("assemble", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)