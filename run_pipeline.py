"""
Simple entrypoint to run the LangGraph multi-agent pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py "Calculate sin(pi/4) + cos(pi/4) and plot both functions"
    python run_pipeline.py --graph sequential "Summarise the history of calculus"
    python run_pipeline.py --list-graphs
"""

import argparse
import asyncio
import sys
import uuid

from graphs import build_graph, graph_descriptions
from agents.agent_states import PipelineResult, get_current_datetime_str
from assemble_node import pipeline_result
from execution_policy import (
    EFFORT_PRESETS,
    DEFAULT_EFFORT,
    graph_effort_compat_error,
    normalize_effort,
    resolve_execution_policy,
)
from utils.logger import log_event

# The effort-aware default topology (effort slider): yotta provides the
# planner -> workers -> verifier -> writer lifecycle; ``parallel`` and
# ``sequential`` remain explicitly selectable compatibility graphs.
DEFAULT_GRAPH = "yotta"


def _resolve_effort_and_graph(graph_name: str, effort: str | None) -> tuple[str, str]:
    """Mirror the API boundary's compatibility contract for direct CLI runs.

    - verification-promising efforts on legacy non-verifying graphs -> error;
    - ``instant`` always executes on yotta (its only implementation);
    - omitted effort resolves to ``unlimited`` (legacy behavior).
    """
    preset = normalize_effort(effort)
    error = graph_effort_compat_error(graph_name, preset)
    if error:
        raise ValueError(error)
    if preset == "instant":
        log_event(
            "instant_route_selected",
            requested_graph=(graph_name or "").strip() or None,
        )
        return "yotta", preset
    return graph_name, preset


def _run_config(effort: str | None = None) -> dict:
    """Build the run config for one pipeline run.

    ``task_id`` keys the run's artifact directory (plan 4.5), so every CLI
    run writes generated files (e.g. plots) into its own ``artifacts/<id>/``
    instead of overwriting the previous run's files. ``effort`` resolves
    through the shared policy module and travels under ``configurable``
    (serializable only — checkpoint-safe).
    """
    preset = normalize_effort(effort)
    policy = resolve_execution_policy(preset)
    log_event("execution_policy_resolved", effort=preset,
              execution_policy=policy.as_dict())
    return {"configurable": {
        "thread_id": "test-run-1",
        "task_id": uuid.uuid4().hex[:12],
        "effort": preset,
        "execution_policy": policy.as_dict(),
    }}


def run(task: str, graph_name: str = DEFAULT_GRAPH,
        effort: str | None = None) -> PipelineResult:
    """Run the pipeline synchronously.

    Returns the typed result: ``status`` (``completed``/``partial``),
    ``final_output``, ``failed_steps``, ``skipped_steps``, ``step_stats``
    (Slice 4) plus effort/verification metadata.
    """
    graph_name, preset = _resolve_effort_and_graph(graph_name, effort)
    graph = build_graph(graph_name)
    result = graph.invoke({"task": task, "current_datetime": get_current_datetime_str()}, config=_run_config(preset))
    return pipeline_result(result)


async def run_async(task: str, graph_name: str = DEFAULT_GRAPH,
                    effort: str | None = None) -> PipelineResult:
    """Run the pipeline asynchronously.

    Returns the same typed result as :func:`run` (Slice 4).
    """
    graph_name, preset = _resolve_effort_and_graph(graph_name, effort)
    graph = build_graph(graph_name)
    result = await graph.ainvoke({"task": task, "current_datetime": get_current_datetime_str()}, config=_run_config(preset))
    return pipeline_result(result)


def _print_stats(step_stats: list[dict]) -> None:
    """Pretty-print per-step execution statistics (Phase 4.9)."""
    # Column widths
    header = f"{'Step':>4}  {'Agent':<20}  {'Status':<10}  {'Duration':>8}  {'In Tok':>7}  {'Out Tok':>7}  {'#Tools':>6}"
    print(header)
    print("-" * len(header))
    total_in = 0
    total_out = 0
    total_tools = 0
    total_duration = 0.0
    for s in sorted(step_stats, key=lambda s: s["step"]):
        print(
            f"{s['step']:>4}  {s['agent']:<20}  {s['status']:<10}  "
            f"{s['duration_s']:>7.3f}s  {s['input_tokens']:>7}  "
            f"{s['output_tokens']:>7}  {s['tool_calls']:>6}"
        )
        total_in += s["input_tokens"]
        total_out += s["output_tokens"]
        total_tools += s["tool_calls"]
        total_duration += s["duration_s"]
    print("-" * len(header))
    print(
        f"{'':>4}  {'':<20}  {'':<10}  "
        f"{total_duration:>7.3f}s  {total_in:>7}  "
        f"{total_out:>7}  {total_tools:>6}"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a task through the multi-agent LangGraph pipeline.",
    )
    parser.add_argument(
        "task",
        nargs="*",
        help="The task to run (quoting is optional — all words are joined).",
    )
    parser.add_argument(
        "--graph", "-g",
        default=DEFAULT_GRAPH,
        help=f"Which graph in graphs/ to run (default: {DEFAULT_GRAPH}). "
             f"See --list-graphs.",
    )
    parser.add_argument(
        "--effort",
        type=str.lower,
        choices=list(EFFORT_PRESETS),
        default=None,
        metavar="{instant,light,standard,thorough,unlimited}",
        help=f"Execution effort preset for this run (case-insensitive; "
             f"default: {DEFAULT_EFFORT}). Per-run only — never persisted.",
    )
    parser.add_argument(
        "--list-graphs",
        action="store_true",
        help="List the graphs discovered in graphs/ and exit.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    if args.list_graphs:
        for name, description in graph_descriptions().items():
            marker = " (default)" if name == DEFAULT_GRAPH else ""
            print(f"  {name}{marker}" + (f" — {description}" if description else ""))
        sys.exit(0)

    task = " ".join(args.task) or (
        "Maximum of x^2*sin(x) in the range between 0 and 2 "
        "Then write a short summary of what the calculation means."
    )

    # Same compatibility contract as the API boundary: verification-promising
    # efforts require yotta; instant always executes on yotta.
    try:
        graph, effort = _resolve_effort_and_graph(args.graph, args.effort)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Running pipeline '{graph}' (effort: {effort}) with task:\n  {task}\n")
    print("=" * 60)
    result = asyncio.run(run_async(task, graph, effort))
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(result["final_output"])

    if result["step_stats"]:
        print()
        print("=" * 60)
        print("Step stats")
        print("=" * 60)
        _print_stats(result["step_stats"])
