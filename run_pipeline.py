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

from graphs import build_graph, graph_descriptions
from agents.agent_states import get_current_datetime_str

DEFAULT_GRAPH = "parallel"


def run(task: str, graph_name: str = DEFAULT_GRAPH) -> tuple[str, list[dict]]:
    """Run the pipeline synchronously.

    Returns ``(final_output, step_stats)`` (Phase 4.9).
    """
    config = {"configurable": {"thread_id": "test-run-1"}}
    graph = build_graph(graph_name)
    result = graph.invoke({"task": task, "current_datetime": get_current_datetime_str()}, config=config)
    return (
        result.get("final_output", "No final output produced."),
        result.get("step_stats", []),
    )


async def run_async(task: str, graph_name: str = DEFAULT_GRAPH) -> tuple[str, list[dict]]:
    """Run the pipeline asynchronously.

    Returns ``(final_output, step_stats)`` (Phase 4.9).
    """
    config = {"configurable": {"thread_id": "test-run-1"}}
    graph = build_graph(graph_name)
    result = await graph.ainvoke({"task": task, "current_datetime": get_current_datetime_str()}, config=config)
    return (
        result.get("final_output", "No final output produced."),
        result.get("step_stats", []),
    )


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

    print(f"Running pipeline '{args.graph}' with task:\n  {task}\n")
    print("=" * 60)
    output, step_stats = asyncio.run(run_async(task, args.graph))
    print("=" * 60)
    print(output)

    if step_stats:
        print()
        print("=" * 60)
        print("Step stats")
        print("=" * 60)
        _print_stats(step_stats)
