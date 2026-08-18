"""CLI entry point: ``python -m evals --graph <name> [--tasks-dir DIR] [--verbose]``.

Exit codes:
    0 — all tasks passed
    1 — at least one task FAIL or ERROR, or an empty sweep
    2 — usage / configuration error (bad graph name, missing tasks dir)
"""

import argparse
import sys
from pathlib import Path

from evals.runner import default_tasks_dir, run_evals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=(
            "Run golden plan-shape tasks through a graph's real orchestrator "
            "with a stub worker (one LLM call per task)."
        ),
    )
    parser.add_argument(
        "--graph", "-g",
        default="parallel",
        help="Graph name from graphs/ (default: parallel).",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help=(
            "Directory of *.yaml golden tasks "
            "(default: evals/tasks/ next to the package)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print task texts and per-assertion details.",
    )
    args = parser.parse_args(argv)

    tasks_dir = args.tasks_dir or default_tasks_dir()

    try:
        ok = run_evals(args.graph, tasks_dir, verbose=args.verbose)
    except ValueError as exc:
        # Unknown graph (build_graph) or missing tasks dir
        print(f"evals: {exc}", file=sys.stderr)
        return 2

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
