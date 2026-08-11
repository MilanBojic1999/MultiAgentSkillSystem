"""Minimal evaluation harness (plan 4.10).

Runs golden YAML tasks through a *real* orchestrator + stub sub-agent graph
(one LLM call per task, no tools executed) and asserts on the shape of the
resulting plan.  Plan-shape only; no output-quality assertions, no judge.

Usage::

    python -m evals --graph parallel [--tasks-dir DIR] [--verbose]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from agents.agent_states import get_current_datetime_str
from graphs import build_graph
from utils.logger import log_event


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def default_tasks_dir() -> Path:
    """``evals/tasks`` resolved from this package, so CWD never matters."""
    return Path(__file__).resolve().parent / "tasks"


# ---------------------------------------------------------------------------
# Stub worker
# ---------------------------------------------------------------------------

def stub_worker(state: dict) -> dict:
    """Stand-in sub-agent: one canned result per step, no tools, no LLM.

    Same contract as the test fakes in ``tests/test_dispatch_dedup.py`` — the
    ``results`` reducer merges the dict, so this works for parallel and
    sequential graphs alike.
    """
    n = state["step"]["step"]
    return {"results": {n: f"eval-out-{n}"}}


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

#: name → (predicate, label).  Canonical column order for table output.
#  Each predicate is ``(plan, expected_value) → bool``.
ASSERTIONS: dict[str, tuple[Any, str]] = {
    "min_steps":      (lambda p, v: len(p) >= v,
                       "len(plan)"),
    "max_steps":      (lambda p, v: len(p) <= v,
                       "len(plan)"),
    "agents_include": (lambda p, v: set(v) <= {s["agent"] for s in p},
                       "agents"),
    "has_dependency": (lambda p, v: any(s.get("depends_on") for s in p) == v,
                       "deps"),
}


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(tasks_dir: Path) -> list[tuple[str, dict]]:
    """Load every ``*.yaml`` gold-task from *tasks_dir*, sorted by stem.

    Returns:
        ``[(name, task_dict), ...]`` where *task_dict* has keys ``task``
        (str, required) and ``expect`` (dict, defaults to ``{}``).

    Skips files that cannot be parsed, with a warning to stderr.
    """
    if not tasks_dir.is_dir():
        raise ValueError(f"Tasks directory not found: {tasks_dir}")

    tasks: list[tuple[str, dict]] = []
    for path in sorted(tasks_dir.glob("*.yaml")):
        name = path.stem
        try:
            raw = yaml.safe_load(path.read_text())
        except (yaml.YAMLError, OSError) as exc:
            log_event("eval_task_unreadable", file=name, error=str(exc))
            print(f"evals: skipping {path.name}: {exc}", file=sys.stderr)
            continue

        if not isinstance(raw, dict):
            log_event("eval_task_unreadable", file=name,
                      error="top level is not a dict")
            print(f"evals: skipping {path.name}: top level is not a dict",
                  file=sys.stderr)
            continue

        task_text = raw.get("task", "")
        if not isinstance(task_text, str) or not task_text.strip():
            log_event("eval_task_unreadable", file=name,
                      error="missing or empty 'task'")
            print(f"evals: skipping {path.name}: missing or empty 'task'",
                  file=sys.stderr)
            continue

        expect = raw.get("expect")
        if expect is None:
            expect = {}
        elif not isinstance(expect, dict):
            log_event("eval_task_unreadable", file=name,
                      error="'expect' is not a dict")
            print(f"evals: skipping {path.name}: 'expect' is not a dict",
                  file=sys.stderr)
            continue

        tasks.append((name, {"task": task_text, "expect": expect}))

    return tasks


# ---------------------------------------------------------------------------
# Plan checking
# ---------------------------------------------------------------------------

def check_plan(plan: list[dict] | None, expect: dict) -> dict:
    """Assert *plan* shape against *expect*.

    Returns:
        ``{"passed": bool, "details": {name: (ok, actual, expected)}}``.
        Unknown keys in *expect* are silently ignored (forward-compatible).
    """
    plan = plan or []
    passed = True
    details: dict[str, tuple[bool, Any, Any]] = {}

    for name, expected in expect.items():
        entry = ASSERTIONS.get(name)
        if entry is None:
            log_event("eval_unknown_assertion", name=name)
            continue

        pred, _label = entry

        # Compute the *actual* value for this assertion
        if name == "min_steps" or name == "max_steps":
            actual = len(plan)
        elif name == "agents_include":
            actual = sorted({s["agent"] for s in plan})
        elif name == "has_dependency":
            actual = any(s.get("depends_on") for s in plan)
        else:  # pragma: no cover — defensive
            actual = None

        ok = bool(pred(plan, expected))
        details[name] = (ok, actual, expected)
        if not ok:
            passed = False

    return {"passed": passed, "details": details}


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

_MAX_TASK_WIDTH = 32
_CELL_PAD = 2


def _render_table(rows: list[dict], columns: list[str], verbose: bool) -> None:
    """Print evaluation results as a human-readable table."""
    # --- column widths ---
    widths: dict[str, int] = {}
    for col in columns:
        widths[col] = len(col)
        for row in rows:
            cell = row.get(col, "")
            widths[col] = max(widths[col], len(str(cell)))

    # --- header ---
    header = "".join(col.ljust(widths[col] + _CELL_PAD) for col in columns)
    print(header)
    print("-" * len(header.rstrip()))

    # --- rows ---
    for row in rows:
        parts: list[str] = []
        for col in columns:
            parts.append(str(row.get(col, "")).ljust(widths[col] + _CELL_PAD))
        print("".join(parts).rstrip())

    # --- summary ---
    total = len(rows)
    n_pass = sum(1 for r in rows if r["result"] == "PASS")
    n_fail = sum(1 for r in rows if r["result"] == "FAIL")
    n_err  = sum(1 for r in rows if r["result"] == "ERROR")
    print()
    print(f"{total} task{'s' if total != 1 else ''}: "
          f"{n_pass} passed, {n_fail} failed, {n_err} errored")


def _truncate(text: str, width: int = _MAX_TASK_WIDTH) -> str:
    """Truncate *text* to *width* chars, appending '…' when truncated."""
    if len(text) <= width:
        return text
    return text[:width - 1] + "…"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evals(
    graph_name: str,
    tasks_dir: Path,
    *,
    verbose: bool = False,
) -> bool:
    """Run every golden task from *tasks_dir* against *graph_name*.

    Args:
        graph_name: Name of a graph in ``graphs/`` (e.g. ``"parallel"``).
        tasks_dir: Directory of ``*.yaml`` golden-task files.
        verbose: If ``True``, print task text before each row and
            per-assertion detail on FAIL rows.

    Returns:
        ``True`` iff every task passed (i.e. all rows are ``PASS``).
        Returns ``False`` for an empty sweep, any FAIL, or any ERROR.
    """
    tasks = load_tasks(tasks_dir)
    if not tasks:
        print(f"evals: no tasks found in {tasks_dir}", file=sys.stderr)
        return False

    # Build once per sweep — real orchestrator (default), stub worker.
    try:
        graph = build_graph(graph_name, sub_agent=stub_worker)
    except ValueError:
        raise  # unknown graph name → let the CLI surface it

    # Determine table columns from the union of assertion keys across tasks,
    # in the canonical ASSERTIONS order.
    all_keys: set[str] = set()
    for _name, task in tasks:
        all_keys.update(task.get("expect", {}))
    columns = ["task"] + [k for k in ASSERTIONS if k in all_keys] + ["result"]

    # Collect rows.
    rows: list[dict] = []
    for name, task in tasks:
        thread_id = f"eval-{graph_name}-{name}"
        expect = task.get("expect", {})

        if verbose:
            task_text = task["task"]
            dashes = "-" * min(len(task_text), 72)
            print(f"\n{dashes}\n{task_text}\n{dashes}")

        # --- invoke ---
        try:
            out = graph.invoke(
                {
                    "task": task["task"],
                    "current_datetime": get_current_datetime_str(),
                },
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as exc:
            log_event("eval_task_error", task=name, error=str(exc))
            row: dict[str, str] = {"task": _truncate(task["task"]), "result": "ERROR"}
            for k in columns:
                if k not in row:
                    row[k] = "-"
            rows.append(row)
            if verbose:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue

        # --- check plan ---
        plan: list[dict] = out.get("plan") or []
        res = check_plan(plan, expect)

        status = "PASS" if res["passed"] else "FAIL"
        row = {"task": _truncate(task["task"]), "result": status}
        for k in columns:
            if k in ("task", "result"):
                continue
            detail = res["details"].get(k)
            if detail is None:
                row[k] = "–"
            else:
                ok, _actual, _expected = detail
                row[k] = "✓" if ok else "✗"

        rows.append(row)

        if verbose and status == "FAIL":
            for k in columns:
                if k in ("task", "result"):
                    continue
                detail = res["details"].get(k)
                if detail is None:
                    continue
                ok, actual, expected = detail
                label = ASSERTIONS[k][1] if k in ASSERTIONS else k
                if not ok:
                    print(f"  ✗ {k}: {label} → actual={actual!r}, expected={expected!r}")

    # --- header line ---
    print(f"\ngraph: {graph_name}        tasks: {tasks_dir}\n")

    # --- table ---
    _render_table(rows, columns, verbose)

    return all(r["result"] == "PASS" for r in rows)
