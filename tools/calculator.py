"""
Expression Calculator
=====================

LangChain tool adapter and CLI for the shared expression parser
(``tools/expr_parser.py``).

CLI usage:
  python tools/calculator.py                       # interactive REPL
  python tools/calculator.py "2 + 3 * sqrt(10)"    # single-shot
"""

from __future__ import annotations

import sys
from typing import Sequence

from langchain_core.tools import tool

from tools.expr_parser import evaluate_expression  # re-exported for backward compat

# Also make the error type available for callers that catch it.
from tools.expr_parser import CalculatorError, ErrorCode  # noqa: F401


# ---------------------------------------------------------------------------
# LangChain tool adapter
# ---------------------------------------------------------------------------

@tool
def calculate(expr: str | list[str], include_steps: bool = False) -> dict:
    """Evaluate one or more mathematical expressions.

    Args:
        expr: A single expression string, e.g. ``"2 + 3 * sqrt(10) - e / pi"``,
            or a list of expression strings to evaluate independently.
        include_steps: When True, include a step-by-step breakdown in the
            ``steps`` list of the result envelope.

    Precedence (tightest → loosest):
        postfix factorial (!)  →  power (^, right-assoc)  →  unary (+/-)
        →  * / %  →  + -

    Key behaviours:
        ``2 * 3^2 = 18``, ``2^3^2 = 512``, ``-2^2 = -4``, ``-3! = -6``.
        Factorial, GCD, LCM, nCr, and nPr require integral inputs — they
        no longer silently round.

    A string returns a JSON-compatible envelope:
        success: ``{"ok": true, "expression": …, "result": {"text": …, "kind": …},
        "error": null, "steps": […]}``
        error: ``{"ok": false, "expression": …, "result": null,
        "error": {"code": …, "message": …, "position": …}, "steps": []}``

    A list returns a batch envelope; each item is evaluated independently,
    so a failing item does not stop the others:
        ``{"ok": …, "expression": [<each input string>], "results": [
        {"text": …, "kind": …} or {"error": {"index": …, "code": …,
        "message": …, "position": …}}, …], "error": null, "steps": …}``

    In the batch envelope, ``ok`` is false if ANY item failed, and ``error``
    is always null (failures live in the per-item ``results`` entries;
    ``index`` is the item's zero-based position in the list and ``position``
    is relative to that item's string).  Successful items keep their
    position and appear as ``{"text", "kind"}``.  With ``include_steps=True``,
    ``steps`` is a list of lists aligned with ``results`` (``[]`` for failed
    items); with the default it is ``[]``.  A comma-separated string inside
    a list is rejected per-item — pass it as a single string instead.  An
    empty list is an argument error (``ok`` false, top-level ``error``
    ``invalid_arguments``).
    """
    if isinstance(expr, str):
        return evaluate_expression(expr, include_steps=include_steps)

    # ---- list input: independent per-item evaluation ----------------------
    if not expr:
        return {
            "ok": False,
            "expression": [],
            "results": [],
            "error": {
                "code": "invalid_arguments",
                "message": "expr list must contain at least one expression",
                "position": None,
            },
            "steps": [],
        }

    results: list[dict] = []
    steps: list[list[str]] = []
    ok = True
    for i, item in enumerate(expr):
        env = evaluate_expression(item, include_steps=include_steps)
        if env["ok"] and env["result"] is not None:
            results.append({
                "text": env["result"]["text"],
                "kind": env["result"]["kind"],
            })
            steps.append(env["steps"])
        elif env["ok"]:
            # Comma-separated item — only supported via the string path.
            results.append({
                "error": {
                    "index": i,
                    "code": "invalid_arguments",
                    "message": (
                        "list item is a comma-separated multi-expression; "
                        "pass it as a single string or split it into "
                        "separate list items"
                    ),
                    "position": None,
                }
            })
            steps.append([])
            ok = False
        else:
            results.append({"error": {"index": i, **env["error"]}})
            steps.append([])
            ok = False

    return {
        "ok": ok,
        "expression": list(expr),
        "results": results,
        "error": None,
        "steps": steps if include_steps else [],
    }


# ---------------------------------------------------------------------------
# CLI / REPL rendering helpers
# ---------------------------------------------------------------------------

def _print_result(envelope: dict, *, show_steps: bool = True) -> None:
    """Render a result envelope to stdout for CLI/REPL use."""
    if envelope["ok"]:
        result = envelope["result"]
        if show_steps and envelope["steps"]:
            print("Steps:")
            for step in envelope["steps"]:
                print(step)
        if result is not None:
            print(result["text"])
        elif "results" in envelope:
            for r in envelope["results"]:
                print(r["text"])
    else:
        err = envelope["error"]
        msg = f"Error: {err['message']}"
        if err.get("position") is not None:
            msg += f" (position {err['position']})"
        print(msg)


def repl() -> None:
    """Interactive REPL loop."""
    print("Expression Calculator  (type 'quit' or Ctrl-C to exit)")
    print("Operators : + - * / ^ % !")
    print("Constants : pi, e, phi, tau, sqrt2, ln2, ...")
    print("Functions : sqrt, sin, cos, log, exp, fact, nCr, ...")
    print()
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        if line.lower() in ('quit', 'exit', 'q'):
            print("Bye.")
            break
        envelope = evaluate_expression(line, include_steps=True)
        _print_result(envelope)


def main(argv: Sequence[str]) -> int:
    """CLI entry point.  Returns an exit code (0 = success, 1 = error)."""
    if len(argv) > 1:
        expr = " ".join(argv[1:])
        envelope = evaluate_expression(expr, include_steps=True)
        _print_result(envelope)
        return 0 if envelope["ok"] else 1

    repl()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
