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
def calculate(expr: str, include_steps: bool = False) -> dict:
    """Evaluate a mathematical expression string.

    Args:
        expr: The expression to evaluate, e.g. ``"2 + 3 * sqrt(10) - e / pi"``.
        include_steps: When True, include a step-by-step breakdown in the
            ``steps`` list of the result envelope.

    Precedence (tightest → loosest):
        postfix factorial (!)  →  power (^, right-assoc)  →  unary (+/-)
        →  * / %  →  + -

    Key behaviours:
        ``2 * 3^2 = 18``, ``2^3^2 = 512``, ``-2^2 = -4``, ``-3! = -6``.
        Factorial, GCD, LCM, nCr, and nPr require integral inputs — they
        no longer silently round.

    Returns a JSON-compatible envelope:
        success: ``{"ok": true, "expression": …, "result": {"text": …, "kind": …},
        "error": null, "steps": […]}``
        error: ``{"ok": false, "expression": …, "result": null,
        "error": {"code": …, "message": …, "position": …}, "steps": []}``
    """
    return evaluate_expression(expr, include_steps=include_steps)


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
