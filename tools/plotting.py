"""Plot a mathematical expression of *x* to an image file.

Uses the shared expression parser (``tools/expr_parser.py``) for the same
syntax as ``tools/calculator.py`` — ``^`` for power, ``asin``/``acos``/``atan``
for inverse trig, etc.

The expression is compiled to a numpy-compatible callable and evaluated across
the requested *x*-range.

Supported functions (numpy-vectorised):
  sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
  exp, log, log2, log10, sqrt, cbrt, abs,
  ceil, floor, sign, round, atan2, pow, hypot,
  max (element-wise, 2 args), min (element-wise, 2 args)

Supported constants:
  pi, e, phi, tau, sqrt2, sqrt3, ln2, ln10, Inf

.. note::
   Discrete-math functions (fact/!, nCr, nPr, gcd, lcm) are not supported
   for plotting — they do not vectorise over numpy arrays.

Usage::

    python tools/plotting.py                          # demo plot
    from tools.plotting import plot_function
    plot_function("sin(x) + 0.5*cos(2*x)", -10, 10)
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from tools.expr_parser import CalculatorError, ErrorCode, compile_expression
from utils.artifacts import get_artifact_path


# ---------------------------------------------------------------------------
# Namespace — maps parser function / constant names to numpy (or math)
# callables so that ``to_python()`` output compiles to a working lambda.
# ---------------------------------------------------------------------------

_PLOTTING_NAMESPACE: dict[str, object] = {
    # -- trig ----------------------------------------------------------------
    "sin":   np.sin,
    "cos":   np.cos,
    "tan":   np.tan,
    "asin":  np.arcsin,    # parser name → numpy name (different)
    "acos":  np.arccos,
    "atan":  np.arctan,
    "sinh":  np.sinh,
    "cosh":  np.cosh,
    "tanh":  np.tanh,
    "atan2": np.arctan2,

    # -- exponentials / logs -------------------------------------------------
    "exp":   np.exp,
    "log":   np.log,
    "log2":  np.log2,
    "log10": np.log10,

    # -- powers / roots ------------------------------------------------------
    "sqrt":  np.sqrt,
    "cbrt":  np.cbrt,
    "pow":   np.power,
    "hypot": np.hypot,

    # -- rounding / sign -----------------------------------------------------
    "abs":   np.abs,
    "ceil":  np.ceil,
    "floor": np.floor,
    "sign":  np.sign,
    "round": np.round,

    # -- element-wise comparisons (2-arg only for plotting) ------------------
    "max":   np.maximum,
    "min":   np.minimum,

    # -- constants -----------------------------------------------------------
    "pi":    np.pi,
    "e":     np.e,
    "phi":   (1 + np.sqrt(5)) / 2,
    "tau":   np.pi * 2,
    "sqrt2": np.sqrt(2),
    "sqrt3": np.sqrt(3),
    "ln2":   np.log(2),
    "ln10":  np.log(10),
    "log2e": np.log2(np.e),
    "log10e": np.log10(np.e),
    "Inf":   np.inf,
    "inf":   np.inf,

    # -- discrete-math stubs (will error on array inputs, but present for
    #    completeness — caller gets a clear numpy error) ---------------------
    "fact":  math.factorial,
    "nCr":   math.comb,
    "nPr":   math.perm,
    "gcd":   math.gcd,
    "lcm":   math.lcm,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_function(
    expression: str,
    x_min: float = -10,
    x_max: float = 10,
    points: int = 1000,
    output_file: str = "plot.png",
) -> str:
    """Plot *expression* (a function of *x*) and save to *output_file*.

    Uses the same expression syntax as ``tools/calculator.py``:

    - ``^`` for exponentiation (``x^2``, ``2^x``)
    - ``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, …
    - ``pi``, ``e``, ``Inf`` for constants
    - ``abs``, ``sqrt``, ``exp``, ``log``, ``log10``, …

    Args:
        expression: A mathematical expression containing ``x``,
                    e.g. ``"sin(x) + 0.5*cos(2*x)"`` or ``"x^2 - 4*x + 3"``.
        x_min: Left bound of the plot range.
        x_max: Right bound of the plot range.
        points: Number of sample points.
        output_file: Path for the output PNG (parent directories are created).

    Returns:
        The *output_file* path.

    Raises:
        CalculatorError: If the expression cannot be parsed or compiled.
    """
    x = np.linspace(x_min, x_max, points)

    fn = compile_expression(expression, variables=["x"],
                            namespace=_PLOTTING_NAMESPACE)

    try:
        y = fn(x)
    except Exception as e:
        raise CalculatorError(
            ErrorCode.DOMAIN_ERROR,
            f"Error evaluating expression: {expression} --> {e}",
        ) from e

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"y = {expression}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    return output_file


# ---------------------------------------------------------------------------
# LangChain tool adapter
# ---------------------------------------------------------------------------

@tool
def plotting_tool(
    expression: str,
    x_min: float = -10,
    x_max: float = 10,
    config: RunnableConfig = None,
) -> str:
    """Generate a plot of a mathematical expression and return the image path.

    The expression should be a function of ``x``.  Syntax matches the
    calculator tool: ``^`` for power, ``sin``/``cos``/``exp``/``sqrt`` etc.

    Example expressions:
        ``"sin(x)"``
        ``"x^2 - 4*x + 3"``
        ``"exp(-x^2)"``

    The image is written into the run's artifact directory with a unique
    filename per invocation (plan 4.5), so concurrent plotting steps never
    overwrite each other. ``config`` is injected by LangChain.
    """
    output_file = get_artifact_path(f"plot-{uuid.uuid4().hex[:8]}.png", config)
    return plot_function(
        expression=expression,
        x_min=x_min,
        x_max=x_max,
        output_file=str(output_file),
    )


# ---------------------------------------------------------------------------
# Demo / smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    file_path = plot_function(
        expression="sin(x) + 0.5*cos(2*x)",
        x_min=-10,
        x_max=10,
    )
    print(f"Saved to {file_path}")
