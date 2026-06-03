import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from langchain_core.tools import tool


ALLOWED_NAMES = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}


def plot_function(
    expression: str,
    x_min: float = -10,
    x_max: float = 10,
    points: int = 1000,
    output_file: str = "plot.png",
):
    """
    Plot a mathematical expression containing x.

    Example:
        plot_function("sin(x)")
        plot_function("x**2 - 4*x + 3")
    """

    x = np.linspace(x_min, x_max, points)

    safe_globals = {"__builtins__": {}}
    safe_locals = {"x": x, **ALLOWED_NAMES}

    try:
        y = eval(expression, safe_globals, safe_locals)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

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

@tool
def plotting_tool(
    expression: str,
    x_min: float = -10,
    x_max: float = 10,
) -> str:
    """
    Generate a plot and return the image path.
    """
    return plot_function(
        expression=expression,
        x_min=x_min,
        x_max=x_max,
        output_file="artifacts/plot.png",
    )

if __name__ == "__main__":
    file_path = plot_function(
        expression="sin(x) + 0.5*cos(2*x)",
        x_min=-10,
        x_max=10,
    )
    print(f"Saved to {file_path}")