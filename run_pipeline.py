"""
Simple entrypoint to run the LangGraph multi-agent pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py "Calculate sin(pi/4) + cos(pi/4) and plot both functions"
"""

import sys
from pipeline_graph import graph


def run(task: str) -> str:
    # thread_id groups checkpoints for this run; use a fixed one for dev/test
    config = {"configurable": {"thread_id": "test-run-1"}}
    result = graph.invoke({"task": task}, config=config)
    return result.get("final_output", "No final output produced.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = (
            "Calculate sin(pi/4) + cos(pi/4) and explain the result. "
            "Then write a short summary of what the calculation means."
        )

    print(f"Running pipeline with task:\n  {task}\n")
    print("=" * 60)
    output = run(task)
    print("=" * 60)
    print(output)
