"""
Simple entrypoint to run the LangGraph multi-agent pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py "Calculate sin(pi/4) + cos(pi/4) and plot both functions"
"""

import sys
from yotta_graph import graph
from agent_states import get_current_datetime_str
import asyncio

def run(task: str) -> str:
    # thread_id groups checkpoints for this run; use a fixed one for dev/test
    config = {"configurable": {"thread_id": "test-run-1"}}
    result = graph.invoke({"task": task, "current_datetime": get_current_datetime_str()}, config=config)
    return result.get("final_output", "No final output produced.")

async def run_async(task: str) -> str:
    # thread_id groups checkpoints for this run; use a fixed one for dev/test
    config = {"configurable": {"thread_id": "test-run-1"}}
    result = await graph.ainvoke({"task": task, "current_datetime": get_current_datetime_str()}, config=config)
    return result.get("final_output", "No final output produced.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        # task = (
        #     "Maximum of (x**2)*sin(x) in the range between 0 and 2 "
        #     "Then write a short summary of what the calculation means."
        # )
        task = """How old is Donald Trump, and what are some key events in his life?""".strip()

    print(f"Running pipeline with task:\n  {task}\n")
    print("=" * 60)
    output = asyncio.run(run_async(task))
    print("=" * 60)
    print(output)
