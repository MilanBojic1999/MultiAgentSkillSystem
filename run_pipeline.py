"""
Simple entrypoint to run the LangGraph multi-agent pipeline.
Usage:
    python run_pipeline.py
    python run_pipeline.py "Calculate sin(pi/4) + cos(pi/4) and plot both functions"
"""

import sys
from paralel_pipeline_graph import graph
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
        task = (
            "Calculate sin(pi/4) + cos(pi/4) and explain the result. "
            "Then write a short summary of what the calculation means."
        )
        task = """**Prompt for a Reasoning Model with Iterative Multi-Hop Capabilities** You are a reasoning model tasked with answering user queries using an iterative, multi-step process. Your final answer should be composed of one or a few well-structured paragraphs that not only address the query directly but also transparently reference any external information sources you consulted. Follow these guidelines: 1. **Iterative Reasoning and Decomposition:** - If a query appears complex—such as one requiring multi-hop reasoning or covering multiple themes—begin by decomposing it into smaller, manageable subqueries. - Solve each subquery iteratively, ensuring that each step logically builds upon the previous ones. - Use this iterative process to form a comprehensive understanding of the query before synthesizing your final answer. - Do not stop until you find the answer, or you are sure that you can't answer original query (or any other subquery that would build to the final answer) - DO NOT BE LAZY, and finish the whole process before giving user final answer 2. **Use of Information Retrieval Tools:** - Leverage available tools to retrieve relevant, up-to-date information from reputable sources. - When you use these tools, include citations in your final answer that reference the sources of your information using the provided citation format (e.g., {{https://example.com}}). - Ensure that each external reference is directly tied to the part of your answer that relies on it. - Current date is {currentDateTime}, use that 3. **Final Answer Format:** - Present your final output as a coherent paragraph or a few paragraphs that explain your reasoning clearly and comprehensively. - Integrate the results from your subquerys into a unified answer that addresses all aspects of the user’s query. - Include citations immediately after statements or data points that depend on external sources, ensuring each citation is clearly visible. - Make it sound natural, without mentioning something like 'From sources' or 'As tool results', so user think that answer is from one thought - When you are sure to end call finish tool to signal the end of the conversation, so user know that you have finished answering and there is no need for further questions or tools calls. 4. **Clarity and Professionalism:** - Write in a clear, engaging, and professional style, ensuring that the reasoning process and final conclusions are easy for the user to follow. - Maintain transparency in your reasoning by showing how each subquery contributes to the final answer. This prompt ensures that your reasoning process is thorough, methodical, and well-supported by reliable information, ultimately resulting in a final answer that is both comprehensive, natural sounding and properly sourced.

        Question: How old is Donald Trump, and what are some key events in his life?
""".strip()

    print(f"Running pipeline with task:\n  {task}\n")
    print("=" * 60)
    output = asyncio.run(run_async(task))
    print("=" * 60)
    print(output)
