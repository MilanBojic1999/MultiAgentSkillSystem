"""Sequential pipeline: one step at a time, in dependency order.

Registered automatically as the graph named ``"sequential"`` (see
``graphs/__init__.py``) — the module name minus its ``_pipeline_graph`` suffix.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agents.agent_states import AgentState
from agents.orchestrator_node import make_orchestrator_agent
from agents.sub_agents_nodes import make_sub_agent_node
from assemble_node import assemble_node

GRAPH_DESCRIPTION = "Run one step at a time in dependency order (reference topology)"


def should_continue(state: dict) -> str:
    plan = state.get("plan", [])
    results = state.get("results", {})
    if len(results) < len(plan):
        return "sub_agent"
    return "assemble"


def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
    """Compile the sequential pipeline.

    Every argument defaults to the production wiring, so ``build()`` takes no
    arguments in production; tests inject fake nodes and their own checkpointer.
    """
    orchestrator = orchestrator or make_orchestrator_agent()
    sub_agent = sub_agent or make_sub_agent_node()

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", orchestrator)
    builder.add_node("sub_agent", sub_agent, retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
    builder.add_node("assemble", assemble_node)

    builder.set_entry_point("orchestrator")
    builder.add_conditional_edges("orchestrator", should_continue, ["sub_agent", "assemble"])
    builder.add_conditional_edges("sub_agent", should_continue, ["sub_agent", "assemble"])
    builder.add_edge("assemble", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
