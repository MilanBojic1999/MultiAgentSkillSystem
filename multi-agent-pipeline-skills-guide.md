# Multi-Agent Pipeline with SKILL.md — LangGraph/LangChain Implementation Plan

> Mapping your existing architecture to LangGraph/LangChain primitives, step by step.
> Your original plan's concepts are preserved; this document shows exactly how each
> piece translates and where LangGraph gives you more for less code.

---

## How Your Plan Maps to LangGraph

Before diving into steps, here is the conceptual translation table so you always know
what "fits where":

| Your Plan Concept | LangGraph Equivalent |
|---|---|
| `pipeline.py` execution loop | `StateGraph` compiled to a `CompiledGraph` |
| `AgentState` / `CheckpointStore` | `TypedDict` state + built-in `MemorySaver` / `SqliteSaver` checkpointer |
| Orchestrator Agent | A graph node that calls an LLM and writes a `plan` to shared state |
| Sub-Agents | Individual graph nodes (or nested sub-graphs) |
| `depends_on` in the plan | LangGraph `conditional_edges` and `Send` API for fan-out |
| `ToolBroker` (native tools) | `@tool`-decorated functions bound to agents via `create_react_agent` |
| `ToolBroker` (MCP tools) | `langchain-mcp-adapters` — wraps MCP servers as `BaseTool` lists |
| Skill Discovery (`load_skill_index`) | A standalone utility function unchanged — injected into agent system prompts |
| Skill Activation (`activate_skill`) | Injected into each node's `ChatPromptTemplate` system message |
| `MCPServerConfig` + `allowed_agents` | Per-agent `MultiServerMCPClient` scoped to allowed server URLs |
| Structured logging | LangSmith tracing (zero-config) + custom `on_*` callbacks |
| Retry / backoff | `@retry` decorator or LangGraph's built-in node-level retry policy |
| Parallel fan-out | LangGraph `Send` API — dispatch N tasks to the same node concurrently |

---

## Phase 0 — Environment Setup

Install the LangGraph/LangChain stack alongside your existing dependencies.

```bash
pip install \
  langgraph>=0.2 \
  langchain>=0.3 \
  langchain-anthropic>=0.3 \
  langchain-mcp-adapters>=0.1 \
  langgraph-checkpoint-sqlite \
  langsmith \
  pyyaml \
  python-dotenv
```

Set environment variables:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
LANGCHAIN_TRACING_V2=true          # enables LangSmith — free tier is fine
LANGCHAIN_API_KEY=ls__...          # from smith.langchain.com
LANGCHAIN_PROJECT=multi-agent-pipeline
```

Your existing `requirements.txt` entries (`pyyaml`, `python-dotenv`, `pytest`,
`httpx`) are still valid — just append the new ones above.

---

## Phase 1 — Define Shared State

In LangGraph, all agents communicate through a single typed state object that flows
through the graph. This replaces your `results: dict[int, str]` dictionary and the
separate `CheckpointStore`.

```python
# state/graph_state.py
from typing import Annotated
from typing_extensions import TypedDict
import operator

class PlanStep(TypedDict):
    step: int
    subtask: str
    agent: str
    skills_needed: list[str]
    depends_on: list[int]

class AgentState(TypedDict):
    # Inputs
    task: str

    # Set by Orchestrator node
    plan: list[PlanStep]

    # Accumulated by sub-agent nodes; reducer merges dicts
    results: Annotated[dict[int, str], lambda a, b: {**a, **b}]

    # Which step is currently executing (used by router)
    current_step: int

    # Final assembled output
    final_output: str
```

The `Annotated[dict, lambda a, b: {**a, **b}]` reducer lets parallel sub-agent nodes
write their results without overwriting each other — LangGraph merges them automatically.

---

## Phase 2 — Skill Loader (Unchanged)

Your `agents/skill_loader.py` is framework-agnostic. Keep it exactly as-is. It produces
plain Python dicts and strings, which you inject into LangChain prompt templates.

```python
# agents/skill_loader.py — NO CHANGES NEEDED
from pathlib import Path
import yaml

def load_skill_index(skills_dir: str) -> list[dict]:
    """Discovery phase — name + description only."""
    index = []
    for skill_dir in Path(skills_dir).iterdir():
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue
        text = skill_file.read_text()
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1])
            index.append({
                "name":        meta.get("name", skill_dir.name),
                "description": meta.get("description", ""),
                "path":        str(skill_file),
            })
    return index

def activate_skill(skill_path: str) -> str:
    """Activation phase — full SKILL.md body."""
    return Path(skill_path).read_text()
```

Multi-source loading (Appendix F in your guide) also stays unchanged.

---

## Phase 3 — Tool Broker → LangChain Tools + MCP Adapters

This is the biggest structural change. Your `ToolBroker` splits into two LangChain
concepts:

**Native tools** → `@tool`-decorated Python functions, each returning a string or dict.  
**MCP tools** → `MultiServerMCPClient` from `langchain-mcp-adapters`, scoped per agent.

### 3.1 Native Tools

```python
# tools/native_tools.py
from langchain_core.tools import tool

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information."""
    # your existing web_search_handler logic here
    ...

@tool
def code_exec(code: str) -> str:
    """Execute Python code and return stdout/stderr."""
    # your existing code_exec_handler logic here
    ...

@tool
def db_query(sql: str) -> str:
    """Run a read-only SQL query and return results as JSON."""
    ...
```

These replace `ToolDefinition(source="native", handler=...)`. The `@tool` decorator
automatically generates the JSON schema that LangChain passes to the LLM — no manual
`parameters` dict needed.

### 3.2 Per-Agent Tool Allowlists

Reproduce your `allowed_agents` access control by simply binding only the permitted
tools to each agent node:

```python
# tools/agent_tools.py
from .native_tools import web_search, code_exec, db_query

# Mirrors your ToolRegistry.list_for_agent() logic
AGENT_NATIVE_TOOLS: dict[str, list] = {
    "researcher": [web_search],
    "analyst":    [web_search, code_exec],
    "writer":     [web_search],
    "db-writer":  [web_search],
    "reviewer":   [web_search, code_exec],
}
```

An agent that isn't in a tool's list simply never receives it — no `PermissionError`
needed at call time because the tool is never offered.

### 3.3 MCP Tools (Shared and Dedicated)

`langchain-mcp-adapters` turns MCP servers into standard LangChain `BaseTool` lists.
Use it inside an `async context manager` at agent call time to scope servers to the
calling agent:

```python
# tools/mcp_tools.py
from langchain_mcp_adapters.client import MultiServerMCPClient

# Mirrors your MCPServerConfig + allowed_agents matrix
MCP_ACCESS_MATRIX: dict[str, dict[str, str]] = {
    "researcher": {
        "google-drive-mcp":    "https://drivemcp.googleapis.com/mcp/v1",
        "gmail-mcp":           "https://gmailmcp.googleapis.com/mcp/v1",
        "google-calendar-mcp": "https://calendarmcp.googleapis.com/mcp/v1",
    },
    "analyst": {
        "google-drive-mcp":    "https://drivemcp.googleapis.com/mcp/v1",
        "google-calendar-mcp": "https://calendarmcp.googleapis.com/mcp/v1",
    },
    "writer": {
        "google-drive-mcp":    "https://drivemcp.googleapis.com/mcp/v1",
        "gmail-mcp":           "https://gmailmcp.googleapis.com/mcp/v1",
        "google-calendar-mcp": "https://calendarmcp.googleapis.com/mcp/v1",
    },
    "db-writer": {
        # Dedicated MCP — only db-writer appears here
        "write-db-mcp": "https://internal-db.example.com/mcp",
    },
    "reviewer": {},
}

async def get_mcp_tools_for_agent(agent_name: str) -> list:
    """
    Returns activated MCP tools scoped to this agent.
    Call inside an async context and pass the tools to create_react_agent.
    """
    server_map = MCP_ACCESS_MATRIX.get(agent_name, {})
    if not server_map:
        return []

    client = MultiServerMCPClient(
        {name: {"url": url, "transport": "streamable_http"}
         for name, url in server_map.items()}
    )
    async with client:
        return client.get_tools()
```

The dedicated-MCP pattern is enforced by exclusion: `write-db-mcp` only appears in
`db-writer`'s row of `MCP_ACCESS_MATRIX`. If another agent were misconfigured to call
it, the MCP server would reject unauthenticated requests anyway — defence in depth.

---

## Phase 4 — Orchestrator Node

The Orchestrator becomes a LangGraph node. It receives the initial task from state,
calls the LLM with the skill index and agent roster in its system prompt, parses the
JSON plan, and writes it back to `AgentState.plan`.

```python
# agents/orchestrator_node.py
import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from agents.skill_loader import load_skill_index
from agents.roster import AGENT_ROSTER

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1000)

SKILL_INDEX = load_skill_index("./skills")   # loaded once at startup

ORCHESTRATOR_SYSTEM = """
You are the Orchestrator in a multi-agent pipeline.

## Your role
1. Analyse the user's task.
2. Decompose it into ordered subtasks.
3. For each subtask, select the best specialist sub-agent from the roster below.
4. Output a JSON plan in the exact format shown.
5. Do NOT execute any subtask yourself.

## Available sub-agents
{agent_roster}

## Available skills (name → description)
{skill_index}

## Output format (JSON only — no prose, no markdown fences)
{{
  "plan": [
    {{
      "step": 1,
      "subtask": "<concise description>",
      "agent": "<agent_name>",
      "skills_needed": ["<skill-name>"],
      "depends_on": []
    }}
  ]
}}
"""

def orchestrator_node(state: dict) -> dict:
    skill_summary  = "\n".join(f"- {s['name']}: {s['description']}" for s in SKILL_INDEX)
    roster_summary = "\n".join(f"- {a['name']}: {a['description']}" for a in AGENT_ROSTER)

    system = ORCHESTRATOR_SYSTEM.format(
        skill_index=skill_summary,
        agent_roster=roster_summary,
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["task"]),
    ])

    plan_data = json.loads(response.content)
    return {"plan": plan_data["plan"], "results": {}, "current_step": 0}
```

---

## Phase 5 — Sub-Agent Nodes

Each sub-agent is a `create_react_agent` wrapped inside a LangGraph node function.
`create_react_agent` gives you the tool-calling loop (your `while stop_reason == "tool_use"`)
for free.

```python
# agents/sub_agent_node.py
import asyncio
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from agents.skill_loader import activate_skill
from agents.roster import AGENT_ROSTER
from tools.agent_tools import AGENT_NATIVE_TOOLS
from tools.mcp_tools import get_mcp_tools_for_agent, MCP_ACCESS_MATRIX

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=1000)

def _build_system_prompt(agent_name: str, agent_description: str,
                          skill_bodies: list[str], context: dict) -> str:
    skill_block   = "\n\n---\n\n".join(skill_bodies)
    context_block = f"\n\n## Upstream context\n{context}" if context else ""
    return f"""You are the {agent_name} specialist agent.
Role: {agent_description}

## Active skills
{skill_block}
{context_block}

Use tools when needed. Return your final answer as plain text. No meta-commentary."""


async def run_sub_agent_async(
    step: dict,
    skill_index: list[dict],
    results: dict,
) -> tuple[int, str]:
    """Run one sub-agent step. Returns (step_number, output_text)."""
    agent_name   = step["agent"]
    agent_cfg    = next(a for a in AGENT_ROSTER if a["name"] == agent_name)
    step_num     = step["step"]

    # Activate only the skills this step needs
    requested   = step.get("skills_needed", agent_cfg["skills"])
    skill_bodies = [
        activate_skill(s["path"])
        for skill_name in requested
        for s in skill_index if s["name"] == skill_name
    ]

    # Gather upstream context from completed dependency steps
    context = {d: results.get(d, "") for d in step.get("depends_on", [])}

    system_prompt = _build_system_prompt(
        agent_name, agent_cfg["description"], skill_bodies, context
    )

    # Combine native tools + MCP tools for this agent
    native_tools = AGENT_NATIVE_TOOLS.get(agent_name, [])
    mcp_tools    = await get_mcp_tools_for_agent(agent_name)
    all_tools    = native_tools + mcp_tools

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        state_modifier=SystemMessage(content=system_prompt),
    )

    result = await agent.ainvoke({"messages": [("user", step["subtask"])]})
    output = result["messages"][-1].content
    return step_num, output


def sub_agent_node(state: dict) -> dict:
    """
    Sequential node: executes the next uncompleted step in the plan.
    For parallel fan-out, see Phase 6 (Send API).
    """
    from agents.skill_loader import load_skill_index
    skill_index = load_skill_index("./skills")

    plan    = state["plan"]
    results = state.get("results", {})

    # Find the next step whose dependencies are all resolved
    for step in plan:
        if step["step"] in results:
            continue
        deps_met = all(d in results for d in step.get("depends_on", []))
        if deps_met:
            step_num, output = asyncio.run(
                run_sub_agent_async(step, skill_index, results)
            )
            return {"results": {step_num: output}}

    return {}
```

---

## Phase 6 — Wire the Graph (Sequential + Parallel)

### 6.1 Sequential Graph

```python
# graph/pipeline_graph.py
from langgraph.graph import StateGraph, END
from state.graph_state import AgentState
from agents.orchestrator_node import orchestrator_node
from agents.sub_agent_node import sub_agent_node

def should_continue(state: dict) -> str:
    """Route to sub_agent if steps remain, else to assembler."""
    plan    = state.get("plan", [])
    results = state.get("results", {})
    if len(results) < len(plan):
        return "sub_agent"
    return "assemble"

def assemble_node(state: dict) -> dict:
    """Combine all step results into the final output."""
    plan    = state["plan"]
    results = state["results"]
    parts   = [f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
               for s in plan]
    return {"final_output": "\n\n".join(parts)}

# Build graph
builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_node)
builder.add_node("sub_agent",    sub_agent_node)
builder.add_node("assemble",     assemble_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator", should_continue)
builder.add_conditional_edges("sub_agent",    should_continue)
builder.add_edge("assemble", END)

graph = builder.compile()
```

### 6.2 Parallel Fan-Out (LangGraph `Send` API)

This replaces your `ThreadPoolExecutor` fan-out. LangGraph's `Send` API dispatches
multiple sub-agent calls to the same node concurrently and merges their results through
the state reducer automatically.

```python
# graph/pipeline_graph_parallel.py
from langgraph.types import Send
from langgraph.graph import StateGraph, END
from state.graph_state import AgentState
from agents.orchestrator_node import orchestrator_node
from agents.sub_agent_node import run_sub_agent_async
from agents.skill_loader import load_skill_index

SKILL_INDEX = load_skill_index("./skills")

def fan_out_router(state: dict):
    """
    After orchestration, dispatch ALL independent steps in parallel via Send.
    Steps with depends_on=[1] wait until step 1 is in results (handled by
    the dependency layer grouping below).
    """
    plan    = state["plan"]
    results = state.get("results", {})

    # Find all steps whose dependencies are satisfied
    ready = [
        s for s in plan
        if s["step"] not in results
        and all(d in results for d in s.get("depends_on", []))
    ]

    if not ready:
        return "assemble"

    # Send each ready step to the sub_agent_node in parallel
    return [Send("parallel_sub_agent", {"step": s, "results": results}) for s in ready]

async def parallel_sub_agent_node(state: dict) -> dict:
    step_num, output = await run_sub_agent_async(
        state["step"], SKILL_INDEX, state["results"]
    )
    return {"results": {step_num: output}}

# Build parallel graph
builder = StateGraph(AgentState)
builder.add_node("orchestrator",         orchestrator_node)
builder.add_node("parallel_sub_agent",   parallel_sub_agent_node)
builder.add_node("assemble",             assemble_node)

builder.set_entry_point("orchestrator")
builder.add_conditional_edges("orchestrator",       fan_out_router,
                               {"assemble": "assemble", Send: "parallel_sub_agent"})
builder.add_conditional_edges("parallel_sub_agent", fan_out_router,
                               {"assemble": "assemble", Send: "parallel_sub_agent"})
builder.add_edge("assemble", END)

parallel_graph = builder.compile()
```

The `Send` API handles what you previously wired with `ThreadPoolExecutor` and
`as_completed`. LangGraph also tracks which branches are in-flight, so the graph only
moves to `assemble` once every `Send` branch has completed.

---

## Phase 7 — Checkpointing (Built-in)

Replace your hand-rolled `CheckpointStore` with LangGraph's built-in checkpointer.
Every node transition is automatically persisted. Interrupted runs resume from the
last completed node — no code change to the nodes themselves.

```python
# graph/pipeline_graph.py (updated compile call)
from langgraph.checkpoint.sqlite import SqliteSaver
import hashlib

def build_graph_with_checkpointing(task: str):
    memory = SqliteSaver.from_conn_string("./.checkpoints/pipeline.db")
    graph  = builder.compile(checkpointer=memory)
    run_id = hashlib.sha256(task.encode()).hexdigest()[:16]
    config = {"configurable": {"thread_id": run_id}}
    return graph, config

# Usage
graph, config = build_graph_with_checkpointing(task)
result = graph.invoke({"task": task}, config=config)
```

To resume after a crash, call `graph.invoke` again with the same `config`. LangGraph
detects the existing checkpoint and skips already-completed nodes — identical to your
`store.is_complete(step_num)` guard, but automatic.

For in-memory-only (dev / testing):

```python
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
```

---

## Phase 8 — Observability

### 8.1 LangSmith (Zero-Config Tracing)

With `LANGCHAIN_TRACING_V2=true` set in your environment, every graph run, node
invocation, LLM call, and tool call is automatically traced in LangSmith. This
replaces your structured `log_event` calls for the core pipeline events.

For custom business events, keep your logger alongside:

```python
# utils/logger.py — unchanged from your original
import logging, json, time
logger = logging.getLogger("multi-agent-pipeline")

def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": time.time(), **kwargs}))
```

Call it inside nodes for domain-specific events:

```python
log_event("skill_activated", agent=agent_name, skills=requested)
log_event("mcp_access",      agent=agent_name, servers=list(MCP_ACCESS_MATRIX.get(agent_name, {}).keys()))
```

### 8.2 Node-Level Retry Policy

LangGraph supports per-node retry policies, replacing your `with_retry` wrapper:

```python
from langgraph.pregel import RetryPolicy

builder.add_node(
    "sub_agent",
    sub_agent_node,
    retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
)
```

---

## Phase 9 — Security (Unchanged Concepts, New Integration Points)

### 9.1 Prompt Injection

Keep `utils/sanitize.py` exactly as-is. Call it inside `sub_agent_node` before
injecting any external data (web results, uploaded files, DB rows) into the skill
or context blocks.

### 9.2 MCP Access Control

The `MCP_ACCESS_MATRIX` in Phase 3 is your enforcement layer. Because each agent's
`MultiServerMCPClient` is constructed with only its allowed server URLs, an agent
cannot even form a valid connection to a server it isn't listed for.

For the dedicated MCP pattern, add a guard in `get_mcp_tools_for_agent` that raises
explicitly if an unexpected agent name appears:

```python
DEDICATED_MCP_AGENTS = {"write-db-mcp": "db-writer"}

async def get_mcp_tools_for_agent(agent_name: str) -> list:
    server_map = MCP_ACCESS_MATRIX.get(agent_name, {})
    for server_name in server_map:
        if server_name in DEDICATED_MCP_AGENTS:
            assert DEDICATED_MCP_AGENTS[server_name] == agent_name, (
                f"Agent '{agent_name}' attempted to access dedicated MCP '{server_name}'"
            )
    ...
```

### 9.3 Output Validation

Keep `utils/validators.py` as-is, calling `validate_step_output` at the end of
each sub-agent node before writing to state.

---

## Phase 10 — File Layout

```
my-pipeline/
├── pipeline.py                     # Entrypoint: build graph, invoke, print output
│
├── graph/
│   ├── pipeline_graph.py           # Sequential StateGraph + compile
│   └── pipeline_graph_parallel.py  # Parallel variant with Send API
│
├── state/
│   └── graph_state.py              # AgentState TypedDict + reducers
│
├── agents/
│   ├── orchestrator_node.py        # Orchestrator LangGraph node
│   ├── sub_agent_node.py           # Sub-agent node using create_react_agent
│   ├── skill_loader.py             # UNCHANGED — discovery + activation
│   └── roster.py                   # UNCHANGED — AGENT_ROSTER list
│
├── tools/
│   ├── native_tools.py             # @tool-decorated native handlers
│   ├── agent_tools.py              # AGENT_NATIVE_TOOLS dict (allowlists)
│   └── mcp_tools.py                # MCP_ACCESS_MATRIX + get_mcp_tools_for_agent
│
├── skills/                         # UNCHANGED — your SKILL.md folders
│   ├── web-researcher/SKILL.md
│   ├── data-analyst/SKILL.md
│   ├── doc-writer/SKILL.md
│   ├── db-write-ops/SKILL.md
│   └── code-reviewer/SKILL.md
│
├── utils/
│   ├── logger.py                   # UNCHANGED
│   ├── sanitize.py                 # UNCHANGED
│   └── validators.py               # UNCHANGED
│
├── tests/
│   ├── test_pipeline.py            # Integration tests (same assertions)
│   ├── test_mcp_access.py          # Verify dedicated vs shared MCP isolation
│   └── test_skill_activation.py    # Skill trigger accuracy tests (unchanged)
│
├── .checkpoints/                   # SQLite DB — gitignored
├── .env
├── requirements.txt
└── Dockerfile
```

Files that survive unchanged from your original plan: `skill_loader.py`, `roster.py`,
all `skills/` content, `logger.py`, `sanitize.py`, `validators.py`, and all skill
test YAML files.

---

## Phase 11 — Entrypoint

```python
# pipeline.py
from graph.pipeline_graph import build_graph_with_checkpointing

def run_pipeline(task: str) -> str:
    graph, config = build_graph_with_checkpointing(task)
    result = graph.invoke({"task": task}, config=config)
    return result["final_output"]

if __name__ == "__main__":
    output = run_pipeline(
        "Read the Q3 report from Google Drive, search for competitor pricing, "
        "and write a 300-word executive briefing."
    )
    print("\n=== FINAL OUTPUT ===\n")
    print(output)
```

For the parallel variant, swap `pipeline_graph` for `pipeline_graph_parallel`.

---

## Implementation Order (Recommended)

Work through phases in this sequence to keep each phase testable before building
the next:

1. **Phase 0** — Install deps, set env vars, confirm `LANGCHAIN_TRACING_V2` traces appear in LangSmith.
2. **Phase 1** — Define `AgentState`. Write a unit test that the reducer merges dicts correctly.
3. **Phase 2** — Confirm your existing `skill_loader.py` loads skills correctly with a quick `assert`.
4. **Phase 3** — Implement native `@tool` functions and `MCP_ACCESS_MATRIX`. Write `test_mcp_access.py` to verify isolation.
5. **Phase 4** — Build the Orchestrator node. Test it standalone by calling `orchestrator_node({"task": "..."})` and inspecting the plan.
6. **Phase 5** — Build the sub-agent node for one agent (e.g. `researcher`). Test it standalone.
7. **Phase 6 (sequential)** — Wire the sequential graph. Run the full pipeline with a simple two-step task.
8. **Phase 7** — Add `SqliteSaver`. Kill the process mid-run and verify it resumes.
9. **Phase 8** — Confirm LangSmith shows the full trace. Add node-level retry.
10. **Phase 6 (parallel)** — Upgrade to the `Send` API fan-out variant. Verify parallel steps complete concurrently.
11. **Phases 9–11** — Add security guards, finalize file layout, wire the entrypoint.

---

## Key Things LangGraph Gives You For Free

| Feature | Your original plan | LangGraph version |
|---|---|---|
| State persistence across nodes | Manual `CheckpointStore` with JSON files | `SqliteSaver` / `MemorySaver` — one line at compile time |
| Parallel execution | `ThreadPoolExecutor` + manual depth grouping | `Send` API — declare intent, graph handles concurrency |
| Tool-use loop | Manual `while stop_reason == "tool_use"` loop | `create_react_agent` handles it |
| Retry logic | Manual `with_retry(fn, ...)` wrapper | `RetryPolicy` on each node |
| Observability | Manual `log_event` for every operation | LangSmith auto-traces everything; add `log_event` for business events only |
| Resume on failure | Manual `is_complete` check per step | Graph resumes from last checkpoint automatically |

---

## Summary Checklist (LangGraph Edition)

**State & Graph**
- [x] `AgentState` TypedDict defined with correct `Annotated` reducer on `results`
- [ ] `StateGraph` has `orchestrator`, `sub_agent` (or `parallel_sub_agent`), and `assemble` nodes
- [ ] `conditional_edges` from orchestrator routes to `sub_agent` or `assemble`
- [ ] Graph compiled with `SqliteSaver` checkpointer for production runs

**Skills**
- [ ] `skill_loader.py` unchanged and unit-tested
- [ ] Skill bodies injected into sub-agent `SystemMessage` (not the user turn)
- [ ] All original SKILL.md activation test YAMLs still passing

**Tools & MCP**
- [ ] All native tools decorated with `@tool` and have docstrings (used as descriptions)
- [ ] `AGENT_NATIVE_TOOLS` dict enforces per-agent native tool allowlists
- [ ] `MCP_ACCESS_MATRIX` declares exactly which agents access which servers
- [ ] Dedicated MCP servers appear in exactly one agent's row
- [ ] `DEDICATED_MCP_AGENTS` guard raises on misconfigured access attempts
- [ ] MCP auth tokens loaded from env vars — never hardcoded

**Security**
- [ ] `sanitize_content()` called on all external inputs before prompt injection
- [ ] Output validation runs before writing results to state

**Observability**
- [ ] `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` set
- [ ] LangSmith project configured and traces visible
- [ ] `log_event` retained for domain-specific business events
- [ ] Node-level `RetryPolicy` set on sub-agent and orchestrator nodes

**Testing**
- [ ] Unit test: state reducer merges parallel results correctly
- [ ] Unit test: `orchestrator_node` produces valid plan JSON
- [ ] Unit test: MCP access matrix blocks cross-agent access
- [ ] Integration test: full pipeline runs with a two-step task end-to-end
- [ ] Integration test: interrupted run resumes from checkpoint

---

*Plan generated June 2026 · Based on your `multi-agent-pipeline-skills-guide.md` · Target stack: LangGraph ≥ 0.2, LangChain ≥ 0.3, langchain-anthropic, langchain-mcp-adapters*