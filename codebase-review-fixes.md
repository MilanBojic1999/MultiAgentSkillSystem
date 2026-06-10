# Codebase Review — Multi-Agent Pipeline: Findings & Extended Fixes

> **Date:** 2026-06-09  
> **Reviewed against:** `multi-agent-pipeline-skills-guide.md` (LangGraph/LangChain Implementation Plan)  
> **Scope:** All files in `/mnt/f/Deep_Learning_and_Stuff/vibe_code_playground/agent_skills/`

---

## Table of Contents

1. [Critical Bugs](#1-critical-bugs)
2. [Architectural & Functional Issues](#2-architectural--functional-issues)
3. [Gaps vs. Implementation Doc](#3-gaps-vs-implementation-doc)
4. [What's Working Well](#4-whats-working-well)
5. [Extended Fixes with Code](#5-extended-fixes-with-code)
6. [Recommended Implementation Order](#6-recommended-implementation-order)

---

## 1. Critical Bugs

### 1.1 `paralel_pipeline_graph.py` — Broken Import (Line 4)

**Severity:** 🔴 Runtime crash

```python
# Current (broken)
from agents.orchestrator_node import orchestrator_node
```

`orchestrator_node.py` defines `orchestrator_agent`, **not** `orchestrator_node`. This parallel graph will fail with `ImportError` the moment it's imported.

**Root cause:** The doc uses the name `orchestrator_node` throughout Phase 4, but the implementation named the function `orchestrator_agent`. The parallel graph references the doc's name, not the actual export.

---

### 1.2 `agent_mcp_tools.py` — MCP Tools Fetched Outside Session Context (Line 31)

**Severity:** 🔴 Silent failure at tool-call time

```python
# Current (broken)
def create_mcp_client(agent_name: str) -> tuple:
    client = MultiServerMCPClient({name: {"url": url, "transport": "streamable_http"}
                                   for name, url in server_map.items()})
    return client, client.get_tools()   # ❌ Synchronous, outside async with
```

`MultiServerMCPClient` establishes a streaming HTTP transport. `get_tools()` called before `async with client:` enters the session context may return tool stubs that have no active transport. When the LLM invokes a tool, it will silently fail or hang.

**Doc intent (Phase 3.3):**
```python
async def get_mcp_tools_for_agent(agent_name: str) -> list:
    async with client:
        return client.get_tools()   # Inside the context manager
```

The current code tries to split client creation from tool acquisition, but returns both prematurely.

---

### 1.3 `sub_agents_nodes.py` — Double-Close of MCP Client (Lines 88–91)

**Severity:** 🔴 Runtime error or resource leak

```python
# Current (broken)
if mcp_client is not None:
    async with mcp_client:
        result = await agent.ainvoke({"messages": [("user", step["subtask"])]})
    close_mcp_client(mcp_client)   # ❌ Already closed by async with
```

The `async with` context manager calls `__aexit__` which tears down the transport. Calling `close_mcp_client()` (which calls `client.close()`) on an already-closed client will either:
- Raise an exception (cancelling the result)
- Silently corrupt state (if `close()` is not idempotent)

The `close_mcp_client` utility function serves no purpose; the context manager handles cleanup.

---

### 1.4 `utils/logger.py` — `logging.basicConfig` Is a No-Op (Lines 5–6)

**Severity:** 🔴 Logs never written to file

```python
# Current (broken)
logger = logging.getLogger(__name__)          # Creates logger first
logging.basicConfig(filename='langgraph_smart_reasoning.log', level=logging.INFO)  # No-op
```

`logging.basicConfig()` only takes effect if called **before** any logger is created. Since `getLogger(__name__)` precedes it, the `basicConfig` call is silently ignored. No log file is ever created on disk, and all `log_event()` calls write to nowhere.

---

### 1.5 `paralel_pipeline_graph.py` — Missing `Send` Key in `path_map` (Line 51)

**Severity:** 🔴 May not route parallel branches correctly

```python
# Current (incomplete)
builder.add_conditional_edges("orchestrator", fan_out_router,
    path_map={"assemble": "assemble"})

# Doc version (correct)
builder.add_conditional_edges("orchestrator", fan_out_router,
    {"assemble": "assemble", Send: "parallel_sub_agent"})
```

`fan_out_router` returns either the string `"assemble"` **or** a `list[Send]`. Without `Send` in the path map, LangGraph has no explicit instruction for resolving `Send`-returning branches. While some LangGraph versions infer this, it's undocumented and fragile. The same issue exists on line 52 for `parallel_sub_agent`'s conditional edges.

---

## 2. Architectural & Functional Issues

### 2.1 `AGENT_ROSTER` Defined in Two Places

- `orchestrator_node.py:24` — hardcoded dict
- `sub_agents_nodes.py:19` — identical copy

**Impact:** Adding an agent requires editing two files. If they drift out of sync, the orchestrator may plan a step for an agent that the sub-agent node doesn't know how to execute (or worse, executes with the wrong description).

**Doc reference (Phase 10):** The file layout specifies `agents/roster.py` as the single source of truth.

---

### 2.2 `run_sub_agent_async` — Type Annotation Mismatch

**File:** `sub_agents_nodes.py:48`

```python
async def run_sub_agent_async(
    step: dict,
    skill_index: list[dict],           # ❌ Annotated as list[dict]
    skill_dictionary_pairs: dict[str, str],
    results: dict,
) -> tuple[int, str]:
```

But `load_skills()` returns `tuple[dict, dict]` — the first element is a `dict` of `{name: data}`, **not** a `list[dict]`. The code accesses `skill_index.keys()` (line 65), which works on dicts but would crash on a list. Static type checkers (mypy, pyright) will flag this.

---

### 2.3 `asyncio.run()` Inside Sync `sub_agent_node` — Nested Event Loop Risk

**File:** `sub_agents_nodes.py:114`

```python
def sub_agent_node(state: dict) -> dict:
    ...
    step_num, output = asyncio.run(
        run_sub_agent_async(step, skill_index, skill_dictionary_pairs, results)
    )
```

**Problem:** If the compiled graph is ever invoked from an existing event loop (FastAPI, Jupyter, pytest-asyncio, or any async framework), `asyncio.run()` raises:

```
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Fix:** LangGraph fully supports `async` node functions. Make `sub_agent_node` async.

---

### 2.4 Skill Filesystem Rescan on Every Sub-Agent Invocation

**File:** `sub_agents_nodes.py:103`

```python
def sub_agent_node(state: dict) -> dict:
    skill_index, skill_dictionary_pairs = load_skills()   # Re-reads filesystem every call
```

The orchestrator loads skills at module level, but the sub-agent node ignores this and rescans the entire `skills/` directory every time it runs. For a 5-step plan, that's 5 redundant filesystem scans.

**Better approach:** Load skills once at module level in `sub_agents_nodes.py` (mirroring the orchestrator), or pass them through `AgentState`.

---

### 2.5 `agent.py` — Dead Code from Pre-LangGraph Era

**File:** `agent.py`

This defines:
- A plain `AgentState` class (superseded by `agent_states.py`'s `TypedDict`)
- An `Agent` class that iterates `self.skill.steps` (pre-LangGraph pattern)

**Impact:** Confuses maintainers. Both `AgentState` in `agent.py` and `AgentState` in `agent_states.py` exist in the same package; an accidental import of the wrong one produces no type errors but completely breaks graph execution.

---

### 2.6 Parallel Graph Lacks Checkpointing

**File:** `paralel_pipeline_graph.py:55`

```python
graph = builder.compile()   # No checkpointer
```

The sequential graph (`pipeline_graph.py:38`) uses `MemorySaver`. The parallel graph doesn't — so interrupted parallel runs cannot be resumed.

---

### 2.7 Orchestrator LLM Response Parsing Is Brittle

**File:** `orchestrator_node.py:80-84`

```python
try:
    plan = json.loads(response.content)["plan"]
    return {"plan": plan, "results": {}, "current_step": 0}
except Exception as e:
    raise ValueError(f"Failed to parse JSON response: {e}")
```

**Issues:**
- No handling for markdown code fences (`` ```json ... ``` ``) that LLMs frequently emit even when told not to
- If `response.content` has trailing text after JSON, `json.loads` fails
- The error message swallows the actual response content — debugging requires re-running

---

### 2.8 No Environment Variable Validation

Both `orchestrator_node.py` and `sub_agents_nodes.py` call `load_dotenv()` and then read `LLM_URL`, `LLM_MODEL`, `LLM_KEY` without checking if they're set. If `.env` is missing or incomplete, `ChatOpenAI` receives `None` for these parameters, producing cryptic errors like:

```
openai.NotFoundError: 404 page not found
```

---

### 2.9 Typo in Filename: `senitize.py`

**File:** `utils/senitize.py`

Should be `utils/sanitize.py`. The orchestrator imports it as `from utils.senitize import sanitize_content` — this works but will confuse every developer who tries to find "sanitize.py".

---

## 3. Gaps vs. Implementation Doc

| # | Doc Requirement | Status | Detail |
|---|---|---|---|
| 3.1 | Agent roster in `agents/roster.py` | ❌ Missing | Duplicated across 2 files |
| 3.2 | `utils/validators.py` with `validate_step_output` | ❌ Missing | Doc Phase 9.3; no output validation exists |
| 3.3 | `sanitize_content()` called on all external inputs | ⚠️ Partial | Only in orchestrator; missing in sub_agent_node before skill/context injection |
| 3.4 | `SqliteSaver` for production checkpointing | ❌ Missing | Only `MemorySaver`; no `.checkpoints/` directory |
| 3.5 | `tests/` directory with unit + integration tests | ❌ Missing | Doc Phase 10 lists 5 test files; zero exist |
| 3.6 | `pipeline.py` entrypoint (Phase 11) | ❌ Missing | Graphs compile at module level; no run wrapper with `run_pipeline(task)` |
| 3.7 | `langgraph-checkpoint-sqlite` in requirements | ❌ Missing | `requirements.txt` has `langgraph>=1.2` but not the SQLite checkpointer package |
| 3.8 | `RetryPolicy` on orchestrator node | ❌ Missing | Only sub_agent has `retry_policy` (sequential graph) |
| 3.9 | MCP auth tokens loaded from env vars | ⚠️ Unclear | `MCP_ACCESS_AGENT` has hardcoded URLs; no token/env-var injection visible |
| 3.10 | LangSmith tracing configured | ⚠️ Unknown | No `.env` file found; `LANGCHAIN_TRACING_V2` status unconfirmed |
| 3.11 | File layout matches doc Phase 10 | ⚠️ Partially | Core files at package root instead of `graph/`, `state/`, `agents/` subdirectories |

---

## 4. What's Working Well

These components correctly follow the doc and LangGraph best practices:

- **State definition** (`agent_states.py`): The `TypedDict` with `Annotated[dict, lambda a, b: {**a, **b}]` reducer is a precise implementation of Phase 1. Parallel results merge correctly.

- **`@tool` decorators** (`calculator.py`, `plotting.py`): Native tools use `langchain_core.tools.tool`, which auto-generates JSON schemas from docstrings and type hints — exactly what Phase 3.1 specifies.

- **Tool allowlisting** (`agent_tools.py`): `AGENT_TOOLS` dict scopes tools per agent. An agent never receives a tool it isn't listed for — matching the doc's "defence by omission" principle.

- **MCP dedicated access guard** (`agent_mcp_tools.py:23-25`): `DEDICATED_MCP_OWNERS` check raises `ValueError` if a server's owner doesn't match the calling agent. This implements Phase 9.2 faithfully.

- **Graph architecture**: Both sequential and parallel graphs follow the correct pattern: entry → orchestrator → conditional edge → sub-agent loop → assemble → END. The `Send` API fan-out in the parallel variant is the right approach.

- **`create_react_agent`**: The tool-calling loop (thinking → tool_call → result → think → ... → final_answer) is fully delegated to LangGraph's prebuilt — no manual `while stop_reason == "tool_use"` loop.

- **Prompt injection detection** (`utils/senitize.py`): Regex patterns cover classic jailbreaks, "You are now X" role-switching, Markdown exfiltration, and system prompt extraction. Good coverage for regex-based detection.

- **SKILL.md format**: Both `roll-dice` and `frontend-design` skills correctly use `---`-delimited YAML frontmatter with `name` and `description` fields.

- **Structured logging adapter** (`utils/logger.py`): The `log_event()` function emits JSON lines with `event`, `ts`, and arbitrary `**kwargs` — useful for log aggregation even after LangSmith covers core traces.

---

## 5. Extended Fixes with Code

### Fix 1: Correct the Import in `paralel_pipeline_graph.py`

The parallel graph must import the function that actually exists.

```python
# paralel_pipeline_graph.py (line 4)
# BEFORE
from agents.orchestrator_node import orchestrator_node

# AFTER
from agents.orchestrator_node import orchestrator_agent

# Line 46 — use the correct function name
builder.add_node("orchestrator", orchestrator_agent)
```

Additionally, rename the function in `orchestrator_node.py` to match the doc's convention, or update all references consistently. **Recommendation:** keep `orchestrator_agent` since the sequential graph also uses it (via `pipeline_graph.py:5`), and update the doc reference in a follow-up pass.

---

### Fix 2: Fix MCP Client Lifecycle

Rewrite `agent_mcp_tools.py` so tools are acquired inside an active session, not before:

```python
# agent_mcp_tools.py — REWRITTEN
from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_ACCESS_AGENT: dict[str, dict[str, str]] = {
    "researcher": {"yotta_mcp": "http://207.189.105.118:8001/mcp"},
}

DEDICATED_MCP_OWNERS: dict[str, str] = {
    "yotta_mcp": "researcher",
}


def _build_mcp_client(agent_name: str) -> MultiServerMCPClient | None:
    """
    Construct a MultiServerMCPClient scoped to the agent's allowed servers.
    Returns None if the agent has no MCP servers configured.
    The caller MUST use `async with client:` to keep the transport alive.
    """
    server_map = MCP_ACCESS_AGENT.get(agent_name, {})
    if not server_map:
        return None

    # Guard: dedicated servers must only be accessed by their owner
    for server_name in server_map:
        owner = DEDICATED_MCP_OWNERS.get(server_name)
        if owner is not None and owner != agent_name:
            raise ValueError(
                f"Agent '{agent_name}' attempted to access dedicated MCP server "
                f"'{server_name}' which is owned by '{owner}'."
            )

    return MultiServerMCPClient(
        {name: {"url": url, "transport": "streamable_http"}
         for name, url in server_map.items()}
    )
```

Then update `sub_agents_nodes.py` to use the client correctly:

```python
# sub_agents_nodes.py — inside run_sub_agent_async, replace lines 75–91
from agent_mcp_tools import _build_mcp_client

# ... inside run_sub_agent_async ...

native_tools = AGENT_TOOLS.get(agent_name, [])
mcp_client = _build_mcp_client(agent_name)

if mcp_client is not None:
    async with mcp_client:
        mcp_tools = mcp_client.get_tools()           # tools alive inside session
        all_tools = native_tools + mcp_tools
        agent = create_react_agent(
            model=llm, tools=all_tools,
            prompt=SystemMessage(content=system_prompt),
        )
        result = await agent.ainvoke({"messages": [("user", step["subtask"])]})
else:
    agent = create_react_agent(
        model=llm, tools=native_tools,
        prompt=SystemMessage(content=system_prompt),
    )
    result = await agent.ainvoke({"messages": [("user", step["subtask"])]})

output = result["messages"][-1].content
return step_num, output
```

Then **delete** `close_mcp_client` — the context manager handles cleanup.

---

### Fix 3: Fix Logger Configuration

```python
# utils/logger.py — REWRITTEN
import logging
import json
import time

# Configure BEFORE creating any logger
logging.basicConfig(
    filename='langgraph_smart_reasoning.log',
    level=logging.INFO,
    format='%(message)s',   # We emit our own JSON; don't double-format
)

logger = logging.getLogger("multi-agent-pipeline")


def log_event(event: str, **kwargs):
    logger.info(json.dumps({"event": event, "ts": time.time(), **kwargs}))
```

Key changes:
- `basicConfig` called **before** `getLogger`
- Logger name changed from `__name__` to `"multi-agent-pipeline"` — avoids module-path coupling
- Added `format='%(message)s'` so the JSON line isn't wrapped in a logging format string

---

### Fix 4: Extract Shared `AGENT_ROSTER`

Create the file the doc always intended:

```python
# agents/roster.py — NEW FILE
"""
Single source of truth for all available sub-agent definitions.
Import this everywhere an agent roster is needed.
"""

AGENT_ROSTER: dict[str, str] = {
    "mathematician": "Expert in solving complex mathematical problems and plotting functions.",
    "researcher": "Skilled in gathering and synthesizing information from various sources.",
    "writer": "Proficient in crafting clear and engaging written content on a wide range of topics.",
}
```

Then update `orchestrator_node.py` and `sub_agents_nodes.py`:

```python
# In both files, REPLACE the hardcoded dict with:
from agents.roster import AGENT_ROSTER
```

---

### Fix 5: Make `sub_agent_node` Async

```python
# sub_agents_nodes.py — REWRITTEN sub_agent_node
async def sub_agent_node(state: dict) -> dict:
    """
    Sequential node: executes the next uncompleted step in the plan.
    Now async to avoid asyncio.run() nesting issues.
    """
    skill_index, skill_dictionary_pairs = load_skills()

    plan    = state["plan"]
    results = state.get("results", {})

    for step in plan:
        if step["step"] in results:
            continue
        deps_met = all(d in results for d in step.get("depends_on", []))
        if deps_met:
            step_num, output = await run_sub_agent_async(
                step, skill_index, skill_dictionary_pairs, results
            )
            return {"results": {step_num: output}}

    return {}
```

Note: `asyncio.run(...)` is replaced with `await ...`. LangGraph natively supports async node functions — no other changes needed.

---

### Fix 6: Cache Skill Loading

Load skills once at module level (mirroring the orchestrator):

```python
# sub_agents_nodes.py — add near top, after imports
from skill_loader import load_skills

_SKILL_INDEX, _SKILL_DICTIONARY_PAIRS = load_skills()   # module-level singleton

# Then in sub_agent_node, use the module-level cache:
async def sub_agent_node(state: dict) -> dict:
    plan    = state["plan"]
    results = state.get("results", {})

    for step in plan:
        if step["step"] in results:
            continue
        deps_met = all(d in results for d in step.get("depends_on", []))
        if deps_met:
            step_num, output = await run_sub_agent_async(
                step, _SKILL_INDEX, _SKILL_DICTIONARY_PAIRS, results
            )
            return {"results": {step_num: output}}
    return {}
```

---

### Fix 7: Robust JSON Parsing in Orchestrator

Handle markdown code fences and trailing content:

```python
# orchestrator_node.py — replace lines 79-84
import re

def _extract_json(text: str) -> dict:
    """
    Extract JSON from LLM response, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose
    """
    # Try direct parse first (best case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find the outermost JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"Could not extract JSON from response: {text[:500]}...")


def orchestrator_agent(state: dict):
    ...
    response = llm.invoke(messages)
    plan_data = _extract_json(response.content)
    plan = plan_data.get("plan", [])

    if not isinstance(plan, list) or len(plan) == 0:
        raise ValueError(f"Orchestrator produced an empty or invalid plan: {plan_data}")

    return {"plan": plan, "results": {}, "current_step": 0}
```

---

### Fix 8: Add Output Validation

Create the missing validator module:

```python
# utils/validators.py — NEW FILE
"""
Output validation for sub-agent results.
Called before writing results to shared state to prevent
malicious or malformed outputs from propagating.
"""

import re
from typing import Any

MAX_OUTPUT_LENGTH = 50_000     # characters
ALLOWED_PATTERNS = [
    re.compile(r".+"),         # At least one non-whitespace character
]
BLOCKED_PATTERNS = [
    re.compile(r"<script", re.IGNORECASE),         # XSS vectors
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"```system", re.IGNORECASE),       # Prompt leakage
]


def validate_step_output(step_num: int, agent_name: str, output: Any) -> str:
    """
    Validate and sanitize a sub-agent's output before it enters shared state.

    Returns the (possibly sanitized) output string.
    Raises ValueError if the output is irredeemably invalid.
    """
    if not isinstance(output, str):
        raise ValueError(
            f"Step {step_num} ({agent_name}): expected str output, got {type(output).__name__}"
        )

    if not output.strip():
        raise ValueError(
            f"Step {step_num} ({agent_name}): output is empty or whitespace-only"
        )

    if len(output) > MAX_OUTPUT_LENGTH:
        raise ValueError(
            f"Step {step_num} ({agent_name}): output is {len(output)} chars "
            f"(max {MAX_OUTPUT_LENGTH})"
        )

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(output):
            raise ValueError(
                f"Step {step_num} ({agent_name}): output contains blocked pattern "
                f"'{pattern.pattern}'"
            )

    return output
```

Then call it in `sub_agent_node` (and `parallel_sub_agent_node`) before returning results:

```python
# In sub_agent_node and parallel_sub_agent_node, after getting output:
from utils.validators import validate_step_output

output = result["messages"][-1].content
output = validate_step_output(step_num, agent_name, output)
return step_num, output
```

---

### Fix 9: Add `sanitize_content` in Sub-Agent Node

The doc specifies calling sanitization on **all** external inputs. Currently it's only called on the user task in the orchestrator. Add it in the sub-agent node for upstream context:

```python
# sub_agents_nodes.py — inside run_sub_agent_async, before building system prompt
from utils.senitize import sanitize_content

# Sanitize upstream context values before injecting into prompt
sanitized_context = {
    d: sanitize_content(str(results.get(d, "")), f"step-{d}-output")
    for d in step.get("depends_on", [])
}

system_prompt = _build_system_prompt(
    agent_name, AGENT_ROSTER[agent_cfg], skill_bodies, sanitized_context
)
```

---

### Fix 10: Add Checkpointer to Parallel Graph

```python
# paralel_pipeline_graph.py — update compile call
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

Also add `RetryPolicy` to parallel nodes:

```python
from langgraph.types import RetryPolicy

builder.add_node("parallel_sub_agent", parallel_sub_agent_node,
                  retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
```

---

### Fix 11: Don't Use `asyncio.run()` in `paralel_pipeline_graph.py`

```python
# paralel_pipeline_graph.py:33 — BEFORE
step_num, output = await run_sub_agent_async(state["step"], skill_index,
    skill_dictionary_pairs, state["results"])

# This is already async — the `await` is correct. No change needed here,
# but the import of `run_sub_agent_async` from sub_agents_nodes is fine.
```

This is actually correct in the parallel graph (it uses `await`), but the sequential graph's `sub_agent_node` has the `asyncio.run()` problem described in Fix 5.

---

### Fix 12: Create Entrypoint `pipeline.py`

```python
# pipeline.py — NEW FILE
"""
Multi-Agent Pipeline entrypoint.
Usage:
    python pipeline.py "Your task description here"
"""
import sys
import hashlib

from pipeline_graph import graph as sequential_graph
# from paralel_pipeline_graph import graph as parallel_graph  # swap for parallel


def run_pipeline(task: str, use_checkpoint: bool = True) -> str:
    """
    Run the multi-agent pipeline on a given task.

    Args:
        task: The natural-language task to decompose and execute.
        use_checkpoint: If True, re-using the same task string will resume
                        from the last completed node.

    Returns:
        The assembled final output string.
    """
    config = None
    if use_checkpoint:
        run_id = hashlib.sha256(task.encode()).hexdigest()[:16]
        config = {"configurable": {"thread_id": run_id}}

    result = sequential_graph.invoke({"task": task}, config=config)
    return result.get("final_output", "No output produced.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py '<task description>'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    output = run_pipeline(task)
    print("\n=== FINAL OUTPUT ===\n")
    print(output)
```

---

### Fix 13: Rename `senitize.py` → `sanitize.py`

```bash
git mv utils/senitize.py utils/sanitize.py
```

Then update imports in `orchestrator_node.py`:

```python
# orchestrator_node.py line 8 — BEFORE
from utils.senitize import sanitize_content

# AFTER
from utils.sanitize import sanitize_content
```

---

### Fix 14: Remove or Isolate Dead `agent.py`

Option A — delete it if truly unused:
```bash
git rm agent.py
```

Option B — move to a deprecated location if it's reference material:
```bash
mkdir -p _archive
git mv agent.py _archive/agent_pre_langgraph.py
```

Option C — add a clear deprecation warning at the top:
```python
# agent.py — add as first lines
import warnings
warnings.warn(
    "agent.py is deprecated. Use agent_states.py (TypedDict) + pipeline_graph.py (LangGraph) instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

---

### Fix 15: Fix Type Annotation in `run_sub_agent_async`

```python
# sub_agents_nodes.py — update function signature
async def run_sub_agent_async(
    step: dict,
    skill_index: dict,                          # FIXED: was list[dict]
    skill_dictionary_pairs: dict[str, str],
    results: dict[int, str],
) -> tuple[int, str]:
```

And update the skill lookup loop for clarity:

```python
# sub_agents_nodes.py lines 60-65 — BEFORE
requested = step.get("skills_needed", [])
skill_bodies = [
    load_skills_body(skill_dictionary_pairs, s_name)
    for skill_name in requested
    for s_name in skill_index.keys() if s_name == skill_name
]

# AFTER — clearer and O(n) instead of O(n*m)
requested = step.get("skills_needed", [])
skill_bodies = []
for skill_name in requested:
    if skill_name in skill_index:
        skill_bodies.append(
            load_skills_body(skill_dictionary_pairs, skill_name)
        )
```

---

### Fix 16: Add Environment Variable Validation

```python
# utils/env.py — NEW FILE
"""
Centralized environment variable access with validation.
"""
import os
from dotenv import load_dotenv

load_dotenv()

_REQUIRED_VARS = ["LLM_URL", "LLM_MODEL", "LLM_KEY"]


def get_env(var: str) -> str:
    """Get a required environment variable. Raises if unset."""
    value = os.getenv(var)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{var}' is not set. "
            f"Add it to your .env file."
        )
    return value


def get_llm_config() -> dict:
    """Return the LLM configuration dict for ChatOpenAI / ChatAnthropic."""
    missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Check your .env file."
        )
    return {
        "model": os.getenv("LLM_MODEL"),
        "openai_api_key": os.getenv("LLM_KEY"),
        "openai_api_base": os.getenv("LLM_URL"),
    }
```

Then in both `orchestrator_node.py` and `sub_agents_nodes.py`:

```python
# REPLACE the manual load_dotenv + os.getenv pattern with:
from utils.env import get_llm_config

llm = ChatOpenAI(**get_llm_config(), max_tokens=4048, temperature=0.9)
```

---

### Fix 17: Add `langgraph-checkpoint-sqlite` to Requirements

```diff
# requirements.txt
 langgraph>=1.2
+langgraph-checkpoint-sqlite>=2.0
 openai>=1.78.0
```

---

### Fix 18: Add Explicit Send Path Map to Parallel Graph

```python
# paralel_pipeline_graph.py — replace lines 51-53
from langgraph.types import Send

builder.add_conditional_edges(
    "orchestrator",
    fan_out_router,
    path_map={
        "assemble": "assemble",
        Send: "parallel_sub_agent",
    },
)
builder.add_conditional_edges(
    "parallel_sub_agent",
    fan_out_router,
    path_map={
        "assemble": "assemble",
        Send: "parallel_sub_agent",
    },
)
```

---

## 6. Recommended Implementation Order

Work through these in order — each group is independent of the next, but within a group, order matters because later fixes build on earlier ones.

### Group A: Crash Prevention (do first)

| Order | Fix | Risk |
|-------|-----|------|
| A1 | Fix 1 — Import in `paralel_pipeline_graph.py` | Prevents import crash |
| A2 | Fix 15 — Type annotation in `run_sub_agent_async` | Prevents static analysis errors |
| A3 | Fix 16 — Environment variable validation | Prevents cryptic LLM errors |
| A4 | Fix 3 — Logger `basicConfig` ordering | Makes logs actually work |

### Group B: Correctness

| Order | Fix | Risk |
|-------|-----|------|
| B1 | Fix 2 — MCP client lifecycle | Prevents silent tool failures |
| B2 | Fix 5 — Make `sub_agent_node` async | Prevents `asyncio.run()` crash in async contexts |
| B3 | Fix 18 — Explicit Send path_map | Ensures parallel routing behaves deterministically |

### Group C: Architecture & Hygiene

| Order | Fix | Risk |
|-------|-----|------|
| C1 | Fix 4 — Extract `AGENT_ROSTER` to `agents/roster.py` | Single source of truth |
| C2 | Fix 6 — Cache skill loading at module level | Performance |
| C3 | Fix 7 — Robust orchestrator JSON parsing | Resilience to LLM formatting quirks |
| C4 | Fix 13 — Rename `senitize.py` → `sanitize.py` | Developer experience |
| C5 | Fix 14 — Remove/archive dead `agent.py` | Clarity |

### Group D: Security & Observability

| Order | Fix | Risk |
|-------|-----|------|
| D1 | Fix 8 — Create `utils/validators.py` and wire `validate_step_output` | Security |
| D2 | Fix 9 — Add `sanitize_content` in sub-agent node | Security |
| D3 | Fix 10 — Add checkpointer + RetryPolicy to parallel graph | Reliability |

### Group E: Production Readiness

| Order | Fix | Risk |
|-------|-----|------|
| E1 | Fix 12 — Create `pipeline.py` entrypoint | Usability |
| E2 | Fix 17 — Add `langgraph-checkpoint-sqlite` to requirements | Production dep |
| E3 | Write tests for state reducer, orchestrator parsing, MCP isolation | Confidence |

---

## Appendix: Post-Fix File Layout

After applying all fixes, the project structure should look like:

```
agent_skills/
├── pipeline.py                         # Entrypoint (NEW)
│
├── agent_states.py                     # AgentState TypedDict + PlanStep
├── agent_tools.py                      # AGENT_TOOLS allowlist dict
├── agent_mcp_tools.py                  # MCP client builder (REWRITTEN)
├── skill_loader.py                     # Skill discovery + activation
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator_node.py            # Orchestrator (IMPORT FIXED)
│   ├── sub_agents_nodes.py             # Sub-agent node (ASYNC FIX)
│   └── roster.py                       # AGENT_ROSTER (NEW — extracted)
│
├── tools/
│   ├── __init__.py
│   ├── calculator.py                   # @tool — math expression parser
│   ├── plotting.py                     # @tool — matplotlib plotter
│   └── bash_tool.py                    # Bash execution tool
│
├── skills/
│   ├── frontend-design/SKILL.md
│   └── roll-dice/SKILL.md
│
├── utils/
│   ├── logger.py                       # (FIXED — basicConfig ordering)
│   ├── sanitize.py                     # (RENAMED from senitize.py)
│   ├── validators.py                   # (NEW)
│   └── env.py                          # (NEW)
│
├── pipeline_graph.py                   # Sequential graph (with MemorySaver)
├── paralel_pipeline_graph.py           # Parallel graph (FIXED imports + checkpointer)
│
├── tests/                              # (NEW — to be populated)
│   ├── test_state.py
│   ├── test_orchestrator.py
│   └── test_mcp_access.py
│
├── .checkpoints/                       # (NEW — gitignored)
├── .env                                # (ensure exists with all required vars)
├── requirements.txt                    # (UPDATED — added sqlite checkpointer)
└── _archive/                           # (NEW — if keeping old agent.py)
    └── agent_pre_langgraph.py
```

---

*Review and extended fixes document generated June 2026 · Based on diff between `multi-agent-pipeline-skills-guide.md` and the actual codebase at `/mnt/f/Deep_Learning_and_Stuff/vibe_code_playground/agent_skills/`*
