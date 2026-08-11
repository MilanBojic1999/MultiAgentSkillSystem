# Multi-Agent Pipeline with Skills

A **plan-and-execute multi-agent system** built on [LangGraph](https://github.com/langchain-ai/langgraph) that decomposes complex user tasks into ordered subtasks, dispatches each to a specialist LLM-powered agent, and assembles the results into a coherent final output.

## Architecture

```
User Task
    │
    ▼
┌─────────────────────┐
│   Orchestrator      │  Analyzes task, decomposes into a plan (DAG of steps),
│   (LLM Agent)       │  assigns each step to the best specialist agent
└────────┬────────────┘
         │ plan
         ▼
┌─────────────────────┐
│   Fan-out / Router  │  Finds steps whose dependencies are satisfied
└────────┬────────────┘
         │
    ┌────┴────┐  (parallel via LangGraph Send API)
    ▼    ▼    ▼
┌──────┐ ┌──────┐ ┌──────┐
│ Math │ │Research│ │Writer│   Specialist sub-agents with tools + skills
└──┬───┘ └──┬───┘ └──┬───┘
   │        │        │
   └────────┼────────┘
            │ results
            ▼
┌─────────────────────┐
│   Assembler          │  Merges all step outputs into the final answer
└─────────────────────┘
```

Graphs live in `graphs/` and register themselves — any module there that defines
a `build()` function is a graph, and its registry name is the file name minus a
trailing `_pipeline_graph` / `_graph` suffix. Two ship today:

- **`graphs/parallel_pipeline_graph.py`** → `parallel` — independent steps run concurrently via LangGraph's `Send` API
- **`graphs/sequential_pipeline_graph.py`** → `sequential` — one step at a time

Pick one with `python run_pipeline.py --graph sequential ...` or the `"graph"`
field of an API request; `parallel` is the default. `python run_pipeline.py
--list-graphs` (or `GET /graphs`) shows what the registry found.

## Directory Structure

```
agent_skills/
├── run_pipeline.py              # CLI entry point (--graph selects the pipeline)
├── api_server.py                # FastAPI REST API server
├── api_client.py                # CLI client for the API (zero dependencies)
├── agent_mcp_tools.py           # MCP client factory (reads config from agent_config.json)
├── assemble_node.py             # Shared assemble node (merges step outputs)
├── llm_factory.py               # create_llm(...) — the single ChatOpenAI construction site
├── skill_loader.py              # SKILL.md file loader (YAML frontmatter + body)
├── config_loader.py             # Unified agent-config loader + validator
├── scaffold/                    # Extension scaffolding tool (python -m scaffold)
├── pyproject.toml               # Packaging metadata + dev deps + pytest config
├── requirements.txt             # Pinned lockfile-style dependencies (Docker)
├── .env.example                 # Documented template — copy to .env
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Docker Compose service definition
├── .dockerignore                # Docker build exclusions
│
├── docs/
│   └── history/                 # Roadmap, test spec, and stale design-history docs
│
├── tests/                       # Hermetic pytest suite (no network / LLM needed)
│
├── graphs/
│   ├── __init__.py              # Auto-discovering graph registry (build_graph, available_graphs)
│   ├── parallel_pipeline_graph.py   # "parallel" — Send-API fan-out (default)
│   └── sequential_pipeline_graph.py # "sequential" — one step at a time
│
├── agents/
│   ├── __init__.py              # Exports orchestrator, sub-agents, loads roster from config
│   ├── agent_config.json        # Unified agent definitions (desc, tools, MCP servers)
│   ├── orchestrator_node.py     # Orchestrator: plans task decomposition
│   └── sub_agents_nodes.py      # Sub-agent execution (sequential + async)
│
├── skills/
│   ├── answer-writer/SKILL.md   # Skill for composing polished final answers
│   ├── frontend-design/SKILL.md # Skill for building frontend interfaces
│   ├── roll-dice/SKILL.md       # Skill for random dice rolls
│   └── yotta-researcher/SKILL.md # Skill for deep research with MCP tools
│
├── tools/
│   ├── __init__.py              # Auto-discovery tool registry — scans for @tool functions
│   ├── agent_tools.py           # Maps agent names to tool lists via config loader
│   ├── calculator.py            # Expression calculator (recursive descent parser)
│   ├── plotting.py              # Matplotlib plotting tool
│   └── bash_tool.py             # Bash command execution tool
│
├── utils/
│   ├── logger.py                # JSON-structured logging
│   ├── json_utils.py            # JSON extraction from LLM output
│   ├── plan_validator.py        # Plan schema + semantic validation
│   ├── sanitize.py              # Prompt injection detection
│   └── validator.py             # Sub-agent output validation
│
└── artifacts/                   # Generated output files (plots, etc.)
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM API (DeepSeek, OpenAI, vLLM, Ollama, etc.)
- Docker (optional — for containerized deployment)

### 2. Installation

```bash
# Clone the repository
git clone <repo-url> agent_skills
cd agent_skills

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / WSL
# or: venv\Scripts\activate  (Windows)

# Install the package (editable) with dev dependencies (pytest, ruff)
pip install -e ".[dev]"

# Alternative: pinned lockfile install (reproducible, e.g. for Docker)
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

```bash
# LLM backend (any OpenAI-compatible API works)
LLM_URL="https://api.deepseek.com"
LLM_MODEL="deepseek-v4-flash"
LLM_KEY="sk-your-api-key-here"

# Orchestrator planner temperature (low = more reliable JSON)
ORCHESTRATOR_TEMPERATURE=0.1

# Path to the unified agent configuration file (optional — defaults to
# agents/agent_config.json relative to the repo root if unset)
CONFIG_PATH="agents/agent_config.json"

# LangSmith tracing (optional — remove or set to false to disable)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_your-langsmith-key
LANGSMITH_PROJECT="YourProjectName"
```

> **Supported backends**: The system uses OpenAI-compatible chat completions. Any service exposing a `/v1/chat/completions` endpoint works — DeepSeek, OpenAI, vLLM, Ollama, LM Studio, Groq, etc.

### 4. Run

**Option A — CLI (direct execution):**

```bash
# Run with the default demo task
python run_pipeline.py

# Run with your own task
python run_pipeline.py "Explain the Fourier transform, plot sin(x) and cos(x), then write a summary"

# Pick a different graph (see what's available first)
python run_pipeline.py --list-graphs
python run_pipeline.py --graph sequential "Summarise the history of calculus"
```

**Option B — FastAPI server + client:**

```bash
# Start the API server
python api_server.py
# or: uvicorn api_server:app --host 0.0.0.0 --port 8000

# In another terminal, run a task via the CLI client
python api_client.py "Calculate sin(pi/4) + cos(pi/4) and plot both functions"

# Check server health
python api_client.py --health

# Async mode (start + poll until done)
python api_client.py --async "Research the history of machine learning"

# List the server's graphs, then run one of them
python api_client.py --list-graphs
python api_client.py --graph sequential "Summarise the history of calculus"
```

**Option C — Docker:**

```bash
# Build and start the service
docker compose up -d

# Run a task through the API
python api_client.py "Calculate sin(pi/4) + cos(pi/4) and explain the result"

# Check server health
python api_client.py --health

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

**What happens when you run:**
1. The **Orchestrator** reads your task and produces a JSON plan breaking it into steps
2. Each step is dispatched to the best **specialist sub-agent** (mathematician, researcher, writer)
3. Independent steps run in parallel via LangGraph's `Send` API
4. Sub-agents execute using their assigned **tools** and activated **skills**
5. The **Assembler** merges all step outputs into the final result

## Running the tests

The suite is pure-Python and hermetic — **no network, no `.env`, and no live
LLM are required.** It stubs the model with a fake chat client and pins dummy
env vars in `tests/conftest.py`, so it passes on a fresh clone:

```bash
pip install -e ".[dev]"   # installs pytest + ruff
pytest                    # runs the full suite
ruff check tests/         # lint the test code
```

Tests live in `tests/` (one module per production module). Key files:

- `tests/test_extension_contracts.py` — **conformance suite** (plan item 4.16).
  Parametrized over the live registries; run after adding any extension to
  verify it is well-formed. A broken skill, agent, tool or graph turns exactly
  one parametrized case red, with a message naming the offender.
- `tests/test_scaffold.py` — verifies that every scaffold kind produces
  output that passes the conformance suite (plan item 4.17).

A few tests are marked `xfail` — they document behavior that isn't implemented
yet (the blocked-forever deadlock guard in `fan_out_router` from the improvement
plan) and flip to passing as those fixes land. Tests that would need a live
endpoint are marked `@pytest.mark.integration` and excluded by default; see
[`docs/history/TESTING_GUIDE.md`](docs/history/TESTING_GUIDE.md) for the full
specification.

## Evaluation Harness

`python -m evals` runs golden plan-shape tasks through the **real orchestrator**
+ a **stub worker** — one LLM call per task, no tools executed. It checks that
the orchestrator produces plans matching the expected shape (step count,
assigned agents, dependency structure) and prints a pass/fail table.

```bash
# Run all golden tasks against the default (parallel) graph
python -m evals

# Run against a different graph
python -m evals --graph sequential

# Verbose: print each task text and per-assertion details on failure
python -m evals --graph parallel --verbose
```

Exit codes: `0` all tasks passed, `1` at least one FAIL or ERROR, `2` usage
error (bad graph name, missing tasks dir).

### Golden task format

Drop `.yaml` files into `evals/tasks/` — the runner picks them up automatically:

```yaml
task: "Calculate the maximum of x^2*sin(x) on [0,2], then summarise it"
expect:
  min_steps: 2           # plan must have at least this many steps
  max_steps: 6           # plan must have at most this many steps
  agents_include:        # these agents must appear in the plan
    - mathematician
    - writer
  has_dependency: true   # at least one step must depend_on another
```

All four `expect` keys are optional — omit any to skip that check, or omit
`expect` entirely for a smoke test ("does the graph run to completion?").
Unknown keys are ignored (forward-compatible).

### When to run it

- After changing the orchestrator prompt (`agents/orchestrator_node.py`)
- After adding or removing agents or skills (the roster the planner sees changes)
- After building a new graph, to verify it produces sane plans
- After model changes, to catch plan-quality regressions

The harness is **not** part of the default `pytest` suite (it needs a live LLM
endpoint). Its unit tests in `tests/test_evals_harness.py` are — they cover
every assertion and edge case without network access.

## Contributor Recipes

The four most common ways to extend the system. The first three are pure
config/data changes — no pipeline code is touched.

> **⚡ Quick start:** `python -m scaffold <kind> <name> [-d "description"]`
> generates a well-formed, immediately-testable stub in one command.
> Full usage and architecture: **[scaffold/README.md](scaffold/README.md)**

### Recipe 0 — Scaffold an extension

Generate well-formed stubs in one command. Each scaffold refuses to overwrite
and passes the conformance suite unchanged — verify with
`pytest tests/test_extension_contracts.py`.

```bash
python -m scaffold graph my-topology -d "Custom DAG with a critique loop"
python -m scaffold tool  word-count   -d "Count the words in a piece of text."
python -m scaffold skill code-review  -d "Review code for bugs and style issues."
python -m scaffold agent coder        -d "Software engineer."
```

Full usage, architecture, and how to extend the scaffold itself:
**[scaffold/README.md](scaffold/README.md)**.

If you prefer to create extensions by hand, the manual recipes follow.

### Recipe 1 — Add a tool

1. Create `tools/<name>.py` containing a LangChain `@tool`-decorated function:

   ```python
   from langchain_core.tools import tool

   @tool
   def word_count(text: str) -> int:
       """Count the words in a piece of text."""
       return len(text.split())
   ```

2. Assign it in `agents/agent_config.json` — add `"word_count"` to the
   `"tools"` list of each agent that should have it.

That's it. `tools/__init__.py` scans the directory at import time and collects
every `@tool` function into `TOOL_REGISTRY` — no imports, exports, or
registration code.

### Recipe 2 — Add an agent

Add one entry to `agents/agent_config.json` — no code:

```json
"coder": {
    "description": "Software engineer skilled in writing and reviewing code.",
    "tools": ["run_bash"],
    "mcp_servers": {},
    "llm": {"temperature": 0.3}
}
```

The orchestrator discovers new agents automatically on the next run. Write the
`description` for the planner, not for humans — it is the only signal the
orchestrator has when deciding which agent gets a step. `mcp_servers` maps
server name → URL; each MCP server must be owned by exactly one agent
(`config_loader.py` enforces this at startup). The optional `"llm"` block
accepts any of `model`, `url`, `api_key_env`, `temperature`, `max_tokens` —
every key is optional, and an absent block means "use the default endpoint."

### Recipe 3 — Add a skill

Create `skills/<skill-name>/SKILL.md` with YAML frontmatter and a body:

```markdown
---
name: my-skill
description: >
  What this skill does and when to use it. Shown to the orchestrator
  so it knows when to activate the skill for a step.
---

Detailed guidance, rules, and examples. This body is injected into the
sub-agent's system prompt when the skill is activated.
```

Discovered automatically by `skill_loader.py` — no registration.

### Recipe 4 — Add a graph

Drop one file in `graphs/`. There is nothing to register: the registry scans the
directory, and any module defining a `build()` function is a graph.

1. **Copy** `graphs/parallel_pipeline_graph.py` to
   `graphs/my_topology_graph.py` and change the topology. Each graph owns its
   own glue (scheduler, routers, assemble); only these building blocks are
   shared:

   | Building block | Source |
   |---|---|
   | `AgentState` / `WorkerState` — shared state TypedDicts | `agents/agent_states.py` |
   | `make_orchestrator_agent()` — task → JSON plan | `agents/orchestrator_node.py` |
   | `make_parallel_sub_agent_node()` / `make_sub_agent_node()` — run one plan step | `agents/sub_agents_nodes.py` |
   | `assemble_node` — merge step outputs into `final_output` | `assemble_node.py` |

2. **Expose a `build()`** with this signature — every argument defaults to the
   production wiring, so `build()` works with no arguments and tests can inject
   fakes:

   ```python
   GRAPH_DESCRIPTION = "One line shown by --list-graphs and GET /graphs"

   def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
       orchestrator = orchestrator or make_orchestrator_agent()
       sub_agent = sub_agent or make_parallel_sub_agent_node()
       ...
       return builder.compile(checkpointer=checkpointer or MemorySaver())
   ```

3. **Run it**: `python run_pipeline.py --graph my_topology "..."` or
   `{"task": "...", "graph": "my_topology"}` against the API. The name is the
   file name minus its `_pipeline_graph` / `_graph` / `_pipeline` suffix;
   set `GRAPH_NAME = "..."` in the module to override it. Modules whose name
   starts with `_` are skipped, so shared helpers can live in `graphs/` too.

Programmatic access:

```python
from graphs import available_graphs, build_graph, graph_descriptions

available_graphs()                       # ['parallel', 'sequential', 'my_topology']
graph = build_graph("my_topology", checkpointer=my_saver)
```

Discovery is lazy — `build_graph("parallel")` imports exactly one graph module,
and a graph file that fails to import only breaks callers asking for *that*
graph (listings report it instead of crashing).

Topology rules (the hard-won lessons behind items 1.1–1.2 of the improvement
plan):

- Route workers back through a **scheduler** node with plain `add_edge`
  calls, and hang the *only* conditional edge off the scheduler. Conditional
  edges are evaluated once per completed task and `Send` dispatches are not
  deduplicated — a conditional edge attached directly to the worker
  double-dispatches dependent steps. The scheduler also propagates
  `[SKIPPED — dependency failed]` markers for steps transitively blocked by
  a failed step (Phase 4.13).
- Pass `add_conditional_edges` a callable router plus a plain **list** of
  target names (`["assemble", "parallel_sub_agent"]`) — never a dict keyed on
  the `Send` class, and never a node-name string in place of the router.
- New state fields go in `AgentState` (`agent_states.py`); fields written by
  parallel branches need a reducer annotation (see `results`) to merge.

## How It Works

### Unified Agent Configuration

All agent definitions — descriptions, tool assignments, and MCP server ownership — live in a single file: `agents/agent_config.json`. The path is set via the `CONFIG_PATH` environment variable in `.env`.

**Agent configuration format** (`agents/agent_config.json`):

```json
{
    "mathematician": {
        "description": "Expert in solving complex mathematical problems and plotting functions.",
        "tools": ["calculate", "plotting_tool", "run_bash"],
        "mcp_servers": {}
    },
    "researcher": {
        "description": "Skilled in gathering and synthesizing information from various sources.",
        "tools": ["run_bash"],
        "mcp_servers": {
            "yotta_mcp": "http://207.189.105.118:8001/mcp"
        }
    },
    "writer": {
        "description": "Proficient in crafting clear and engaging written content on a wide range of topics.",
        "tools": [],
        "mcp_servers": {}
    }
}
```

Each agent entry has four fields (two required, two optional):
- `description` — human-readable role summary (shown to the orchestrator)
- `tools` — list of tool names to assign to this agent (resolved against the auto-discovered `TOOL_REGISTRY`)
- `mcp_servers` — dict of MCP server name → URL owned by this agent
- `llm` *(optional)* — per-agent LLM overrides (`model`, `url`, `api_key_env`, `temperature`, `max_tokens`). Every key is optional; an absent block means "use the env-configured default". `api_key_env` names an environment variable — never stores a literal key.

The `config_loader.py` module loads this file at import time and validates MCP ownership, `llm`-block key typos, and that required fields are present — a bad key fails loudly at startup instead of silently falling back to the default endpoint mid-run.

To add a new agent, see [Recipe 2](#recipe-2--add-an-agent).

### Skills

Skills are reusable capability documents (written in Markdown with YAML frontmatter) that get injected into a sub-agent's system prompt when activated for a step. They describe *how* to perform a specific kind of task.

**Skill format** (`skills/<skill-name>/SKILL.md`):

```markdown
---
name: my-skill
description: >
  Brief description of what this skill does and when to use it.
  This is shown to the orchestrator so it knows when to activate the skill.
---

# Skill body

These instructions are injected into the sub-agent's system prompt.
Write detailed guidance, rules, and examples here.
```

To add a new skill, see [Recipe 3](#recipe-3--add-a-skill).

**Available skills:**

| Skill | Purpose |
|---|---|
| `answer-writer` | Compose polished, well-cited final answers synthesizing research |
| `frontend-design` | Create production-grade frontend interfaces with distinctive design |
| `roll-dice` | Generate random dice rolls via bash |
| `yotta-researcher` | Deep research skill leveraging MCP tools for gathering and synthesizing information |

### The Pipeline

**Shared State** (`agent_states.py`):
```python
class AgentState(TypedDict):
    task: str                    # User's original request
    plan: list[PlanStep]         # Orchestrator's decomposition
    results: dict[int, str]      # Accumulated step outputs (incl. failure markers)
    failed_steps: list[int]      # Steps that failed after retries (Phase 4.13)
    step_stats: list[StepStats]  # Per-step timing, tokens, and tool calls (Phase 4.9)
    final_output: str            # Assembled final answer (with warning header)
    current_datetime: str        # Current date/time for context
```

Each `PlanStep` has:
- `step` — integer identifier
- `subtask` — concise description of the step
- `agent` — which specialist to use
- `skills_needed` — which skills to activate
- `depends_on` — list of step numbers that must complete first

**Execution flow:**

```
START → orchestrator → router → [sub_agent(s)] → router → assembler → END
```

1. **Orchestrator** — An LLM call that decomposes the task into a JSON plan
2. **Router** — Checks which steps have all dependencies satisfied; dispatches them via `Send` API
3. **Sub-Agents** — Independent steps run in parallel as LangGraph ReAct agents with tools + skills
4. **Assembler** — Concatenates all step outputs into the final answer

**Sequential vs Parallel:**
- `graphs/sequential_pipeline_graph.py` (`--graph sequential`) runs one step at a time, in dependency order
- `graphs/parallel_pipeline_graph.py` (`--graph parallel`, the default) runs all ready steps concurrently via LangGraph's `Send` API

To build your own topology from these parts, see
[Recipe 4](#recipe-4--add-a-graph).

### Tools

Tools are LangChain `@tool`-decorated Python functions that sub-agents can call.

**Auto-discovery**: `tools/__init__.py` scans all `.py` files in the `tools/` directory at import time and collects every `@tool`-decorated function into a `TOOL_REGISTRY` dict. No manual exports or registration needed — drop a file and it's picked up automatically.

| Tool | File | Description |
|---|---|---|
| `calculate(expr)` | `tools/calculator.py` | Recursive descent expression parser. Supports arithmetic, trig, log, constants (pi, e, phi), 30+ functions, factorial, combinatorics |
| `plotting_tool(expr, x_min, x_max)` | `tools/plotting.py` | Plots a mathematical expression using NumPy + Matplotlib. Returns the image path |
| `run_bash(command, timeout)` | `tools/bash_tool.py` | Executes a bash command in a sandboxed subprocess |
| `run_bash_with_approval(command)` | `tools/bash_tool.py` | Same as above but requires user confirmation first |

**Tool assignment** — edit the `"tools"` list for the agent in `agents/agent_config.json`:
```json
"mathematician": {
    "description": "Expert in solving complex mathematical problems.",
    "tools": ["calculate", "plotting_tool", "run_bash"],
    "mcp_servers": {}
}
```

To add a new tool, see [Recipe 1](#recipe-1--add-a-tool).

### MCP Integration

The system supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting sub-agents to external tool servers.

**Configuration** — MCP server assignments live in `agents/agent_config.json` under each agent's `"mcp_servers"` key:

```json
"researcher": {
    "description": "Skilled in gathering and synthesizing information.",
    "tools": ["run_bash"],
    "mcp_servers": {
        "yotta_mcp": "http://207.189.105.118:8001/mcp"
    }
}
```

Each MCP server must have a single owning agent for security — `config_loader.py` validates this at startup and `agent_mcp_tools.py` re-checks at runtime. When a sub-agent runs, it combines its native LangChain tools with any MCP tools from servers it owns.

### FastAPI REST API

`api_server.py` exposes the pipeline as a RESTful service:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/graphs` | GET | List the graphs discovered in `graphs/`, with descriptions and which is the default |
| `/run` | POST | Run the pipeline synchronously (blocks until complete). Body: `{"task": "...", "graph": "parallel"}` — `graph` is optional |
| `/run-async` | POST | Start a pipeline run in the background (same body). Returns a `task_id` immediately (HTTP 202) |
| `/status/{task_id}` | GET | Poll for async task status. Returns `"running"`, `"completed"` (with `final_output`, `step_stats`), or `"failed"` (with `error`) |

The API uses Pydantic models for request/response validation and includes CORS middleware (open by default — tighten in production). A zero-dependency CLI client (`api_client.py`) is provided for interacting with the API from the terminal.

### Security Features

- **Prompt injection detection** (`utils/sanitize.py`) — Scans user input for jailbreak patterns (ignore instructions, system prompt extraction, data exfiltration via markdown images)
- **Output validation** (`utils/validator.py`) — Blocks XSS vectors (`<script`), prompt leakage patterns, empty outputs, and oversized outputs (>50K chars)
- **Sandboxed bash** — `run_bash` drops privileges to `nobody` user before executing
- **MCP ownership validation** — Agents can only access MCP servers they explicitly own; `config_loader.py` enforces exclusive ownership at startup; `agent_mcp_tools.py` re-checks at runtime
- **Retry policy** — Sub-agent nodes have `RetryPolicy(max_attempts=2)` for automatic retries on transient errors
- **Failure containment** — A step that exhausts retries is recorded as `[STEP FAILED]` and its dependents are marked `[SKIPPED — dependency failed]` without being dispatched; the assembler prepends a warning header to `final_output` so partial results are clearly flagged

## Configuration Reference

| Environment Variable | Purpose | Default |
|---|---|---|
| `LLM_URL` | OpenAI-compatible API base URL | `https://api.deepseek.com` |
| `LLM_MODEL` | Model name to use | `deepseek-v4-flash` |
| `LLM_KEY` | API key for the LLM service | (required) |
| `CONFIG_PATH` | Path to the unified agent config JSON | `agents/agent_config.json` |
| `LOG_LEVEL` | Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) | `INFO` |
| `LOG_FILE` | Log file path (empty string disables file logging) | `langgraph_smart_reasoning.log` |
| `LOG_CONSOLE` | Also emit log events to stderr (`true`/`false`) | `false` |
| `DEBUG` | Include full tracebacks in API error responses | `false` |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `true` |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | LangSmith API key | (optional) |
| `LANGSMITH_PROJECT` | LangSmith project name | `TestingLG` |

## Docker

The service is containerized for easy deployment. Key details:

- **Base image**: `python:3.11-slim`
- **System dependencies**: `libfreetype6-dev`, `libpng-dev`, `libgomp1` (required by matplotlib)
- **Port**: `8000` (FastAPI + Uvicorn)
- **Volumes**: `artifacts/` (plots survive restarts), `skills/` (new skills picked up without rebuild, read-only)
- **Healthcheck**: polls `/health` endpoint every 30s

```bash
# Build and start
docker compose up -d --build

# Check status
docker compose ps
python api_client.py --health

# Stop
docker compose down
```

## Dependencies

| Package | Purpose |
|---|---|
| `langgraph>=1.2` | Graph-based agent orchestration |
| `openai>=1.78.0` | OpenAI-compatible API client |
| `langchain-openai` | LangChain wrapper for chat models |
| `langchain-mcp-adapters` | MCP tool integration |
| `langsmith==0.8.8` | LLM tracing and observability |
| `pyyaml` | YAML frontmatter parsing for SKILL.md |
| `python-dotenv` | `.env` file loading |
| `numpy` | Numerical computing (plotting) |
| `matplotlib` | Plot generation |
| `fastapi>=0.115.0` | REST API server |
| `uvicorn[standard]>=0.34.0` | ASGI server for FastAPI |

## Example: End-to-End

Given the task: *"Calculate sin(pi/4) + cos(pi/4) and explain the result. Then write a short summary."*

The **Orchestrator** produces:
```json
{
  "plan": [
    {
      "step": 1,
      "subtask": "Calculate sin(pi/4) + cos(pi/4)",
      "agent": "mathematician",
      "skills_needed": [],
      "depends_on": []
    },
    {
      "step": 2,
      "subtask": "Explain the mathematical meaning of the result",
      "agent": "mathematician",
      "skills_needed": [],
      "depends_on": [1]
    },
    {
      "step": 3,
      "subtask": "Write a short summary of the calculation and its meaning",
      "agent": "writer",
      "skills_needed": ["answer-writer"],
      "depends_on": [1, 2]
    }
  ]
}
```

**Execution:**
- Step 1 runs immediately (no dependencies) → mathematician calculates `sqrt(2) ≈ 1.414`
- Step 2 runs after step 1 → mathematician explains the trigonometric identity
- Step 3 runs after steps 1 and 2 → writer synthesizes a polished summary using the `answer-writer` skill

**Final output** — all three step results assembled into one cohesive document.

## Documentation

This README is the single entry point; everything else lives in
`docs/history/`:

| Document | What it is |
|---|---|
| [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md) | **Active roadmap** — phased plan with acceptance criteria per item |
| [`docs/history/TESTING_GUIDE.md`](docs/history/TESTING_GUIDE.md) | **Active test-suite specification** — implemented by `tests/` |
| `docs/history/multi-agent-pipeline-skills-guide.md` | Design history — original implementation plan (stale) |
| `docs/history/langgraph-multi-agent-skills-plan.md` | Design history — LangGraph adaptation plan (stale) |
| `docs/history/codebase-review-fixes.md` | Design history — June 2026 code review (stale) |

When you add a feature, update the relevant [recipe](#contributor-recipes) in
the same PR — documentation drift is how this repo once ended up with four
overlapping, contradictory planning documents.

## Known Issues

Remaining work is tracked in
[`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md). The two
bugs that section once listed (the parallel graph failing at import, and
`validate_plan` being called with the wrong arity) are fixed; their regression
tests in `tests/` are green.

Caveats:

- **MCP client**: Ensure the MCP server is reachable before running agents that depend on it; the pipeline will fail if an MCP-dependent agent is dispatched and the server is down.
- **Skill body parsing**: `skill_loader.py` uses `maxsplit=2` when splitting on `---` to handle body content containing dash sequences. Ensure SKILL.md files follow the standard `---\n(YAML frontmatter)\n---\n(body)` format.
- **Sequential pipeline**: `graphs/sequential_pipeline_graph.py` is a reference topology — it re-evaluates its router after every step and has no scheduler barrier. `parallel` is the default everywhere.
- **Old import path**: the root `paralel_pipeline_graph.py` (misspelled) is an import-compat shim for one release and no longer exposes a pre-compiled `graph` singleton — use `build_graph("parallel")`.

## License

This project is provided as-is for educational and experimental use. See individual skill files for any additional license terms (e.g., `skills/frontend-design/SKILL.md` references a LICENSE.txt).
