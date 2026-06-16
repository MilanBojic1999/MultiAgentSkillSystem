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

Two pipeline implementations are provided:
- **`pipeline_graph.py`** — Sequential execution (one step at a time)
- **`paralel_pipeline_graph.py`** — Parallel execution (independent steps run concurrently via LangGraph's `Send` API)

The default runner (`run_pipeline.py`) uses the parallel pipeline.

## Directory Structure

```
agent_skills/
├── run_pipeline.py              # CLI entry point (uses parallel pipeline)
├── pipeline_graph.py            # Sequential LangGraph pipeline
├── paralel_pipeline_graph.py    # Parallel LangGraph pipeline (Send API fan-out)
├── api_server.py                # FastAPI REST API server
├── api_client.py                # CLI client for the API (zero dependencies)
├── agent_states.py              # Shared state definition (TypedDict)
├── agent_mcp_tools.py           # MCP client factory (reads config from agent_config.json)
├── skill_loader.py              # SKILL.md file loader (YAML frontmatter + body)
├── config_loader.py             # Unified agent-config loader + validator
├── requirements.txt             # Python dependencies
├── .env                         # LLM config, LangSmith tracing, config path
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Docker Compose service definition
├── .dockerignore                # Docker build exclusions
│
├── agents/
│   ├── __init__.py              # Exports orchestrator, sub-agents, loads roster from config
│   ├── agent_config.json        # Unified agent definitions (desc, tools, MCP servers)
│   ├── agent_rouster.json       # Legacy agent roster (kept for compatibility)
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
│   ├── senitize.py              # Prompt injection detection
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

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy or edit the `.env` file with your API credentials:

```bash
# LLM backend (any OpenAI-compatible API works)
LLM_URL="https://api.deepseek.com"
LLM_MODEL="deepseek-v4-flash"
LLM_KEY="sk-your-api-key-here"

# Path to the unified agent configuration file
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

Each agent entry has three fields:
- `description` — human-readable role summary (shown to the orchestrator)
- `tools` — list of tool names to assign to this agent (resolved against the auto-discovered `TOOL_REGISTRY`)
- `mcp_servers` — dict of MCP server name → URL owned by this agent

The `config_loader.py` module loads this file at import time and validates that no MCP server is claimed by more than one agent (each MCP server must have exactly one owner).

**Adding a new agent:**

1. Add an entry to `agents/agent_config.json`:
   ```json
   "coder": {
       "description": "Software engineer skilled in writing and reviewing code.",
       "tools": ["run_bash"],
       "mcp_servers": {}
   }
   ```

2. That's it — the orchestrator discovers new agents automatically on the next run. No other code changes needed.

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

**Adding a new skill:**

1. Create a directory: `skills/my-new-skill/`
2. Create `SKILL.md` inside it with the YAML frontmatter and body
3. That's it — the skill is discovered automatically via `skill_loader.py`

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
    results: dict[int, str]      # Accumulated step outputs
    final_output: str            # Assembled final answer
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
- `pipeline_graph.py` runs one step at a time, in dependency order
- `paralel_pipeline_graph.py` runs all ready steps concurrently via LangGraph's `Send` API (used by default)

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

**Adding a new tool:**

1. Create a new `.py` file in `tools/` (e.g., `tools/web_search.py`)
2. Define a function decorated with `@tool` from LangChain inside it
3. Assign the tool name to agents in `agents/agent_config.json` under their `"tools"` list
4. That's it — the tool is discovered automatically at import time. No other registration needed.

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
| `/run` | POST | Run the pipeline synchronously (blocks until complete). Body: `{"task": "..."}` |
| `/run-async` | POST | Start a pipeline run in the background. Returns a `task_id` immediately (HTTP 202) |
| `/status/{task_id}` | GET | Poll for async task status. Returns `"running"`, `"completed"` (with `final_output`), or `"failed"` (with `error`) |

The API uses Pydantic models for request/response validation and includes CORS middleware (open by default — tighten in production). A zero-dependency CLI client (`api_client.py`) is provided for interacting with the API from the terminal.

### Security Features

- **Prompt injection detection** (`utils/senitize.py`) — Scans user input for jailbreak patterns (ignore instructions, system prompt extraction, data exfiltration via markdown images)
- **Output validation** (`utils/validator.py`) — Blocks XSS vectors (`<script`), prompt leakage patterns, empty outputs, and oversized outputs (>50K chars)
- **Sandboxed bash** — `run_bash` drops privileges to `nobody` user before executing
- **MCP ownership validation** — Agents can only access MCP servers they explicitly own; `config_loader.py` enforces exclusive ownership at startup; `agent_mcp_tools.py` re-checks at runtime
- **Retry policy** — Parallel sub-agent nodes have `RetryPolicy(max_attempts=2)` for automatic retries on failure

## Configuration Reference

| Environment Variable | Purpose | Default |
|---|---|---|
| `LLM_URL` | OpenAI-compatible API base URL | `https://api.deepseek.com` |
| `LLM_MODEL` | Model name to use | `deepseek-v4-flash` |
| `LLM_KEY` | API key for the LLM service | (required) |
| `CONFIG_PATH` | Path to the unified agent config JSON | `agents/agent_config.json` |
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

## Known Issues

1. **Logging**: `utils/logger.py` calls `logging.basicConfig` after `getLogger`, which means the file handler may not be attached before the first log call.
2. **MCP client**: Ensure the MCP server is reachable before running agents that depend on it; the pipeline will fail if an MCP-dependent agent is dispatched and the server is down.
3. **Skill body parsing**: `skill_loader.py` uses `maxsplit=2` when splitting on `---` to handle body content containing dash sequences. Ensure SKILL.md files follow the standard `---\n(YAML frontmatter)\n---\n(body)` format.
4. **Parallel pipeline**: The parallel pipeline (`paralel_pipeline_graph.py`) is the default in both `run_pipeline.py` and `api_server.py`. The sequential pipeline (`pipeline_graph.py`) exists for reference but is not wired to a runner.

## License

This project is provided as-is for educational and experimental use. See individual skill files for any additional license terms (e.g., `skills/frontend-design/SKILL.md` references a LICENSE.txt).
