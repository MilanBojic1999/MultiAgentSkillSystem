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
- **`paralel_pipeline_graph.py`** — Parallel execution (independent steps run concurrently)

## Directory Structure

```
agent_skills/
├── run_pipeline.py              # Entry point for the sequential pipeline
├── pipeline_graph.py            # Sequential LangGraph pipeline
├── paralel_pipeline_graph.py    # Parallel LangGraph pipeline (Send API fan-out)
├── agent_states.py              # Shared state definition (TypedDict)
├── agent_mcp_tools.py           # MCP client factory for agent tools
├── skill_loader.py              # SKILL.md file loader (YAML frontmatter + body)
├── requirements.txt             # Python dependencies
├── .env                         # LLM config + LangSmith tracing
│
├── agents/
│   ├── __init__.py              # Exports orchestrator, sub-agents, loads roster
│   ├── agent_rouster.json       # Agent definitions (name → description)
│   ├── orchestrator_node.py     # Orchestrator: plans task decomposition
│   └── sub_agents_nodes.py      # Sub-agent execution (sequential + async)
│
├── skills/
│   ├── answer-writer/SKILL.md   # Skill for composing polished final answers
│   ├── frontend-design/SKILL.md # Skill for building frontend interfaces
│   └── roll-dice/SKILL.md       # Skill for random dice rolls
│
├── tools/
│   ├── __init__.py              # Tool exports
│   ├── agent_tools.py           # Maps agent names to their tool lists
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

# LangSmith tracing (optional — remove or set to false to disable)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=lsv2_your-langsmith-key
LANGSMITH_PROJECT="YourProjectName"
```

> **Supported backends**: The system uses OpenAI-compatible chat completions. Any service exposing a `/v1/chat/completions` endpoint works — DeepSeek, OpenAI, vLLM, Ollama, LM Studio, Groq, etc.

### 4. Run

```bash
# Run with the default demo task
python run_pipeline.py

# Run with your own task
python run_pipeline.py "Explain the Fourier transform, plot sin(x) and cos(x), then write a summary"
```

**What happens when you run:**
1. The **Orchestrator** reads your task and produces a JSON plan breaking it into steps
2. Each step is dispatched to the best **specialist sub-agent** (mathematician, researcher, writer)
3. Sub-agents execute using their assigned **tools** and activated **skills**
4. The **Assembler** merges all step outputs into the final result

## How It Works

### Agents

Agents are specialist LLM-powered workers. Each has a name, a role description, and an assigned set of tools.

**Agent definitions** live in `agents/agent_rouster.json`:

```json
{
    "mathematician": "Expert in solving complex mathematical problems and plotting functions.",
    "researcher": "Skilled in gathering and synthesizing information from various sources.",
    "writer": "Proficient in crafting clear and engaging written content on a wide range of topics."
}
```

**Adding a new agent:**

1. Add an entry to `agents/agent_rouster.json`:
   ```json
   "coder": "Software engineer skilled in writing and reviewing code."
   ```

2. Optionally assign tools in `tools/agent_tools.py`:
   ```python
   AGENT_TOOLS = {
       # ... existing entries ...
       "coder": [run_bash],
   }
   ```

3. Optionally configure MCP access in `agent_mcp_tools.py` (see MCP section below).

The orchestrator discovers new agents automatically on the next run — no other code changes needed.

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

### The Pipeline

**Shared State** (`agent_states.py`):
```python
class AgentState(TypedDict):
    task: str                    # User's original request
    plan: list[PlanStep]         # Orchestrator's decomposition
    results: dict[int, str]      # Accumulated step outputs
    final_output: str            # Assembled final answer
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
2. **Router** — Checks which steps have all dependencies satisfied; dispatches them
3. **Sub-Agents** — Each step runs as a LangGraph ReAct agent with tools + skills
4. **Assembler** — Concatenates all step outputs into the final answer

**Sequential vs Parallel:**
- `pipeline_graph.py` runs one step at a time, in dependency order
- `paralel_pipeline_graph.py` runs all ready steps concurrently via LangGraph's `Send` API

### Tools

Tools are LangChain `@tool`-decorated Python functions that sub-agents can call:

| Tool | Description |
|---|---|
| `calculate(expr)` | Recursive descent expression parser. Supports arithmetic, trig, log, constants (pi, e, phi), 30+ functions, factorial, combinatorics |
| `plotting_tool(expr, x_min, x_max)` | Plots a mathematical expression using NumPy + Matplotlib. Returns the image path |
| `run_bash(command, timeout)` | Executes a bash command in a sandboxed subprocess |
| `run_bash_with_approval(command)` | Same as above but requires user confirmation first |

**Tool assignment** in `tools/agent_tools.py`:
```python
AGENT_TOOLS = {
    "mathematician": [calculate, plotting_tool, run_bash],
    "researcher": [run_bash],
    "writer": [],
}
```

**Adding a new tool:**

1. Create a new file in `tools/` (e.g., `tools/web_search.py`)
2. Define a function decorated with `@tool` from LangChain
3. Export it in `tools/__init__.py`
4. Assign it to agents in `tools/agent_tools.py`

### MCP Integration

The system supports the [Model Context Protocol](https://modelcontextprotocol.io/) for connecting sub-agents to external tool servers.

**Configuration** in `agent_mcp_tools.py`:
```python
MCP_ACCESS_AGENT = {
    "researcher": {"yotta_mcp": "http://207.189.105.118:8001/mcp"},
}

DEDICATED_MCP_OWNERS = {
    "yotta_mcp": "researcher",  # Only "researcher" owns this server
}
```

Each MCP server must have a single owning agent for security. When a sub-agent runs, it combines its native LangChain tools with any MCP tools from servers it owns.

### Security Features

- **Prompt injection detection** (`utils/senitize.py`) — Scans user input for jailbreak patterns (ignore instructions, system prompt extraction, data exfiltration via markdown images)
- **Output validation** (`utils/validator.py`) — Blocks XSS vectors (`<script`), prompt leakage patterns, empty outputs, and oversized outputs (>50K chars)
- **Sandboxed bash** — `run_bash` drops privileges to `nobody` user before executing
- **MCP ownership** — Agents can only access MCP servers they explicitly own
- **Retry policy** — Sub-agent nodes have `RetryPolicy(max_attempts=2)` for automatic retries on failure

## Configuration Reference

| Environment Variable | Purpose | Default |
|---|---|---|
| `LLM_URL` | OpenAI-compatible API base URL | `https://api.deepseek.com` |
| `LLM_MODEL` | Model name to use | `deepseek-v4-flash` |
| `LLM_KEY` | API key for the LLM service | (required) |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `true` |
| `LANGSMITH_ENDPOINT` | LangSmith API endpoint | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | LangSmith API key | (optional) |
| `LANGSMITH_PROJECT` | LangSmith project name | `TestingLG` |

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
2. **MCP client**: `create_mcp_client()` returns `(client, None)` when no MCP is configured but callers expect `(client, tools)`; ensure the MCP server is reachable before running agents that depend on it.
3. **Parallel pipeline**: `paralel_pipeline_graph.py` needs a separate entry point or runner (currently only `run_pipeline.py` exists for the sequential pipeline).
4. **Skill index in state**: The sequential `sub_agent_node` uses the module-level loaded skill index rather than state-passed skills; ensure the skills directory is populated before running.

## License

This project is provided as-is for educational and experimental use. See individual skill files for any additional license terms (e.g., `skills/frontend-design/SKILL.md` references a LICENSE.txt).
