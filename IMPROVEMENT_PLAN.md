# Improvement Plan — Multi-Agent Skills Pipeline

> **Status:** Active roadmap
> **Created:** 2026-07-07
> **Audience:** Developers joining or extending this codebase
>
> This is the single forward-looking plan for the project. Each item states the
> problem, why it matters, the affected files, a code sketch where the design is
> non-obvious, and acceptance criteria. Work through phases in order — Phase 1
> items are prerequisites for safely building anything in Phases 3–4.

---

## Table of Contents

- [Architecture Recap](#architecture-recap)
- [Phase 1 — Correctness Fixes](#phase-1--correctness-fixes)
- [Phase 2 — Developer Experience & Adoption](#phase-2--developer-experience--adoption)
- [Phase 3 — Architecture Refactor (Graph Factories)](#phase-3--architecture-refactor-graph-factories)
- [Phase 4 — New Features](#phase-4--new-features)
- [Suggested Order & Dependencies](#suggested-order--dependencies)

---

## Architecture Recap

The system is a **plan-and-execute** pipeline built on LangGraph:

1. **Orchestrator** (`agents/orchestrator_node.py`) — LLM call that decomposes the
   user task into a JSON plan: a DAG of steps, each assigned to a specialist agent
   with a list of skills.
2. **Sub-agents** (`agents/sub_agents_nodes.py`) — each step runs a
   `create_react_agent` with the agent's native tools (`tools/` auto-discovery),
   MCP tools (`agent_mcp_tools.py`), and the bodies of the requested skills
   (`skill_loader.py`).
3. **Assemble** — concatenates step outputs into `final_output`.

Two graph topologies exist: sequential (`pipeline_graph.py`) and parallel via the
`Send` API (`paralel_pipeline_graph.py`). Agent definitions (description, tools,
MCP servers) live in one file: `agents/agent_config.json`, loaded and validated by
`config_loader.py`. Entry points: `run_pipeline.py` (CLI) and `api_server.py`
(FastAPI: `/run`, `/run-async`, `/status/{id}`).

---

## Phase 1 — Correctness Fixes

### 1.1 Duplicate `Send` dispatch — add a scheduler node ⚠️ highest priority

**Problem.** In `paralel_pipeline_graph.py` the fan-out conditional edge hangs
directly off the worker node (`paralel_pipeline_graph.py:52`). LangGraph
evaluates conditional edges **once per completed task**, and `Send` tasks are
*not* deduplicated (only plain string routing is). So if a layer ran N parallel
steps, `fan_out_router` runs N times on the same merged state and each
evaluation dispatches the same newly-ready steps.

**Failure scenario.** Plan: steps 1 and 2 in parallel, step 3 depends on both.
Steps 1 and 2 finish in the same superstep → the router runs twice → step 3 is
`Send`-dispatched **twice**. Two identical LLM runs are paid for; whichever
result merges last wins (nondeterministic).

**Fix.** Route workers back through a no-op **scheduler** node via a plain edge.
Plain node routing deduplicates (a node executes at most once per superstep), so
the fan-out router runs exactly once per layer.

```python
def scheduler_node(state: dict) -> dict:
    """Synchronization barrier between dependency layers. No-op."""
    return {}

builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator_agent,
                 retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
builder.add_node("scheduler", scheduler_node)
builder.add_node("parallel_sub_agent", parallel_sub_agent_node,
                 retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
builder.add_node("assemble", assemble_node)

builder.set_entry_point("orchestrator")
builder.add_edge("orchestrator", "scheduler")
# The ONLY conditional edge — evaluated once per superstep:
builder.add_conditional_edges("scheduler", fan_out_router,
                              ["assemble", "parallel_sub_agent"])
builder.add_edge("parallel_sub_agent", "scheduler")   # plain edge = deduped
builder.add_edge("assemble", END)
```

Note the path map fix too: the current
`{"assemble": "assemble", Send: "parallel_sub_agent"}` uses the `Send` *class*
as a dict key — it only appears to work because `Send` objects bypass the path
map entirely. Use a plain list of target names.

**Acceptance criteria.**
- A regression test builds this topology with a stub worker that counts
  invocations per step; for a 2-parallel + 1-dependent plan, every step runs
  **exactly once** (see 2.3).
- Orchestrator gets a `RetryPolicy` so plan-validation failures (1.2) trigger a
  re-plan instead of killing the run.

---

### 1.2 Plan validation at the orchestrator

**Problem.** The plan is used as-is after JSON extraction
(`agents/orchestrator_node.py:110-116`). Consequences today:

- Unknown agent name → bare `StopIteration` deep inside a worker
  (`agents/sub_agents_nodes.py:57` — that `next(...)` scan is also pointless; it
  returns `agent_name` back).
- Unknown skill names → silently dropped (`sub_agents_nodes.py:62-66`).
- `depends_on` pointing at a non-existent step, or a dependency cycle →
  `fan_out_router` finds nothing ready and returns `"assemble"` → final output
  with **empty sections and no error**. In the sequential graph it's worse:
  `should_continue` loops forever because `sub_agent_node` returns `{}` and
  results never grow.

**Fix.** Validate once, right after JSON extraction, in a new
`utils/plan_validator.py`. Pydantic for the schema, explicit checks for
semantics:

```python
class PlanStepModel(BaseModel):
    step: int = Field(ge=1)
    subtask: str = Field(min_length=1)
    agent: str
    skills_needed: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)

def validate_plan(plan, known_agents, known_skills) -> list[dict]:
    # 1. schema-validate every step (PlanStepModel)
    # 2. step numbers unique
    # 3. agent in known_agents             -> hard error (can't execute)
    # 4. unknown skills_needed entries     -> drop + log_event warning (soft)
    # 5. depends_on ⊆ step numbers, no self-reference -> hard error
    # 6. dependency graph acyclic (Kahn's algorithm)  -> hard error
    # returns normalized list[dict], sorted by step
```

Raise a dedicated `PlanValidationError(ValueError)` so the orchestrator's
`RetryPolicy(retry_on=(ValueError,))` re-plans on hard errors. Also add
defensive guards so "impossible" states fail loudly instead of silently:

- `fan_out_router`: if nothing is ready but unfinished steps remain, raise
  `RuntimeError` listing the blocked steps.
- sequential `sub_agent_node`: same guard instead of returning `{}`.

**Acceptance criteria.** Unit tests cover: valid plan passes and is normalized;
unknown agent raises; unknown skill is dropped with a warning; dangling
`depends_on` raises; cycle raises; duplicate step numbers raise.

---

### 1.3 Orchestrator LLM settings

**Problem.** The orchestrator — whose only job is emitting strict JSON — runs at
`temperature=0.9` (`agents/orchestrator_node.py:26`). This is the main source of
malformed plans. Both modules also duplicate identical `ChatOpenAI`
construction at import time.

**Fix.** Centralize in `llm_factory.py` (full design in Phase 4.3 — build the
factory now, add per-agent overrides later):

```python
def create_llm(model=None, url=None, api_key_env=None,
               temperature=None, max_tokens=None) -> ChatOpenAI:
    """Falls back to LLM_MODEL / LLM_URL / LLM_KEY from .env.
    Raises a clear EnvironmentError naming any missing variable.
    Instances are cached by parameter tuple (lru_cache)."""
```

Orchestrator uses `create_llm(temperature=float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.1")))`.
Sub-agents keep the creative default (0.9) for now. Also fix `max_tokens=4048`
(an odd non-power-of-two — presumably meant 4096).

**Stretch (optional, vLLM-specific).** Since the backend is vLLM, replace
`_extract_json` fragility with structured output / guided decoding:
`llm.with_structured_output(PlanModel)`. Keep `_extract_json` as fallback for
other OpenAI-compatible backends.

**Acceptance criteria.** One `ChatOpenAI` construction site in the codebase;
orchestrator temperature ≤ 0.2 by default and env-overridable.

---

### 1.4 Per-step failure containment

**Problem.** When a sub-agent exhausts its 2 retries, the exception propagates
and the whole graph dies — a 5-step run with one flaky step produces *nothing*.

**Fix.** Catch in the worker node, record the failure as the step result, and
let downstream logic decide:

```python
async def parallel_sub_agent_node(state: dict) -> dict:
    try:
        step_num, output = await run_sub_agent_async(...)
        return {"results": {step_num: output}}
    except Exception as e:
        log_event("sub_agent_step_failed", step=state["step"]["step"], error=str(e))
        return {"results": {state["step"]["step"]: f"[STEP FAILED] {e}"},
                "failed_steps": [state["step"]["step"]]}
```

Add `failed_steps: Annotated[list[int], operator.add]` to `AgentState`. The
router treats a failed step's dependents as blocked → marks them
`[SKIPPED — dependency failed]` in results. `assemble_node` prepends a warning
header when `failed_steps` is non-empty.

**Design note.** Keep the node-level `RetryPolicy` for transient errors; this
catch is the *final* fallback after retries. Don't let validation errors from
`utils/validator.py` bypass retries — they should still raise inside
`run_sub_agent_async` so the retry re-runs the LLM.

**Acceptance criteria.** A run where one step always throws still produces a
`final_output` containing the other steps' results plus explicit
failed/skipped markers.

---

### 1.5 Small fixes checklist

| Fix | Where |
|---|---|
| Delete the pointless `next(...)` agent scan (validation now guarantees the name) | `agents/sub_agents_nodes.py:57` |
| `tools_used=result["messages"][-1].tool_calls` always logs `[]` — collect tool calls from **all** AI messages | `agents/sub_agents_nodes.py:106` |
| Stale docstring demanding `async with client:` that no caller uses (fine for streamable HTTP — fix the docs, not the code) | `agent_mcp_tools.py:37-39` |
| Deduplicate `assemble_node` (copy-pasted in both graph files) into a shared module | `pipeline_graph.py:17`, `paralel_pipeline_graph.py:38` |
| Replace `print()` debugging in the worker with `log_event` | `agents/sub_agents_nodes.py:83,107-108` |
| ✅ Renamed `graphs/paralel_pipeline_graph.py` → `graphs/parallel_pipeline_graph.py` (root shim kept for one release, item 3.3). Still to do: `utils/senitize.py` → `utils/sanitize.py` | repo root, `utils/` |
| Delete legacy `agents/agent_rouster.json` (superseded by `agent_config.json`) and its README mention | `agents/`, `README.md` |
| Skill frontmatter parsing: `content.split("---")[1]` breaks on files not starting with `---`; use a regex (`^---\n(.*?)\n---`) or the `python-frontmatter` package | `skill_loader.py:16` |

---

## Phase 2 — Developer Experience & Adoption

### 2.1 `.env.example` + friendly config errors

**Problem.** Required env vars (`LLM_URL`, `LLM_MODEL`, `LLM_KEY`,
`CONFIG_PATH`) are only discoverable by grepping for `os.getenv`. If
`CONFIG_PATH` is unset, `config_loader.py:26` crashes with a bare `TypeError`.

**Fix.** Commit a documented `.env.example`:

```bash
# --- LLM endpoint (any OpenAI-compatible server: vLLM, OpenRouter, Ollama) ---
LLM_URL=http://localhost:8000/v1
LLM_MODEL=your-model-name          # must match the server's --model flag
LLM_KEY=not-needed-for-vllm        # any non-empty string works for vLLM

# --- Orchestrator (planner wants low temperature for strict JSON) ---
ORCHESTRATOR_TEMPERATURE=0.1

# --- Paths ---
CONFIG_PATH=agents/agent_config.json

# --- Optional: LangSmith tracing ---
# LANGSMITH_TRACING=true
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
# LANGSMITH_API_KEY=...
# LANGSMITH_PROJECT=agent-skills
```

In `config_loader.py`: default `CONFIG_PATH` to
`Path(__file__).parent / "agents" / "agent_config.json"` and raise a
`FileNotFoundError` that names the path tried and points at `.env.example`.

**Acceptance criteria.** Fresh clone + `cp .env.example .env` + edit 3 lines +
`python run_pipeline.py` works. Unset/missing config produces a one-line
actionable error, not a traceback into `open()`.

### 2.2 Installable package + pinned dependencies

**Problem.** Imports like `from tools import ...` only resolve when running
from the repo root; the venv lives inside the source tree; `requirements.txt`
is mostly unpinned, so a fresh install six months from now breaks on a
LangGraph API change.

**Fix.**
- Add `pyproject.toml` (setuptools or hatchling), declare the dependencies with
  the currently-installed versions as minimums
  (`langgraph>=1.2.4`, `langchain-openai>=1.2.2`, `langchain-mcp-adapters>=0.2.2`,
  `fastapi>=0.137`, ...), and a `[project.optional-dependencies] dev = ["pytest", "ruff"]`.
- `pip install -e .` becomes the documented setup step.
- Keep `requirements.txt` as a lock-style pinned file (`pip freeze` of the
  known-good venv) for Docker reproducibility.
- Note: full `src/` package restructure is **Phase 3** — here we only add
  packaging metadata around the current flat layout.

**Acceptance criteria.** `pip install -e ".[dev]"` in a fresh venv → CLI, API
server, and tests all run from any working directory.

### 2.3 Tests + CI

> **Full specification: [`TESTING_GUIDE.md`](TESTING_GUIDE.md)** — layout,
> conftest design, per-module case tables, code sketches, and the CI workflow.
> This section is the summary; the guide is what a test developer implements
> from.

**Problem.** Zero tests. The most bug-prone logic (routers, dependency
scheduling, plan validation, skill parsing) is pure Python that needs no LLM —
yet the only verification today is a live vLLM endpoint. The cost is already
visible: the current working tree contains two bugs that any of the tests
below would have caught immediately (see "Known bugs" in the guide):

1. `paralel_pipeline_graph.py:57-58` passes the *string* `"scheduler"` to
   `add_conditional_edges`, which requires a callable — the module raises
   `TypeError` at import (verified on langgraph 1.2.4). Those two edges must
   be plain `add_edge` calls — which is precisely the 1.1 fix. Line 59 also
   still uses the `Send` class as a path-map key instead of a plain list.
2. `agents/orchestrator_node.py:85` calls `validate_plan(plan)` but the
   signature is `validate_plan(plan, known_agents, known_skills)` — every run
   raises `TypeError`, masked by the broad `except` into a misleading
   `"Failed to parse JSON response"`.

**Fix.** `tests/` with pytest, in this priority order (regression tests for
the two bugs above land *first*, red, with the fixes in the same PRs):

- `conftest.py` — dummy `LLM_*` env defaults set **before** any pipeline
  import (modules construct `ChatOpenAI` clients and load skills/config at
  import time), CWD pinned to the repo root (`skill_loader.root_dir` is
  CWD-relative), `create_llm.cache_clear()` fixture. The guide's
  "import-time side-effect" table documents every landmine.
- `test_dispatch_dedup.py` — the 1.1 regression: module-imports test (red
  today, bug 1), exactly-once dispatch on the corrected topology built from
  stub nodes + the real `fan_out_router`/`scheduler_node` (parametrized over
  linear, diamond, and wide-fan plans), and the 1.4 failure-containment case
  (which also flags that `failed_steps` is still missing from `AgentState`).
- `test_orchestrator_node.py` — `GenericFakeChatModel` monkeypatched over the
  module-global `llm` (red today, bug 2): valid plan validated and returned,
  fenced JSON, empty plan, non-JSON, unknown agent, injection-flagged task.
- `test_plan_validator.py` — all 1.2 acceptance cases: valid/normalized,
  schema errors, duplicate steps, unknown agent, unknown-skill soft-drop,
  self/dangling/cyclic `depends_on`, `PlanValidationError` is a `ValueError`.
- `test_fan_out_router.py` / `test_should_continue.py` — routing as pure
  functions: layer-by-layer dispatch, `"assemble"` when done; blocked-forever
  guard encoded as `xfail` until the 1.2 guards land.
- `test_skill_loader.py`, `test_json_utils.py`, `test_step_output_validator.py`,
  `test_sanitize.py`, `test_config_loader.py`, `test_llm_factory.py` — pure
  parsing/validation utilities; `sanitize` tests pin the false-positive
  boundary, `config_loader` tests use fresh-import machinery.

Conventions: `pytest.raises(..., match=...)` everywhere (error messages are
contract), `xfail(strict=False)` for desired-but-unimplemented guards, an
`integration` marker (excluded by default) reserved for future live-endpoint
tests, `[tool.pytest.ini_options]` in `pyproject.toml`.

CI: `.github/workflows/ci.yml` — checkout, Python 3.13 with pip cache,
`pip install -e ".[dev]"`, `ruff check .`, `pytest -q` on push/PR. No secrets,
no services; expect an initial batch of ruff findings, fixed in a dedicated
commit.

**Acceptance criteria.** `pytest` passes locally without any network access or
`.env` file; both known-bug regression tests failed before their fixes and
pass after; CI is green on the PR that introduces it; README gains a
"Running the tests" section in the same PR.

### 2.4 Documentation consolidation + contributor recipes

**Problem.** Four overlapping documents (`README.md`,
`multi-agent-pipeline-skills-guide.md`, `langgraph-multi-agent-skills-plan.md`,
`codebase-review-fixes.md`) that have drifted from the code. Contradictory docs
are worse than missing docs.

**Fix.**
- README stays the entry point: quickstart (clone → `.env` → run), architecture
  diagram, and four **recipes**:
  1. *Add a tool* — drop a `@tool`-decorated function in `tools/<name>.py`;
     auto-discovered; assign it in `agent_config.json`.
  2. *Add an agent* — add an entry to `agents/agent_config.json`
     (description, tools, mcp_servers). No code.
  3. *Add a skill* — create `skills/<name>/SKILL.md` with YAML frontmatter
     (`name`, `description`) + body.
  4. *Add a graph* — becomes a clean recipe only after Phase 3; until then,
     document "copy `paralel_pipeline_graph.py`, register in `run_pipeline.py`/
     `api_server.py`".
- Move the guide/plan docs into `docs/history/` with a one-line "design
  history, may be stale" banner.
- Add a short `CLAUDE.md` **in this directory** describing the pipeline
  architecture and conventions — the parent directory's `CLAUDE.md` describes a
  different project (an Angular learning platform), so AI coding tools
  currently receive actively misleading context here.

### 2.5 Logging hygiene

**Problem.** Hardcoded log file name (`utils/logger.py:5`,
`langgraph_smart_reasoning.log`), no level configuration, `print()` mixed in,
and API error responses return full tracebacks to clients
(`api_server.py:141`).

**Fix.** `LOG_LEVEL` / `LOG_FILE` env vars with sane defaults; console handler
for dev; tracebacks in API responses gated behind a `DEBUG` flag (return a
generic message + task id otherwise, full detail stays in the log).

---

## Phase 3 — Architecture Refactor (Graph Factories)

**Goal:** make "create a new graph" a one-file task. Do this *after* the two
live bugs from item 2.3 are fixed (their red `xfail` regression tests must
turn green first), so behavior is pinned before any restructuring.

**Structural decision (2026-07-23).** Exactly **two** nodes get their own
files, both under `agents/`: the orchestrator and the sub-agent worker. All
graph-local glue — scheduler, routers, assemble — is defined **inside the
graph file that uses it**. There is no `nodes/` package.

Rationale: the two LLM-bearing nodes carry all the expensive, reusable logic
(prompt construction, skill injection, tool/MCP wiring, output validation) and
must behave identically across graphs. Glue nodes are the opposite — they *are*
the topology, they're a few lines each, and they legitimately diverge between
graphs (a reflection graph's router is not the parallel graph's router).
Keeping them in the graph file means reading one file shows the complete shape
of that pipeline.

**Promotion rule ("rule of three").** Accept glue duplication between two
graphs (e.g., both current graphs carrying a near-identical `assemble_node` —
it's ~8 lines of presentation logic). Only when a **third** graph needs the
same glue verbatim does it get promoted to a shared module. Never promote
pre-emptively.

**Problem being fixed.** Everything is wired at import time through
module-level singletons: LLM clients, skill index, config, and — worst — the
graphs themselves compile at import with a baked-in `MemorySaver`.
Consequences: callers can't choose a checkpointer (the FastAPI server
accumulates checkpoints in memory forever), tests can't inject fake LLMs, and
the graph files can't be imported at all without full LLM env config.

**Target layout** — this is mostly the current tree; the phase converts
modules to factories rather than moving files:

```
agent_skills/
  llm_factory.py                   # exists (Phase 1.3)
  assemble_node.py                 # DISSOLVES into the graph files (see below)
  agents/
    agent_states.py                # AgentState + WorkerState + PlanStep
    orchestrator_node.py           # make_orchestrator_node(...) factory
    sub_agents_nodes.py            # make_worker_node(...) factory + run_sub_agent_async
  graphs/
    __init__.py                    # GRAPH_REGISTRY + build_graph()
    paralel_pipeline_graph.py      # glue + build() (rename per item 1.5 when convenient)
    sequential_pipeline_graph.py   # glue + build()
```

### 3.1 Node factories for the two shared nodes

Convert each module-level node into a `make_*` factory whose arguments all
default to the current config/env-derived values — so production callers pass
nothing, and tests inject fakes:

```python
# agents/orchestrator_node.py
def make_orchestrator_node(llm=None, agent_roster=None, skill_index=None):
    llm = llm or create_llm(temperature=ORCHESTRATOR_TEMPERATURE)
    roster = agent_roster or AGENT_ROSTER
    index = skill_index or SKILL_INDEX

    def orchestrator_agent(state: AgentState) -> dict:
        ...  # existing logic unchanged, reading llm/roster/index from the closure
    return orchestrator_agent
```

```python
# agents/sub_agents_nodes.py
def make_worker_node(run_step=None):
    """run_step defaults to run_sub_agent_async; tests inject a stub coroutine."""
    run_step = run_step or run_sub_agent_async

    async def worker_node(state: WorkerState) -> dict:
        ...  # existing dispatch + failure-containment logic (Phase 1.4)
    return worker_node
```

Keep a module-level `orchestrator_agent = make_orchestrator_node()` default
instance as an import-compat shim during the transition; delete it once both
graphs and all tests use the factories. `WorkerState` (the `Send` payload
schema: `step`, `results`, `current_datetime`) moves into
`agents/agent_states.py` next to `AgentState` instead of abusing `AgentState`
for worker inputs.

### 3.2 Graph files own their glue + a `build()` factory

Each graph file follows one template — this is also the "add a graph" recipe
for the README (item 2.4):

```python
# graphs/paralel_pipeline_graph.py

# --- graph-local glue: this graph OWNS its scheduler/router/assemble ---

def scheduler_node(state: AgentState) -> dict:
    """No-op synchronization barrier between dependency layers."""
    return {}

def fan_out_router(state: AgentState):
    ...  # ready-step computation; returns "assemble" or [Send(...), ...]

def assemble_node(state: AgentState) -> dict:
    ...  # join step outputs into final_output

# --- factory ---

def build(*, checkpointer=None, orchestrator=None, worker=None):
    orchestrator = orchestrator or make_orchestrator_node()
    worker = worker or make_worker_node()

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", orchestrator,
                     retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)))
    builder.add_node("parallel_sub_agent", worker,
                     retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)))
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("assemble", assemble_node)

    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")           # plain edge
    builder.add_conditional_edges(                          # the ONLY conditional edge
        "scheduler", fan_out_router, ["assemble", "parallel_sub_agent"])
    builder.add_edge("parallel_sub_agent", "scheduler")     # plain edge (deduped)
    builder.add_edge("assemble", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())

# Import-compat shim — existing callers keep working during the transition.
graph = build()
```

The shared `assemble_node.py` at the repo root dissolves into the graph files
(per the structural decision above) and is deleted once nothing imports it.
The topology rules baked into the template are non-negotiable for every future
graph: workers return to the scheduler via **plain `add_edge`**, there is
**one** conditional edge (conditional edges run once per completed task and
`Send` dispatches are not deduplicated — the scheduler is what makes fan-out
safe), and path maps are **plain lists** of node names.

### 3.3 Registry + entry-point changes ✅ done (2026-07-27)

**Implemented as auto-discovery rather than a hardcoded dict** — a hand-written
`GRAPH_REGISTRY` is one more place to forget when adding a graph, which
contradicts the "extension is config/data, not code" convention that already
governs tools, agents and skills. The registry now scans `graphs/`:

- A module in `graphs/` **is** a graph as soon as it defines `build()`. No
  registry line, no decorator, no config entry.
- Name = module name minus a trailing `_pipeline_graph` / `_graph` /
  `_pipeline` suffix (`parallel_pipeline_graph` → `parallel`). A module can
  override it with `GRAPH_NAME` and document itself with `GRAPH_DESCRIPTION`
  (falling back to the first line of its docstring). Modules named `_*` are
  skipped, so shared helpers may live in `graphs/` too.
- **Lazy**: `build_graph("parallel")` imports exactly one module. Only listing
  (`available_graphs()` / `graph_descriptions()`) imports every candidate, and
  it tolerates broken modules — one unimportable graph file no longer takes
  down every entry point. Asking for a broken or `build`-less graph by name
  gets an error naming the file and the real cause.
- `GRAPH_REGISTRY` survives as a lazy read-only `Mapping` of name → `build`
  callable, so the sketch's `GRAPH_REGISTRY[name](checkpointer=...)` still
  works for introspection. `clear_registry_cache()` exists for tests that add
  or remove graph modules at runtime.

Decorator-based registration was considered and rejected: decorators only fire
on import, which forces eager import of every graph module and reintroduces the
one-broken-file-breaks-everything failure mode.

Entry points:

- `api_server.py` accepts an optional `"graph": "parallel"` field in
  `RunRequest` (unknown name → HTTP 400 listing the available graphs), compiles
  each graph once into a process-level cache, and creates the shared
  checkpointer in the lifespan hook (`_make_checkpointer` — the single swap
  point for 4.7). New `GET /graphs` lists what was discovered.
- `run_pipeline.py` gets `--graph` (default `parallel`) and `--list-graphs`;
  `api_client.py` mirrors both.
- The pre-compiled `graph = build()` shims are gone; the root
  `paralel_pipeline_graph.py` (sic) remains as a thin re-export of the renamed
  `graphs/parallel_pipeline_graph.py` for one release.

Also landed here from 1.5: `graphs/paralel_pipeline_graph.py` →
`graphs/parallel_pipeline_graph.py`.

### 3.4 Migration order (each step lands green)

1. `WorkerState` into `agents/agent_states.py`; type the worker node with it.
2. ✅ done — factory-ified `orchestrator_node.py` and `sub_agents_nodes.py`;
   tests switched to injected fakes. The module-level shim instances
   (`orchestrator_agent`, `sub_agent_node`, `parallel_sub_agent_node`) and the
   module-level `llm = create_llm()` are **deleted** (2026-07-27): the LLM is
   now resolved inside `run_sub_agent_async` (overridable via the factories'
   `llm=` argument), so importing `agents`, `graphs` or `api_server` no longer
   requires LLM configuration. The one surviving shim is the root
   `paralel_pipeline_graph.py` re-export, kept for one release.
3. Add `build()` to both graph files; keep `graph = build()` shims. ✅ done —
   except that `assemble_node.py` was *not* dissolved into the graph files: two
   graphs sharing an 8-line node is below the rule-of-three promotion bar, so
   it stays shared until a third graph wants its own.
4. ✅ done (3.3) — registry + API/CLI graph selection; the `graph = build()`
   shims are deleted and `tests/` build graphs through `build(...)`.

**Acceptance criteria.**
- Adding a demo graph (e.g., orchestrate → execute → writer-synthesis) touches
  exactly one new file in `graphs/` plus one registry line — and nothing in
  `agents/`.
- Tests build graphs via `build(orchestrator=fake, worker=fake, checkpointer=...)`
  with no LLM env vars and no network.
- The API can select the graph per request; the CLI per flag.
- No graph file defines LLM logic; `agents/` defines no topology.

---

## Phase 4 — New Features

Ordered roughly by value ÷ effort. Items marked ⛓ depend on Phase 3.

### 4.1 Streaming progress (SSE) — no refactor needed

Expose LangGraph's native streaming over the API so callers see the plan and
each step completion live instead of a silent blocking call:

```python
@app.post("/run-stream")
async def run_stream(req: RunRequest):
    async def event_gen():
        config = {"configurable": {"thread_id": f"api-{uuid.uuid4().hex[:8]}"}}
        async for update in graph.astream(
            {"task": req.task, "current_datetime": get_current_datetime_str()},
            config=config, stream_mode="updates",
        ):
            yield f"data: {json.dumps(jsonable(update))}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")
```

Update `api_client.py` to consume it. This also becomes the transport for any
future web UI.

### 4.2 Human-in-the-loop plan approval — no refactor needed

The checkpointer prerequisite already exists. Compile with
`interrupt_before=["scheduler"]` (post-1.1 topology): the first invoke returns
after planning; the caller inspects/edits `state["plan"]`, optionally
`graph.update_state(config, {"plan": edited})`, then resumes with
`graph.invoke(None, config)`. Expose as `POST /run?approve_plan=true` +
`POST /resume/{thread_id}`. Doubles as a debugging tool for bad plans.

### 4.3 Per-agent LLM configuration

Extend `agent_config.json` with an optional `"llm"` block, resolved by the
Phase 1.3 factory — absent block ⇒ the single env-configured endpoint, so the
current single-vLLM setup needs zero config changes:

```json
"mathematician": {
  "description": "...",
  "tools": ["calculate", "plotting_tool"],
  "mcp_servers": {},
  "llm": {"temperature": 0.2, "max_tokens": 2048,
          "model": "other-model", "url": "http://other:8000/v1",
          "api_key_env": "OTHER_LLM_KEY"}
}
```

Keys in the block are all optional; API keys are referenced by env-var *name*
(`api_key_env`), never stored in the config file.

### 4.4 Multi-turn conversations

`thread_id` plumbing exists end to end. Accept `thread_id` in `RunRequest`; on
follow-up turns, feed the previous `final_output`/`results` to the orchestrator
as context so it plans a delta ("now also plot the derivative") instead of
starting from scratch. Requires a small `AgentState` addition
(`history` or reuse of checkpointed state) and an orchestrator prompt section.

### 4.5 Artifact handling

The plotting tool writes files the API can't return. Give each run an artifacts
directory keyed by thread/task id (pass it via `configurable`), have file-producing
tools write there and return the relative path, and add
`GET /artifacts/{task_id}/{filename}` (FastAPI `FileResponse`).

### 4.6 stdio MCP servers

`create_mcp_client` (`agent_mcp_tools.py:49`) only supports `streamable_http`
URLs. Accept the standard command form too:

```json
"mcp_servers": {
  "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]},
  "yotta_mcp":  {"url": "http://.../mcp"}
}
```

Detect shape (`url` key vs `command` key) and map to
`{"transport": "streamable_http"}` / `{"transport": "stdio"}` respectively.
Keep backward compatibility with the current plain-string URL form.

### 4.7 Durable runs + persistent task store ⛓

Swap `MemorySaver` for `SqliteSaver`/`PostgresSaver` (checkpointer injection
comes from Phase 3) so a crashed run resumes from the last completed step. Back
the API's in-memory `_task_store` (`api_server.py:67`) with the same database —
today a server restart loses all task state, and the store grows unboundedly.

### 4.8 Re-planning / reflection graph ⛓ (also needs 1.1 + 1.2)

The first genuinely new graph, built from Phase 3 parts:

```
orchestrator → scheduler ⇄ workers
                  │ (step done)
                  ▼
              critique  ──accept──▶ scheduler (next layer)
                  │
                  └──revise──▶ orchestrator (re-plan remaining steps)
```

A cheap critique node scores each step output; on failure it either re-dispatches
the step with feedback appended, or sends the whole remaining plan back to the
orchestrator. Cap iterations via a `revision_count` field in state. Without the
1.1 scheduler fix this topology multiplies the duplicate-`Send` bug; without 1.2
a revised plan can silently deadlock — hence the dependencies.

### 4.9 Cost / token tracking ⛓ (wants 1.3)

Aggregate per-step token usage via LangChain callbacks
(`UsageMetadataCallbackHandler` or `response.usage_metadata`), store in state
(`step_stats: Annotated[list[StepStats], operator.add]`), return in the API
response: `steps: [{step, agent, tokens_in, tokens_out, duration_s}]`.

### 4.10 Evaluation harness ⛓

`evals/` with golden tasks (YAML: task, required plan properties, output
assertions, optional LLM-judge rubric) + a runner that builds graphs via the
Phase 3 factory with either the real endpoint or a cheap judge model. This is
what answers "is the new graph actually better?" — the project's stated goal.

### 4.11 Writer-synthesized final output

Replace the mechanical string-join in `assemble_node` with an optional pass
through the writer agent (skill: `answer-writer`), controlled per-graph or per
request (`"synthesize": true`). Falls back to the join when the writer fails.
Since each graph owns its `assemble_node` after Phase 3, this is simply a new
graph (or a variant assemble function inside an existing graph file) — no
shared-node changes needed; possible before Phase 3 as a flag in the existing
graphs.

---

## Suggested Order & Dependencies

```
1.1 scheduler fix ──┐
1.2 plan validation ─┼─▶ 2.3 tests+CI ─▶ Phase 3 refactor ─▶ 4.7 / 4.8 / 4.9 / 4.10
1.3 llm factory ────┘                          ▲
                                               │
2.1 .env.example ── 2.2 packaging ─────────────┘
2.4 docs / 2.5 logging          (any time)
4.1 streaming / 4.2 HITL / 4.3 per-agent LLM / 4.6 stdio MCP   (any time after 1.1–1.3)
```

Rules of thumb:

- **Nothing new lands on top of the duplicate-`Send` topology** — 1.1 first, it
  is a ~20-line change.
- **Phase 3 does not start until 2.3's tests are green**, so the refactor has a
  behavioral safety net.
- DX items (2.1, 2.4, 2.5) are independent and good first issues for a new
  contributor.
- When adding a feature, update the recipe section of the README in the same PR
  — docs drift is how this repo got four stale planning documents.



## Parting notes

One honest caveat about the plan. The duplicate-Send bug (item 1.1) is based on my analysis of how LangGraph evaluates conditional edges per completed task — I didn't get to verify it empirically against your installed langgraph 1.2.4 before we switched to documentation mode. The regression test in item 2.3 (test_dispatch_dedup.py) is designed to settle it either way, and the scheduler-node pattern is the correct topology regardless. I'd make that test the very first thing whoever picks up Phase 1 writes — before the fix, so they see it fail.

Commit the plan. IMPROVEMENT_PLAN.md is untracked on your cleaning-main branch, which also has ~20 uncommitted modified files. If other devs are going to follow this plan, it only exists once it's committed — and the file/line references in it will drift as the code changes, so landing Phase 1 sooner keeps the document accurate.

One security item not in the plan: agents/agent_config.json contains a hardcoded public IP for the yotta MCP server (207.189.105.118:8001). If this repo gets shared beyond your team or made public, that's exposed infrastructure with no auth in front of it, sitting in git history forever. Consider moving MCP URLs to env vars (fits naturally with the 4.6 config-shape change) before widening access.

Small local-only state: I installed pytest into your venv this session — it's not recorded in requirements.txt, so teammates won't get it until item 2.2's dev-dependencies land. Everything else in your working tree is untouched by me except the plan document.

For future sessions: I've saved the key decisions to memory — that IMPROVEMENT_PLAN.md is the active roadmap, your incremental-in-place preference, the undecided LLM-backend/learning-platform questions, and that you want written plans before code changes. So a future session can pick up "implement Phase 1" without re-deriving any of this.

Good luck with the pipeline — the bones here are genuinely solid, and Phase 1 is a small amount of work for a large jump in reliability.