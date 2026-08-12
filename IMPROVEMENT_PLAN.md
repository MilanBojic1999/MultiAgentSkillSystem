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
- [Phase 4 — Baseline Hardening & New Features](#phase-4--baseline-hardening--new-features)
  - [Tier 1 — the baseline must be sound](#tier-1--the-baseline-must-be-sound)
  - [Tier 2 — prove the extension points work](#tier-2--prove-the-extension-points-work)
  - [Tier 3 — capability demonstrations](#tier-3--capability-demonstrations)
  - [Tier 4 — deferred](#tier-4--deferred)
  - [Scope statement](#scope-statement--decide-before-advertising-the-baseline)
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
| ✅ Renamed `graphs/paralel_pipeline_graph.py` → `graphs/parallel_pipeline_graph.py` (root shim kept for one release, item 3.3). Still to do: `utils/sanitize.py` → `utils/sanitize.py` | repo root, `utils/` |
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

## Phase 4 — Baseline Hardening & New Features

> **Reframed 2026-07-27.** The destination for this project is now decided: it is
> a **baseline other developers fork and extend** — "add a graph, agent, skill,
> tool, MCP server or model with the least amount of pain." That changes how
> Phase 4 is prioritised:
>
> 1. **Extension points that are still code changes outrank user-facing
>    features.** 4.3 and 4.6 are the last two holes in the extension matrix, so
>    they move ahead of streaming and HITL.
> 2. **Every reference implementation must actually run.** A template whose
>    example graph doesn't execute costs more than any missing feature — hence
>    4.12 at the top.
> 3. **Application-level concerns get demoted.** 4.4 and 4.7 are decisions a
>    fork makes, not defaults a baseline should bake in.
>
> Item numbers are **unchanged** from the previous revision — `api_server.py:104`
> cites 4.7 and `llm_factory.py:27` cites 4.3. New items continue from 4.12.

### The extension matrix — what this phase is graded against

| Extension point | Mechanism | State |
|---|---|---|
| Tool | drop a `@tool` function in `tools/*.py` | ✅ auto-discovered, recipe documented |
| Agent | entry in `agents/agent_config.json` | ✅ no code |
| Skill | `skills/<name>/SKILL.md` | ✅ no code |
| Graph | module in `graphs/` defining `build()` | ✅ auto-discovered, lazy, fault-tolerant |
| **MCP server** | `mcp_servers` block in the agent config | ❌ `streamable_http` URLs only; a public IP hardcoded (4.6) |
| **Model per agent** | — | ❌ one global env-configured endpoint (4.3) |
| **State schema** | — | ❌ hardwired to plan-and-execute (see *Scope statement*) |

### Tiers

| Tier | Items | Gate to leave the tier |
|---|---|---|
| **1 — the baseline must be sound** | 4.12, 4.13, 4.3, 4.6, 4.14, 4.15, 4.18 | Both shipped graphs run end to end; no extension point requires editing code |
| **2 — prove the extension points work** | 4.16, 4.17, 4.10, 4.9 | A deliberately broken extension turns exactly one test red |
| **3 — capability demonstrations** | 4.8, 4.11, 4.1, 4.2, 4.5 | A third graph exists that was built without touching `agents/` (verification target; reassessed 2026-08-12 as optional for fork usability — see the Tier 3 section intro) |
| **4 — deferred** | 4.4, 4.7 | Revisit only when a concrete consumer demands them |

---

## Tier 1 — the baseline must be sound

### 4.12 The sequential graph has never executed a step ⚠️ highest priority

**Problem.** `make_sub_agent_node` is declared `async def` but calls
`asyncio.run(run_step(...))` (`agents/sub_agents_nodes.py:147`). Verified
empirically against the current tree:

- **`ainvoke`** — used by `run_pipeline.run_async`, `api_server._run_pipeline`
  and every async test →
  `RuntimeError: asyncio.run() cannot be called from a running event loop`
- **`invoke`** — used by `run_pipeline.run` →
  `TypeError: No synchronous function provided to "sub_agent". Either
  initialize with a synchronous function or invoke via the async API`

So `--graph sequential` and `{"graph": "sequential"}` fail on the first step, on
every entry path. The fix is one word (`await`); the cost of having shipped it
is not.

**Why it outranks everything else.** `graphs/sequential_pipeline_graph.py` is
the file README Recipe 4 tells a new contributor to copy when writing their own
graph. It is *the template*, and it does not run.

**Why the tests missed it.** `tests/test_should_continue.py:30` is the only test
that invokes the node, and it is `xfail(strict=False)` for an unrelated reason
(the missing 1.2 blocked-forever guard). The `RuntimeError` is absorbed as an
expected failure. `xfail(strict=False)` is for behaviour that is *specified but
unimplemented* — using it on a code path that has never been proven to work
converts a coverage hole into a green test.

**Fix.**
- `await run_step(...)` instead of `asyncio.run(run_step(...))`.
- Give the sequential worker the same failure containment as the parallel one
  (4.13), so the two reference topologies behave identically.
- Add the 1.2 blocked-forever guard: raise `RuntimeError` naming the blocked
  steps instead of returning `{}`. Returning `{}` makes `should_continue` loop
  forever, because `len(results)` never grows.
- Add `RetryPolicy(max_attempts=2, retry_on=(ValueError,))` to the sequential
  graph's orchestrator node — the parallel graph has it, this one doesn't
  (`graphs/sequential_pipeline_graph.py:36`).
- Replace the `xfail` in `test_should_continue.py` with a real assertion.

**Acceptance criteria.** `build_graph("sequential")` executes a 3-step linear
plan end to end under **both** `invoke` and `ainvoke` with a stub `run_step`
(mirroring `tests/test_dispatch_dedup.py`); no `xfail` remains in
`test_should_continue.py`; a plan whose remaining steps are all blocked raises
`RuntimeError` naming them.

---

### 4.13 Finish Phase 1.4 failure containment

**Problem.** 1.4 landed one third of its design.
`make_parallel_sub_agent_node` catches, logs and records `[STEP FAILED] …` plus
`failed_steps` — but the two consumers of that information were never written:

- `fan_out_router` (`graphs/parallel_pipeline_graph.py:30-34`) only tests
  `d in results`. A failed step **is** in `results`, so its dependents are
  considered ready and run anyway, receiving the failure text as "Upstream
  context". One failure silently degrades every downstream step instead of
  stopping that branch — and pays for the LLM calls.
- `assemble_node` (`assemble_node.py`) emits no warning header, so failed and
  successful steps are presented identically in `final_output`.
- The sequential worker has no containment at all (see 4.12).

**Fix.** Compute the blocked set transitively and write skip markers from the
scheduler node — which currently does nothing and already runs exactly once per
layer, which is precisely when this propagation must happen:

```python
def scheduler_node(state: AgentState) -> dict:
    """Synchronisation barrier: also propagates skips from failed steps."""
    blocked = _transitive_dependents(state["plan"], set(state.get("failed_steps", [])))
    results = state.get("results", {})
    return {"results": {s: "[SKIPPED — dependency failed]"
                        for s in blocked if s not in results}}
```

`fan_out_router` then needs no change: a skipped step is in `results`, so it is
never dispatched, and its own dependents are already in `blocked`.

`assemble_node` prepends when `failed_steps` is non-empty:

```
> ⚠️ 2 of 5 steps failed (steps 3, 4). The output below is partial.
```

**Design note.** This is the one responsibility the no-op scheduler can take on
without becoming topology-specific — it is a property of *dependency layers*,
not of this particular fan-out strategy. Keep the node-level `RetryPolicy` as
the transient-error net; containment is the final fallback after retries.

**Acceptance criteria.** Extend the containment case in
`tests/test_dispatch_dedup.py`: a diamond plan whose step 1 always throws
produces a `final_output` containing step 2's real result, `[STEP FAILED]` for
step 1, `[SKIPPED — dependency failed]` for step 3, the warning header, and
**no worker invocation for step 3**.

### 4.3 Per-agent LLM configuration ⬆ promoted to Tier 1

**Status.** Half the plumbing already exists: `llm_factory.create_llm(model,
url, api_key_env, temperature, max_tokens)` accepts every parameter and caches
per exact tuple. What is missing is the *read* side —
`run_sub_agent_async` (`agents/sub_agents_nodes.py:49`) calls a bare
`create_llm()` and ignores the agent's config.

**Why Tier 1 now.** "Point one agent at a different model" is the first thing a
fork will want, and today it is a code edit. It is one of the two remaining
holes in the extension matrix.

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

Every key in the block is optional; an absent block means the single
env-configured endpoint, so existing configs need zero changes. API keys are
referenced by env-var **name** (`api_key_env`) and never stored in the config.

**Fix.** In `run_sub_agent_async`, when no explicit `llm=` was injected:

```python
llm = llm or create_llm(**_llm_kwargs(AGENT_CONFIG.get(agent_name, {}).get("llm", {})))
```

`_llm_kwargs` whitelists the five accepted keys and raises `ValueError` naming
the agent and the offending key otherwise. Validate the block in
`config_loader.py` at import time — a typo in `agent_config.json` must fail
loudly at startup, not silently fall back to the default endpoint mid-run.

**Acceptance criteria.** An agent with `"llm": {"temperature": 0.2}` gets a
distinct cached client; an agent with no block gets the env default; an unknown
key inside `"llm"` raises at config load naming agent + key;
`tests/test_config_loader.py` and `tests/test_llm_factory.py` cover both paths.

---

### 4.6 MCP transport shapes + env-var URLs ⬆ promoted to Tier 1

**Problem — two, in the same file.**

1. `create_mcp_client` (`agent_mcp_tools.py:50-55`) hardcodes
   `{"transport": "streamable_http"}`, so only URL servers work. **stdio** —
   the transport most published MCP servers ship with — is unreachable.
2. `agents/agent_config.json` contains a literal public IP
   (`http://207.189.105.118:8001/mcp`) with no auth in front of it, committed to
   git history. For a repo intended to be forked and shared, that is both the
   wrong default and an exposure: a fork inherits it without deciding to.

**Fix.** Accept three config shapes, detected by key:

```json
"mcp_servers": {
  "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]},
  "yotta_mcp":  {"url": "${YOTTA_MCP_URL}"},
  "legacy":     "http://host:8001/mcp"
}
```

- dict containing `url` → `{"transport": "streamable_http"}`
- dict containing `command` → `{"transport": "stdio"}` (with `args`, `env`)
- plain string → current behaviour, kept for backward compatibility
- `${VAR}` anywhere in a URL is expanded from the environment; a missing
  variable raises, naming the agent, the server and the variable

Move the yotta URL to `YOTTA_MCP_URL` and document it in `.env.example`. The IP
remains in git history — rotating or firewalling that host is a separate,
non-code decision and is out of scope here.

**Acceptance criteria.** New `tests/test_mcp_client.py`: each of the three
shapes maps to the correct transport dict; `${VAR}` expands; a missing variable
raises naming agent/server/variable; the existing ownership check still fires.
No live MCP server required — assert on the dict passed to
`MultiServerMCPClient`, which means extracting it into a testable
`_build_server_map()` helper.

---

### 4.14 Structured plan output + re-plan with feedback

*(This was the optional "stretch" under item 1.3; it never landed, and it is
the largest remaining reliability lever.)*

**Problem.** Plan generation is still the biggest source of failed runs, and
all three mitigations are absent:

1. `_extract_json` regex-scrapes JSON out of free text (`utils/json_utils.py`).
   The backend is vLLM, which supports guided decoding — this fragility is
   optional.
2. `RetryPolicy(retry_on=(ValueError,))` on the orchestrator re-invokes with
   **identical messages**, at `temperature=0.1`, against an `lru_cache`d
   client. The retry re-rolls the dice without telling the model what was
   wrong, so it very often reproduces the same bad plan.
3. `agents/orchestrator_node.py:92-93` wraps *every* exception — including
   `PlanValidationError` — into `"Failed to parse JSON response: {e}"`, the
   most misleading possible message for a semantic validation failure.

**Fix.**

```python
def _plan_once(messages):
    if STRUCTURED_OUTPUT:                      # env-gated, default on
        return llm.with_structured_output(PlanModel).invoke(messages)
    return extract_json(llm.invoke(messages).content)

for attempt in range(MAX_PLAN_ATTEMPTS):       # default 2
    try:
        return validate_plan(_plan_once(messages), set(roster), set(index))
    except (PlanValidationError, ValueError) as e:
        log_event("orchestrator_replan", attempt=attempt, error=str(e))
        messages += [AIMessage(content=raw),
                     HumanMessage(content=REPLAN_PROMPT.format(error=e))]
raise PlanValidationError(...)                 # exhausted → node RetryPolicy
```

`PlanModel` is a thin wrapper (`plan: list[PlanStepModel]`) around the model
that already exists in `utils/plan_validator.py`. Keep `_extract_json` as the
fallback for backends without guided decoding (`STRUCTURED_OUTPUT=false`), and
stop swallowing `PlanValidationError` into the JSON-parse message.

**Interaction with the node `RetryPolicy`.** In-node re-planning handles
*semantic* failures with feedback; the node-level retry stays as the outer net
for transport errors. Cap in-node attempts at 2 so the two loops don't multiply
into four LLM calls.

**Acceptance criteria.** `tests/test_orchestrator_node.py` gains: a fake model
returning an invalid plan then a valid one yields the valid plan within a single
node call and logs `orchestrator_replan`; the feedback message contains the
validator's error text; two consecutive invalid plans raise
`PlanValidationError` (not `"Failed to parse JSON response"`);
`STRUCTURED_OUTPUT=false` still works through `_extract_json`.

---

### 4.15 Drop `run_bash` from the shipped baseline

**Decision (2026-07-27).** Remove the shell tool entirely rather than sandbox
it.

**Problem.** `tools/bash_tool.py` gives `shell=True` execution to both
`mathematician` and `researcher` (`agents/agent_config.json`), driven by
free-form natural language arriving at `POST /run`, behind
`allow_origins=["*"]` and no authentication. The privilege drop only fires when
running as root. For a repo meant to be forked, shipping remote code execution
as a **default** is the wrong starting posture — a fork inherits it without
ever deciding to.

**Fix.**
- Delete `tools/bash_tool.py`; remove `"run_bash"` from both agents in
  `agents/agent_config.json`.
- **`skills/roll-dice/SKILL.md` depends on it** — its body is
  `echo $((RANDOM % <sides> + 1))`. Replace it with a native `tools/dice.py`
  `@tool`, which is a better Recipe 1 demonstration anyway: it exercises the
  tool extension point instead of shelling out of it.
- `run_bash_with_approval` goes with it — it calls `input()` and can never work
  under the API server or inside a graph run.
- README: document the removal and the reasoning under Recipe 1, so a fork that
  genuinely wants shell access adds it deliberately.

**Not in scope here.** CORS policy and API authentication are deployment
decisions a baseline cannot make for its forks. Note them in the README's
security section instead of guessing.

**Acceptance criteria.** No `subprocess` call is reachable from a default
agent; `roll-dice` works through the native tool; the conformance suite (4.16)
asserts every tool named in `agent_config.json` resolves.

---

### 4.18 Template hygiene

Small, but it is the first impression a fork gets:

- `utils/sanitize.py` → `utils/sanitize.py` — the last unchecked row of item
  1.5. A misspelled module in a template gets copied forward forever.
- Delete the root `paralel_pipeline_graph.py` shim. It protects downstream
  importers who do not exist; item 3.3 gave it "one release", and this is that
  release. **Deleting the shim without also removing `paralel_pipeline_graph`
  from `py-modules` in `pyproject.toml` (and the README note describing the
  shim) breaks `pip install -e .` — verified 2026-08-12; this one line is a
  hard prerequisite before the baseline is advertised** (see the Tier 3
  reassessment). ✅ done (2026-08-12) — shim file, `py-modules` entry and
  README note all removed.
- Confirm `old_agent.py`, `testing_main.py`, `debug_log.log` (currently ~300 MB
  in the working tree), `scraped_pages/` and `artifacts/` are **untracked** as
  well as gitignored — a `.gitignore` entry added after a file was committed
  does not remove it from the repo.
- Pin `langgraph` with an upper bound in `pyproject.toml`. A baseline that
  silently breaks on LangGraph 2.0 fails exactly the people it exists for.

---

## Tier 2 — prove the extension points work

### 4.16 Extension-point conformance tests

**Problem.** All five extension points auto-discover, which means a broken
extension fails at **runtime**, deep inside a graph, on the machine of whoever
forked the repo. Nothing asserts that the currently-registered inventory is
well-formed. `tests/test_graph_registry.py` tests the registry *mechanism*
using synthetic probe modules — no test looks at the real graphs, agents,
skills or tools.

**Fix.** One module, parametrized over the live registries, so it grows
automatically as a fork adds extensions:

```python
@pytest.mark.parametrize("name", sorted(available_graphs()))
def test_every_graph_compiles(name):
    build_graph(name, orchestrator=fake_orchestrator, sub_agent=fake_worker,
                checkpointer=MemorySaver())

@pytest.mark.parametrize("agent", sorted(AGENT_CONFIG))
def test_every_agent_resolves(agent):
    # description non-empty; every tool name in TOOL_REGISTRY;
    # every mcp_servers entry a recognised shape; any "llm" block valid (4.3)

@pytest.mark.parametrize("skill", sorted(load_skills()[0]))
def test_every_skill_parses(skill):
    # frontmatter has name + description; name matches its directory; body non-empty

@pytest.mark.parametrize("tool", sorted(TOOL_REGISTRY))
def test_every_tool_is_usable(tool):
    # non-empty description; JSON-serialisable args_schema
```

**Why this is Tier 2 and not optional.** For a baseline, this file *is* the
extension contract. It is also the cheapest possible defence against the
failure mode this repo has already hit twice: a shipped module that doesn't
import (item 2.3, bug 1) or doesn't run (4.12).

**Acceptance criteria.** Adding a deliberately broken skill, agent, tool or
graph turns exactly one parametrized case red, with a message naming the
offender. README documents it as the command to run after adding an extension.

---

### 4.17 `scaffold` command ✅ done (2026-08-06)

**Fix.** `python -m scaffold <kind> <name>` for `graph | agent | skill | tool`,
emitting a working, immediately-testable stub:

- `graph` — the Phase 3.2 template with the non-negotiable topology rules
  already correct (plain edge back to the scheduler, exactly one conditional
  edge, list path map), plus `GRAPH_DESCRIPTION`
- `tool` — a `@tool` function with a docstring and a typed signature
- `skill` — `SKILL.md` with valid frontmatter
- `agent` — appends a validated entry to `agents/agent_config.json`

Refuses to overwrite an existing file; prints the next step
(`pytest tests/test_extension_contracts.py`) after writing.

**Acceptance criteria.** For each kind, scaffolding into a temporary tree
produces something that passes 4.16 unmodified. Roughly 150 lines plus
templates; no new dependency.

**Implementation.** `scaffold.py` (290 lines including templates), 49 tests
in `tests/test_scaffold.py`, registered in `pyproject.toml`. All four kinds
pass `test_extension_contracts.py`.

---

### 4.10 Minimal evaluation harness ⬇ scoped down

**Revised scope.** The original item (YAML golden tasks + output assertions +
LLM-judge rubric) is a project in itself. Build the half that needs no judge
first: **plan-shape assertions only**, in `evals/tasks/*.yaml`:

```yaml
task: "Calculate the maximum of x^2*sin(x) on [0,2], then summarise it"
expect:
  min_steps: 2
  max_steps: 6
  agents_include: [mathematician, writer]
  has_dependency: true        # at least one step depends_on another
```

The runner builds a graph with the **real orchestrator** and a **stub worker**,
so a full sweep costs one LLM call per task and executes no tools. Output
quality assertions and the judge model are a later increment.

**Why it matters more under the baseline framing.** This is the harness a
downstream developer points at *their* orchestrator prompt or *their* graph to
check they haven't regressed. It is also the only way to answer "is 4.8
actually better?"

**Acceptance criteria.** `python -m evals` runs every task against `--graph`,
prints a pass/fail table and exits non-zero on failure; carries the
`integration` marker so it stays out of the default `pytest` run.

---

### 4.9 Per-step stats

Design unchanged: aggregate usage via LangChain callbacks
(`UsageMetadataCallbackHandler` / `response.usage_metadata`), accumulate into
`step_stats: Annotated[list[StepStats], operator.add]`, return in the API
response.

**One revision.** On a self-hosted vLLM the interesting numbers are **latency
and tool-call counts**, not cost. Record `duration_s` and `tool_calls`
alongside tokens — `run_sub_agent_async` already collects the tool calls
(`agents/sub_agents_nodes.py:105-109`) and currently throws them away after
logging them.

Pairs with 4.10: a plan-shape eval that also reports token and latency deltas
answers "better?" quantitatively rather than by impression.

---

## Tier 3 — capability demonstrations

> **Necessity reassessment (2026-08-12).** None of these items is required for
> the baseline's purpose — a developer forking and extending the repo. Tier 3
> exists to *prove* the extension points (confidence and advertising), not to
> *enable* them; the scaffold (4.17) and conformance tests (4.16) already cover
> extension mechanically. Two things outside this tier do block a fork today,
> and both are effectively Tier 1 work:
>
> 1. `pyproject.toml` still lists the deleted `paralel_pipeline_graph` shim in
>    `py-modules` → the README's first documented command,
>    `pip install -e ".[dev]"`, fails. One-line fix; hard prerequisite before
>    advertising the baseline (tracked under 4.18). ✅ done (2026-08-12)
> 2. 4.5 (plot-path collision) is a silent wrong-output defect in a default
>    tool, not a feature — by the plan's own rule ("a reference implementation
>    that does not run is worse than a missing feature") it belongs in Tier 1.
>    ✅ done (2026-08-12)
>
> Ranked by necessity for a fork developer:
>
> | Item | Needed by a fork? | Honest framing |
> |---|---|---|
> | 4.5 artifacts | **Yes** | ✅ fixed 2026-08-12 — per-run artifact dirs, unique filenames, `GET /artifacts` |
> | 4.1 streaming | only once a UI consumer exists | ~15-line transport add-on per the sketch |
> | 4.2 HITL | no | dev-experience nicety / bad-plan debugger |
> | 4.8 reflection graph | no | proof of the one-file-graph claim, not capability |
> | 4.11 synthesis graph | no | the learning platform's Writing-agent feature, not a baseline requirement |
>
> Conclusion: Tier 3 can wait for concrete consumers (4.1 when the UI exists,
> 4.11 for the platform); its gate remains the verification target for when it
> resumes. The scope statement below is the zero-cost item that should land
> before the baseline is advertised.

### 4.8 Re-planning / reflection graph — reframed as Phase 3's acceptance test

Design unchanged:

```
orchestrator → scheduler ⇄ workers
                  │ (layer done)
                  ▼
              critique  ──accept──▶ scheduler (next layer)
                  │
                  └──revise──▶ orchestrator (re-plan remaining steps)
```

**What changes is why it is worth building.** Phase 3's acceptance criterion
says "adding a graph touches exactly one new file in `graphs/` and nothing in
`agents/`". That claim is currently **unverified** — both shipped graphs predate
the refactor. 4.8 is the test of it.

**Added constraint.** If building this requires touching `agents/`, that is a
finding to record in this document — not something to quietly work around.

Keep it small: a critique node scoring the last completed layer, a
`revision_count` cap in state, one route back to the orchestrator. Depends on
4.13 (a rejected step and a failed step need the same blocked-dependents
machinery) and benefits from 4.10 and 4.14 (the re-plan-with-feedback prompt is
the same idea at node scale).

---

### 4.11 Writer-synthesized output — as a variant graph, not a flag

Original design stands, with one revision: build it as
`graphs/synthesis_pipeline_graph.py` carrying its own `assemble_node`, rather
than as a `"synthesize": true` flag on the existing graphs. Three reasons: it
keeps the parallel graph's assemble a pure function; it exercises the "each
graph owns its glue" decision from Phase 3; and it is the **third** graph, which
is exactly what the rule-of-three says should finally settle whether the shared
`assemble_node.py` gets promoted or dissolved.

Note what this deliberately does *not* change: the baseline's default
`final_output` stays a mechanical `## Step N:` join. That is the right default
for a framework — predictable, no extra LLM call, no extra failure mode. The
synthesis graph is the demonstration of how a fork does better.

---

### 4.1 Streaming progress (SSE) — keep, with a correction

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

**Correction to the sketch.** `stream_mode="updates"` yields **node-level**
events only: the plan, then one event per completed step. Token-level streaming
from sub-agents additionally requires `stream_mode="messages"` **and**
`subgraphs=True`, because `create_react_agent` compiles to a subgraph whose
internals are invisible to the parent stream by default. Decide which is being
shipped — `"updates"` is enough for a progress UI; `"messages"` is what a chat
UI needs.

Also: `_get_graph` caches compiled graphs per process, so `/run-stream` shares
the checkpointer with every other endpoint — give each stream its own
`thread_id`, exactly as `_run_pipeline` does. Update `api_client.py` to consume
it; this also becomes the transport for any future web UI.

---

### 4.2 Human-in-the-loop plan approval — keep, with the interrupt point corrected

**The previous sketch was wrong.** `interrupt_before=["scheduler"]` interrupts
before *every* layer, not just after planning — the scheduler is re-entered once
per superstep by design; that is the entire point of item 1.1. Use
`interrupt_after=["orchestrator"]`, which fires exactly once.

The rest stands: the checkpointer prerequisite already exists, the first invoke
returns after planning, the caller inspects/edits `state["plan"]`, optionally
calls `graph.update_state(config, {"plan": edited})`, then resumes with
`graph.invoke(None, config)`. Expose as `POST /run?approve_plan=true` +
`POST /resume/{thread_id}`. Doubles as a debugging tool for bad plans.

---

### 4.5 Artifact handling — a live defect, then a convention ⬆ effectively Tier 1 ✅ done (2026-08-12)

**It is a bug, not just a missing feature.** `plotting_tool`
(`tools/plotting.py:78`) writes a fixed `artifacts/plot.png`. Under the parallel
graph, two plotting steps in the same layer overwrite each other and both return
the same path — nondeterministic output, silently.

**Fix, in two parts.**

1. Per-run artifacts directory keyed by thread/task id, passed via
   `configurable`; file-producing tools write there and return the relative
   path; add `GET /artifacts/{task_id}/{filename}` (FastAPI `FileResponse`).
2. Because this is a baseline: make it a **documented convention** every
   file-producing tool follows — `get_artifact_path(name) -> Path`, resolved
   from the run context — and add it to README Recipe 1. Otherwise every tool a
   fork adds reinvents the same collision.

---

## Tier 4 — deferred

### 4.4 Multi-turn conversations — deferred

An application-level concern, not a baseline one. The pieces a fork needs
(`thread_id` plumbing, a checkpointer seam) already exist; the pieces that don't
(context-window budgeting, history truncation policy, orchestrator
delta-planning prompts) are product decisions this repo cannot make on a fork's
behalf. Revisit if the baseline ever gains a reference UI.

### 4.7 Durable runs — collapsed to a documented seam

`_make_checkpointer` (`api_server.py:112`) is already the single swap point,
which was the actual deliverable of this item. What remains is small and mostly
documentation:

- Ship one **`AsyncSqliteSaver`** example — async, because every API path uses
  `ainvoke`; the synchronous `SqliteSaver` will not work there.
- Document the seam and the `.checkpoints/` gitignore entry that already exists.
- Backing `_task_store` with the same database stays deferred: it is a
  deployment concern, and an unbounded in-memory dict is an acceptable baseline
  default **as long as it is documented as one**.

---

## Scope statement — decide before advertising the baseline

`AgentState` (`agents/agent_states.py:20`) hardwires `plan` / `results` /
`current_step` / `final_output`, and both shared nodes require that schema. A
fork building a supervisor loop, a plain ReAct chat agent or an
evaluator-optimizer can reuse the graph registry, the tool/skill/config loaders
and `llm_factory` — but **not** `agents/`. So the honest description today is:

> a baseline for **plan-and-execute** agentic systems

Two options, and this document should commit to one:

- **(a) Narrow the claim.** Free. The plan-and-execute niche is well covered by
  what already exists, and the fixed state schema is exactly what lets the two
  shared nodes stay reusable across graphs.
- **(b) Generalise.** Make `AgentState` a protocol/generic, let each graph
  declare its own state, and reduce `agents/` to nodes parameterised over a
  state contract. This is real work and it is **Phase 5** — not something to
  smuggle into a Phase 4 item.

**Recommendation: (a) now, (b) only when a concrete second topology family
actually demands it.**

---

## Suggested Order & Dependencies

Phases 1–3 are complete (Phase 1.5's `sanitize` rename carries over into 4.18).
What remains:

```
Tier 1 — the baseline must be sound
  4.12 sequential graph fix ──▶ 4.13 failure containment ──┐
  4.14 structured output + replan ─────────────────────────┼──▶ Tier 2
  4.3 per-agent LLM / 4.6 MCP shapes / 4.15 drop run_bash ─┘
        4.18 template hygiene   (any time)

Tier 2 — prove the extension points work
  4.16 conformance tests ──▶ 4.17 scaffold
  4.10 minimal evals ──▶ 4.9 step stats

Tier 3 — capability demonstrations
  4.8 reflection graph      (needs 4.13; wants 4.10, 4.14)
  4.11 synthesis graph      (the third graph — settles assemble_node's fate)
  4.1 streaming / 4.2 HITL / 4.5 artifacts

**Reassessed 2026-08-12:** Tier 3 is optional for the baseline's
fork-and-extend purpose (see the section intro). The two items that actually
block a fork developer today are the `pyproject.toml` shim entry (4.18) and
the plot-path collision (4.5) — pull both ahead of Tier 3; defer the rest
until concrete consumers demand them (4.1: a UI; 4.11: the learning platform).
✅ Both landed 2026-08-12 (see 4.5 and 4.18).

Tier 4 — deferred
  4.4 multi-turn, 4.7 durable task store
```

Rules of thumb:

- **A reference implementation that does not run is worse than a missing
  feature.** 4.12 lands before anything else in Phase 4.
- **Extension holes outrank features.** 4.3 and 4.6 are the last two places
  where extending the baseline means editing code — they come before streaming,
  HITL and artifacts.
- **Every new extension ships with its conformance case** (4.16) and its README
  recipe, in the same PR.
- `xfail(strict=False)` is for behaviour that is *specified but unimplemented* —
  never for a code path that has never been proven to work. That is how 4.12
  stayed hidden.
- When adding a feature, update the recipe section of the README in the same PR
  — docs drift is how this repo got four stale planning documents.



## Parting notes

One honest caveat about the plan. The duplicate-Send bug (item 1.1) is based on my analysis of how LangGraph evaluates conditional edges per completed task — I didn't get to verify it empirically against your installed langgraph 1.2.4 before we switched to documentation mode. The regression test in item 2.3 (test_dispatch_dedup.py) is designed to settle it either way, and the scheduler-node pattern is the correct topology regardless. I'd make that test the very first thing whoever picks up Phase 1 writes — before the fix, so they see it fail.

Commit the plan. IMPROVEMENT_PLAN.md is untracked on your cleaning-main branch, which also has ~20 uncommitted modified files. If other devs are going to follow this plan, it only exists once it's committed — and the file/line references in it will drift as the code changes, so landing Phase 1 sooner keeps the document accurate.

One security item not in the plan: agents/agent_config.json contains a hardcoded public IP for the yotta MCP server (207.189.105.118:8001). If this repo gets shared beyond your team or made public, that's exposed infrastructure with no auth in front of it, sitting in git history forever. Consider moving MCP URLs to env vars (fits naturally with the 4.6 config-shape change) before widening access.

Small local-only state: I installed pytest into your venv this session — it's not recorded in requirements.txt, so teammates won't get it until item 2.2's dev-dependencies land. Everything else in your working tree is untouched by me except the plan document.

For future sessions: I've saved the key decisions to memory — that IMPROVEMENT_PLAN.md is the active roadmap, your incremental-in-place preference, the undecided LLM-backend/learning-platform questions, and that you want written plans before code changes. So a future session can pick up "implement Phase 1" without re-deriving any of this.

Good luck with the pipeline — the bones here are genuinely solid, and Phase 1 is a small amount of work for a large jump in reliability.