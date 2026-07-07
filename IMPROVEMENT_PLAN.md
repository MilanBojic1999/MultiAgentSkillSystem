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
| Rename `paralel_pipeline_graph.py` → `parallel_pipeline_graph.py` and `utils/senitize.py` → `utils/sanitize.py` (keep import-shim modules for one release: `from parallel_pipeline_graph import *`) | repo root, `utils/` |
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

**Problem.** Zero tests. The most bug-prone logic (routers, dependency
scheduling, plan validation, skill parsing) is pure Python that needs no LLM —
yet the only verification today is a live vLLM endpoint.

**Fix.** `tests/` with pytest:

- `conftest.py` — set dummy `LLM_*` env defaults (`os.environ.setdefault`)
  *before* pipeline imports, so module import doesn't require a real endpoint.
- `test_plan_validator.py` — all cases from 1.2.
- `test_fan_out_router.py` — given a plan + partial results, asserts which
  steps are dispatched / when `"assemble"` is returned / that blocked-forever
  states raise.
- `test_dispatch_dedup.py` — the 1.1 regression test: real topology + stub
  worker node that counts invocations; asserts exactly-once execution per step.
- `test_skill_loader.py` — frontmatter/body parsing against a tmp skills dir.

CI: GitHub Actions workflow — `ruff check .` + `pytest` on push/PR.

**Acceptance criteria.** `pytest` passes locally without any network access or
`.env` file; CI is green on the PR that introduces it.

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

**Goal:** make "create a new graph" a one-file task. Do this *after* Phase 1
lands and is covered by tests, so behavior is pinned before the move.

**Problem.** Everything is wired at import time through module-level
singletons: LLM clients, skill index, config, and — worst — the graphs
themselves compile at import with a baked-in `MemorySaver`. Consequences:
callers can't choose a checkpointer (the FastAPI server accumulates
checkpoints in memory forever), tests can't inject fake LLMs, and a new graph
means copy-pasting node wiring.

**Target layout** (rename-heavy; do it as one focused PR):

```
agent_pipeline/
  config.py          # env + paths, one place
  llm_factory.py     # create_llm(...) — from Phase 1.3
  states.py          # AgentState, WorkerState, PlanStep
  planning.py        # plan validation (from utils/plan_validator.py)
  nodes/
    orchestrator.py  # make_orchestrator_node(llm, roster, skill_index) -> node fn
    worker.py        # make_worker_node(llm_for_agent, tools, skills) -> node fn
    scheduler.py     # scheduler_node, fan_out_router
    assemble.py      # join-based and writer-based variants
  graphs/
    __init__.py      # GRAPH_REGISTRY: dict[str, GraphBuilder]
    parallel.py      # build(config, checkpointer=None) -> CompiledGraph
    sequential.py
```

**Key sketches.**

```python
# graphs/__init__.py
GRAPH_REGISTRY = {"parallel": parallel.build, "sequential": sequential.build}

def build_graph(name: str, *, checkpointer=None, **overrides):
    try:
        return GRAPH_REGISTRY[name](checkpointer=checkpointer, **overrides)
    except KeyError:
        raise ValueError(f"Unknown graph '{name}'. Available: {sorted(GRAPH_REGISTRY)}")
```

```python
# nodes/orchestrator.py — dependency injection instead of module globals
def make_orchestrator_node(llm, agent_roster, skill_index):
    def orchestrator_node(state: AgentState) -> dict:
        ...  # same logic, but llm/roster/index come from the closure
    return orchestrator_node
```

```python
# states.py — Send payloads get their own schema instead of abusing AgentState
class WorkerState(TypedDict):
    step: PlanStep
    results: dict[int, str]
    current_datetime: str
```

- `api_server.py` accepts `"graph": "parallel"` in `RunRequest` and resolves via
  the registry; graphs are built once at startup (lifespan hook), with an
  injectable checkpointer.
- Old top-level modules (`pipeline_graph.py`, `paralel_pipeline_graph.py`)
  become thin import shims for one release, then get deleted.

**Acceptance criteria.** Adding a demo graph (e.g., orchestrate → execute →
writer-synthesis) touches exactly one new file plus one registry line; tests
build graphs with `GenericFakeChatModel` and no env vars; API can select the
graph per request.

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
Trivial after Phase 3's `nodes/assemble.py` split; possible before it as a flag
in the existing graphs.

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