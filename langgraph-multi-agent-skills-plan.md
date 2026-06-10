# Implementation Plan: Multi-Agent Pipeline with Agent Skills on LangGraph/LangChain

> Adaptation of *Building a Multi-Agent Pipeline with Agent Skills & a Tool Broker*
> to LangGraph 1.x / LangChain 1.x as the base runtime.
> June 2026.

---

## 0. The Big Picture: What Changes, What Stays, What Gets Deleted

Your original plan is architecturally sound — it is essentially a **plan-and-execute
graph with a policy-enforcing tool layer**. LangGraph is purpose-built for exactly
that shape. The migration is therefore mostly a *re-hosting*, with three categories
of change:

| Your component | Fate in LangGraph | Why |
|---|---|---|
| Skill library + `SKILL.md` files | **KEEP unchanged** | LangChain has no native skill concept. Your loader, frontmatter format, and progressive disclosure survive verbatim. This is your differentiator — don't touch it. |
| `skill_loader.py` | **KEEP unchanged** | Pure Python, framework-agnostic. |
| Orchestrator (JSON plan via raw SDK call) | **REPLACE** with a planner **node** using `.with_structured_output(Plan)` | Pydantic-validated plans; no fragile `json.loads` on raw text. |
| Sub-agent runner (hand-rolled tool-use loop) | **DELETE** — replaced by `create_agent` (LangChain 1.x) compiled subgraphs | ~100 lines of `while stop_reason == "tool_use"` loop disappear. |
| `CheckpointStore` (custom JSON files) | **DELETE** — replaced by LangGraph checkpointers (`SqliteSaver` / `PostgresSaver`) | Built-in, durable, supports resume-from-failure, time travel, and human-in-the-loop interrupts for free. |
| Tool Broker: registry + allowlists | **KEEP, slimmed** — becomes a policy layer that *resolves* per-agent tool lists | LangGraph binds tools per agent structurally; your broker decides *which* tools each agent gets. |
| Tool Broker: rate limits / cost budgets | **PORT** into LangChain middleware (`wrap_tool_call`) or a tool wrapper | One uniform enforcement point for native AND MCP tools (see §4.4 — this is a major simplification). |
| MCP integration via Anthropic SDK `mcp_servers=` | **REPLACE** with `langchain-mcp-adapters` `MultiServerMCPClient` | MCP tools become first-class `BaseTool` objects, executed client-side. Your "the SDK already executed it" special case in the tool loop disappears entirely. |
| Dependency-aware sequential loop in `pipeline.py` | **REPLACE** with graph edges + `Send` API for parallel fan-out | Your `depends_on` DAG becomes literal graph topology. Fan-out (your Advanced Topic A) is native. |
| Retry-with-backoff helper | **DELETE** — `RetryPolicy` on nodes | Built-in. |
| Structured logging | **AUGMENT** with LangSmith tracing | Keep your `log_event`; LangSmith gives full trace trees, token/cost accounting per run for free. |
| `MCPServerConfig` + access matrix | **KEEP nearly unchanged** | It becomes the input to per-agent `MultiServerMCPClient` construction. |
| Activation tests, single-responsibility rule, description discipline | **KEEP unchanged** | Framework-independent skill hygiene. |

**Net effect:** you delete roughly 40% of the custom infrastructure code (tool loop,
checkpoint store, retry, sequential scheduler) and keep 100% of the conceptual
architecture (Skills, Broker-as-policy, dedicated/shared MCP, orchestrator/specialist
split).

---

## 1. Key Architectural Decision: Plan-and-Execute vs. Supervisor

LangGraph supports two canonical multi-agent topologies. You must pick one as the
spine (you can hybridize later):

**A — Plan-and-Execute (recommended for you).** A planner node produces the full
dependency-aware plan up front (exactly your current orchestrator), a scheduler
node dispatches ready steps (possibly in parallel via `Send`), and a joiner/replanner
node assembles results. This is a direct 1:1 mapping of your existing design,
preserves your `depends_on` DAG, gives deterministic, auditable, checkpointable
plans, and is cheaper (one planning call instead of a routing call per step).

**B — Supervisor (dynamic routing).** The orchestrator is itself an agent whose
"tools" are handoffs to sub-agents; it decides the next step after each result.
More adaptive for open-ended tasks, but less deterministic and more expensive.
Note: the `langgraph-supervisor` library now exists mainly for legacy compatibility —
the LangChain team recommends implementing the supervisor pattern directly via
tool-calling handoffs if you go this route.

**Recommendation:** build **A** as the spine, and add a *replanner* edge (joiner →
planner) so the orchestrator can revise the plan when a step fails or returns
surprising results. That gives you 90% of supervisor adaptivity inside the
plan-and-execute skeleton.

---

## 2. Target Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH StateGraph (compiled)                     │
│                                                                          │
│   START ──► planner ──► scheduler ──► [Send: step_executor × N] ──┐     │
│                ▲             ▲                                      │     │
│                │             └──────────── joiner ◄─────────────────┘     │
│                │                             │                            │
│                └───── replan? ◄──────────────┤                            │
│                                              ▼                            │
│                                          synthesizer ──► END             │
│                                                                          │
│   step_executor = generic node that:                                     │
│     1. looks up agent config from roster                                 │
│     2. activates requested SKILL.md bodies → system prompt               │
│     3. asks ToolBroker for this agent's tools (native + MCP, filtered)   │
│     4. invokes a create_agent subgraph                                   │
│     5. writes result into shared state                                   │
│                                                                          │
│   Checkpointer: SqliteSaver / PostgresSaver (thread_id = stable run id)  │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   TOOL BROKER (policy layer) │
                    │  • per-agent tool resolution │
                    │  • MCP allowlist filtering   │
                    │  • rate limit / cost wrapper │
                    └──────┬───────────────┬───────┘
                           │               │
                  Native @tool fns   MultiServerMCPClient
                                     (per-agent instances,
                                      dedicated + shared MCPs)
```

The crucial difference from your original diagram: **MCP tools and native tools
flow through one uniform path.** With `langchain-mcp-adapters`, MCP tools are
converted into ordinary LangChain `BaseTool` objects executed by the agent's tool
node — so the Broker can wrap, rate-limit, log, and allowlist them *identically*
to native tools. Your original design had to special-case MCP because the Anthropic
SDK executed those calls server-side. That asymmetry is gone.

---

## 3. Phase Plan

### Phase 0 — Environment & Dependencies (½ day)

```
# requirements.txt
langchain>=1.0                    # create_agent, middleware
langgraph>=1.0                    # StateGraph, Send, checkpointers
langchain-anthropic>=0.3          # ChatAnthropic
langchain-mcp-adapters>=0.1       # MultiServerMCPClient
langgraph-checkpoint-sqlite>=2.0  # dev checkpointer
langgraph-checkpoint-postgres>=2.0  # prod checkpointer
pyyaml>=6.0                       # SKILL.md frontmatter (unchanged)
python-dotenv>=1.0
pytest>=8.0
pytest-asyncio>=0.23
langsmith>=0.2                    # tracing (optional but recommended)
```

Pin versions; the LangChain 1.x API surface (`langchain.agents.create_agent`,
middleware) is the stable target — avoid pre-1.0 tutorials that use
`langgraph.prebuilt.create_react_agent` (it still works but `create_agent` is the
current canonical entry point).

Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` from day one; retrofitting
observability is always worse.

Decision to make now: **everything async.** `MultiServerMCPClient` is async-first;
build the graph with `async def` nodes and `graph.ainvoke()` throughout.

### Phase 1 — Skill Library (no changes, ~0 days)

Carry over verbatim from your guide: Steps 1–2 (domain mapping, single-responsibility
rule, frontmatter spec, description-as-activation-signal, scripts/references folders),
the multi-source loader from Appendix F, and the activation test suites from Step 9.1.

One addition for LangGraph: store the *skill index* (names + descriptions only) in
graph state at planner time so the plan itself records which index version it was
made against — useful when skills change mid-flight between checkpoint resume.

### Phase 2 — Graph State Schema (½ day)

This replaces the ad-hoc `results: dict[int, str]` in `pipeline.py`. State is the
contract between every node:

```python
# graph/state.py
import operator
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    step: int
    subtask: str
    agent: str
    skills_needed: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)

class Plan(BaseModel):
    plan: list[PlanStep]

class StepResult(TypedDict):
    step: int
    agent: str
    output: str
    ok: bool
    error: str | None

class PipelineState(TypedDict):
    task: str
    skill_index: list[dict]                                  # discovery-phase index
    plan: list[PlanStep]
    # Annotated reducer: parallel step_executor branches APPEND results
    # concurrently without clobbering each other — this is what makes
    # Send-based fan-out safe.
    results: Annotated[list[StepResult], operator.add]
    completed_steps: Annotated[set[int], operator.or_]
    replan_count: int
    final_output: str
```

The `Annotated[..., operator.add]` reducer is the LangGraph idiom that makes
parallel fan-out write-safe — it replaces the manual results-dict bookkeeping and
the write-lock concerns from your failure-modes table.

### Phase 3 — Orchestrator as Planner Node (1 day)

Your `run_orchestrator` becomes a node with structured output. No JSON parsing,
no prompt-format pleading:

```python
# graph/planner.py
from langchain_anthropic import ChatAnthropic
from .state import PipelineState, Plan

planner_llm = ChatAnthropic(model="claude-sonnet-4-5").with_structured_output(Plan)

PLANNER_SYSTEM = """You are the Orchestrator in a multi-agent pipeline.
Decompose the task into ordered subtasks. For each subtask select the best
specialist from the roster and list the skills it needs. Do not execute anything.

## Available sub-agents
{agent_roster}

## Available skills (name → description)
{skill_index}
"""

async def planner_node(state: PipelineState) -> dict:
    skill_summary = "\n".join(
        f"- {s['name']}: {s['description']}" for s in state["skill_index"]
    )
    roster_summary = "\n".join(
        f"- {a['name']}: {a['description']}" for a in AGENT_ROSTER
    )
    plan: Plan = await planner_llm.ainvoke([
        ("system", PLANNER_SYSTEM.format(
            agent_roster=roster_summary, skill_index=skill_summary)),
        ("user", state["task"]),
    ])
    _assert_acyclic(plan.plan)          # keep your circular-dep check
    return {"plan": plan.plan}
```

Validation that previously lived in your checklist ("dependency graph is correct —
no circular deps", "agent name exists in roster") now lives in Pydantic validators
on `PlanStep`/`Plan` — the plan is rejected *before* execution starts.

### Phase 4 — Tool Broker as Policy Layer + MCP via Adapters (2–3 days)

This is the heart of the migration. The Broker stops *executing* tools and starts
*resolving and wrapping* them.

**4.1 — `MCPServerConfig` survives nearly unchanged:**

```python
# tool_broker/mcp_config.py  (same dataclass, plus transport)
@dataclass
class MCPServerConfig:
    name: str
    url: str
    transport: str = "streamable_http"   # or "stdio" for local servers
    allowed_agents: list[str] = field(default_factory=list)  # [] = all
    description: str = ""
    env_auth_key: str | None = None

MCP_SERVERS = [ ... ]   # identical declarations + access-matrix comment
```

**4.2 — Per-agent MCP clients enforce dedicated vs. shared structurally.**
Instead of one global client, build one `MultiServerMCPClient` per agent containing
*only* the servers that agent is allowed. A `researcher` cannot call `write-db-mcp`
because its client literally never learns that server's URL — same guarantee your
original design achieved via `mcp_servers_for_agent`, now expressed in adapter
config:

```python
# tool_broker/mcp_tools.py
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from .mcp_config import MCP_SERVERS

def _servers_for_agent(agent_name: str) -> dict:
    cfgs = {}
    for cfg in MCP_SERVERS:
        if cfg.allowed_agents and agent_name not in cfg.allowed_agents:
            continue
        entry = {"url": cfg.url, "transport": cfg.transport}
        if cfg.env_auth_key:
            token = os.environ.get(cfg.env_auth_key)
            if token is None:
                raise RuntimeError(f"Missing env var {cfg.env_auth_key}")  # fail fast
            entry["headers"] = {"Authorization": f"Bearer {token}"}
        cfgs[cfg.name] = entry
    return cfgs

async def mcp_tools_for_agent(agent_name: str) -> list:
    client = MultiServerMCPClient(_servers_for_agent(agent_name))
    return await client.get_tools()      # ordinary BaseTool objects
```

Cache the tool lists per agent at startup (they rarely change), but rebuild on
auth failure.

**4.3 — Native tools become `@tool` functions:**

```python
# tool_broker/tools/web_search.py
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use 3-6 word queries."""
    ...
```

Registration metadata (`tags`, `rate_limit`, `allowed_agents`) moves into a thin
registry dict keyed by tool name — your `ToolDefinition` minus the handler plumbing.

**4.4 — One uniform policy wrapper for native AND MCP tools.**
Because MCP tools now execute client-side as `BaseTool`s, rate limiting, cost
budgets, logging, and prompt-injection sanitisation apply at a single choke point.
Two equivalent implementations; pick one:

*(a) LangChain 1.x middleware* — attach to `create_agent`:

```python
# tool_broker/policy.py
from langchain.agents.middleware import AgentMiddleware

class BrokerPolicy(AgentMiddleware):
    def __init__(self, agent_name: str, broker):
        self.agent_name, self.broker = agent_name, broker

    async def wrap_tool_call(self, request, handler):
        self.broker.check_access(request.tool_name, self.agent_name)   # allowlist
        self.broker.enforce_rate_limit(request.tool_name)              # sliding window
        self.broker.charge_budget(self.agent_name, request.tool_name)  # cost ceiling
        log_event("tool_call", agent=self.agent_name, tool=request.tool_name)
        result = await handler(request)
        return self.broker.sanitize(result)     # prompt-injection scrub on tool output
    ```

*(b) Tool wrapping* — a `guarded(tool, agent_name, broker)` function that returns a
new `BaseTool` whose `_arun` performs the same checks before delegating. Use this
if you want zero coupling to the middleware API.

**4.5 — The Broker facade shrinks to its policy core:**

```python
# tool_broker/broker.py
class ToolBroker:
    """Resolves and polices tools. Never executes them — agents do."""

    async def tools_for_agent(self, agent_name: str) -> list:
        native = [t for t in NATIVE_TOOLS
                  if self._allowed(t.name, agent_name)]
        mcp = await mcp_tools_for_agent(agent_name)
        return native + mcp

    # check_access / enforce_rate_limit / charge_budget / sanitize
    # — port directly from your existing broker.py
```

Your capability resolver (intent → tool) becomes unnecessary in the common path:
the agent's LLM selects tools from its bound schema list, which is already
agent-scoped. Keep the resolver only if you want deterministic non-LLM routing for
specific intents.

### Phase 5 — Sub-Agents as `create_agent` Subgraphs with Skill Injection (2 days)

Your entire hand-rolled tool-use loop is replaced by one factory. Skills inject
into the **system prompt** exactly as before (this preserves your hard-won rule:
skill bodies in system prompt, never in the user turn):

```python
# agents/specialist.py
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from agents.skill_loader import activate_skill

SPECIALIST_TEMPLATE = """You are the {name} specialist agent.
Role: {description}

## Active skills
{skill_instructions}
{context_block}

Use your tools when needed. Return your final answer as plain text."""

async def build_specialist(agent_cfg: dict, skill_paths: list[str],
                           context: str, broker) -> object:
    skill_bodies = [activate_skill(p) for p in skill_paths]   # activation phase
    system = SPECIALIST_TEMPLATE.format(
        name=agent_cfg["name"],
        description=agent_cfg["description"],
        skill_instructions="\n\n---\n\n".join(skill_bodies),
        context_block=f"\n\n## Upstream context\n{context}" if context else "",
    )
    tools = await broker.tools_for_agent(agent_cfg["name"])
    return create_agent(
        model=ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4000),
        tools=tools,
        system_prompt=system,
        middleware=[BrokerPolicy(agent_cfg["name"], broker)],
    )
```

Progressive disclosure is preserved end-to-end: the planner saw only the discovery
index (~30–50 tokens/skill); the specialist receives only the activated bodies for
its step; reference files load on demand if you also give specialists a
`read_skill_reference(path)` tool restricted to the skills directory.

### Phase 6 — Graph Wiring: Scheduler, Fan-Out, Joiner (2 days)

Your sequential `for step in steps` loop, the `depends_on` context-gathering, and
Advanced Topic A (fan-out) all collapse into the graph topology:

```python
# graph/build.py
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

def schedule(state: PipelineState):
    """Dispatch every step whose dependencies are satisfied — in parallel."""
    done = state["completed_steps"]
    ready = [s for s in state["plan"]
             if s.step not in done and set(s.depends_on) <= done]
    if not ready:
        return "joiner"
    return [Send("step_executor", {"step": s, **_shared(state)}) for s in ready]

async def step_executor(payload: dict) -> dict:
    step = payload["step"]
    context = _summarised_upstream(payload, step.depends_on)  # keep <4k tokens
    agent_cfg = ROSTER_BY_NAME[step.agent]
    specialist = await build_specialist(agent_cfg, _skill_paths(step), context, broker)
    try:
        out = await specialist.ainvoke({"messages": [("user", step.subtask)]})
        result = StepResult(step=step.step, agent=step.agent,
                            output=out["messages"][-1].content, ok=True, error=None)
    except Exception as e:                       # graceful degradation, as before
        result = StepResult(step=step.step, agent=step.agent,
                            output="", ok=False, error=str(e))
    return {"results": [result], "completed_steps": {step.step}}

def after_join(state: PipelineState):
    if any(not r["ok"] for r in state["results"]) and state["replan_count"] < 2:
        return "planner"                          # adaptive replan on failure
    if len(state["completed_steps"]) < len(state["plan"]):
        return "scheduler"                        # more waves of ready steps
    return "synthesizer"

g = StateGraph(PipelineState)
g.add_node("planner", planner_node)
g.add_node("scheduler", lambda s: s)              # pass-through anchor
g.add_node("step_executor", step_executor)
g.add_node("joiner", lambda s: s)
g.add_node("synthesizer", synthesizer_node)       # merges results → final_output
g.add_edge(START, "planner")
g.add_edge("planner", "scheduler")
g.add_conditional_edges("scheduler", schedule, ["step_executor", "joiner"])
g.add_edge("step_executor", "joiner")
g.add_conditional_edges("joiner", after_join,
                        ["planner", "scheduler", "synthesizer"])
g.add_edge("synthesizer", END)
```

Notes on this design: `Send` gives you wave-based parallelism — every step whose
deps are met runs concurrently, then the joiner schedules the next wave. The
`operator.add` reducer on `results` makes concurrent writes safe. The replan edge
implements §1's hybrid: failures route back through the planner with prior results
in state, bounded by `replan_count`.

### Phase 7 — Checkpointing & Resume (½ day — mostly deletion)

Delete `state/checkpoint_store.py` entirely. Compile with a checkpointer and use a
deterministic thread id, which reproduces your skip-completed-steps semantics
exactly — LangGraph persists state after every super-step, so a crashed run resumed
with the same `thread_id` continues from the last completed wave:

```python
import hashlib
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# prod: from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async def run_pipeline(task: str) -> str:
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        app = g.compile(checkpointer=saver)
        thread_id = hashlib.sha256(task.encode()).hexdigest()[:16]  # stable run_id
        final = await app.ainvoke(
            {"task": task, "skill_index": load_skill_index("./skills"),
             "results": [], "completed_steps": set(), "replan_count": 0},
            config={"configurable": {"thread_id": thread_id}},
        )
        return final["final_output"]
```

Bonus capabilities you get free that your custom store didn't have: time-travel
(inspect/fork any historical state), and `interrupt()` for human approval gates —
worth inserting before any `db-writer` step that mutates production data.

### Phase 8 — Observability, Retries, Error Handling (1 day)

Replace `with_retry` with node-level policies:

```python
from langgraph.types import RetryPolicy
g.add_node("step_executor", step_executor,
           retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0,
                                    backoff_factor=2.0))
```

Keep your structured `log_event` for business events (`step_start`, `step_done`,
`run_complete`) and let LangSmith handle trace trees, latency, and token/cost
accounting per run — this replaces your manual §10.4 cost arithmetic with exact
per-model accounting. Tag every trace with `thread_id` and `agent` metadata so the
MCP access audit ("which agent called which MCP tool when") is a saved LangSmith
filter rather than log archaeology.

### Phase 9 — Testing (1–2 days)

Port your three test layers; they translate cleanly:

1. **Skill activation tests** (unchanged): run each positive/negative phrase
   through the planner node with the skill index loaded; assert `skills_needed`.
   Structured output makes assertions trivial — no text parsing.
2. **Broker policy tests**: assert `PermissionError` when a fake `researcher`
   invokes a `write-db-mcp` tool through `BrokerPolicy`; assert rate-limit raises
   on the N+1th call. Run MCP servers as local `FastMCP` fixtures over stdio.
3. **Graph integration tests**: invoke the compiled graph with `MemorySaver` and a
   fake model (LangChain's `GenericFakeChatModel`) scripted to emit a known plan;
   assert wave ordering, parallel dispatch of independent steps, checkpoint resume
   (kill after wave 1, re-invoke same `thread_id`, assert wave 1 not re-executed),
   and the replan edge on injected step failure.

### Phase 10 — Deployment (1 day)

Two options, both compatible with the same compiled graph:

- **Self-hosted Docker** (your current plan): same Dockerfile, plus `COPY skills/
  /app/skills/`, Postgres checkpointer DSN via env, and all `env_auth_key` secrets.
- **LangGraph Platform / Studio**: expose the graph via `langgraph.json`; you get a
  managed API, the visual graph debugger in Studio (invaluable for inspecting
  scheduler waves), and built-in persistence. Studio works locally via
  `langgraph dev` even if you never deploy to the platform — use it during
  development regardless.

---

## 4. Updated File & Folder Layout

```
my-pipeline/
├── main.py                         # run_pipeline entrypoint
├── langgraph.json                  # optional: Studio/Platform manifest
│
├── graph/
│   ├── state.py                    # PipelineState, Plan, PlanStep, reducers
│   ├── planner.py                  # planner node (structured output)
│   ├── build.py                    # StateGraph wiring, schedule(), joiner
│   └── synthesizer.py              # final merge node
│
├── agents/
│   ├── specialist.py               # build_specialist (create_agent factory)
│   ├── skill_loader.py             # UNCHANGED from original plan
│   └── roster.py                   # UNCHANGED agent definitions
│
├── tool_broker/
│   ├── broker.py                   # slim policy facade (resolve, not execute)
│   ├── policy.py                   # BrokerPolicy middleware (rate/cost/allowlist)
│   ├── mcp_config.py               # UNCHANGED MCPServerConfig + access matrix
│   ├── mcp_tools.py                # per-agent MultiServerMCPClient construction
│   └── tools/                      # native @tool functions
│
├── skills/                         # UNCHANGED skill library
├── tests/
│   ├── test_activation.py
│   ├── test_broker_policy.py
│   └── test_graph.py
└── requirements.txt
```

---

## 5. Failure-Mode Table Deltas

Most of your Appendix G survives. Rows that change:

| Original symptom | LangGraph-era resolution |
|---|---|
| Pipeline re-runs completed steps | Gone by construction — checkpointer + stable `thread_id`. |
| Context window overflow at step 8 | Still your responsibility: summarise upstream results in `_summarised_upstream` before injection. LangGraph 1.x also offers summarization middleware on agents for long tool loops. |
| MCP call succeeds but result is ignored | Gone — adapters return tool results as ordinary `ToolMessage`s in the agent loop. |
| Agent calls an MCP it shouldn't | Now *two* layers: structural (server absent from that agent's client config) and policy (`BrokerPolicy` check). Test both. |
| Rate limit hit on parallel fan-out | Real and sharper with `Send` parallelism — enforce the sliding window in `BrokerPolicy` with an `asyncio.Lock`-guarded counter shared across branches. |
| New: concurrent state clobbering in fan-out | Prevented by `Annotated[..., operator.add]` reducers — never write bare keys from parallel branches. |
| New: stdio MCP in server context | Prefer `streamable_http` transports for anything deployed; stdio only for local dev fixtures. |

---

## 6. Build Order & Effort Estimate

| Phase | Deliverable | Est. effort |
|---|---|---|
| 0 | Env, deps, LangSmith wired | 0.5 d |
| 2 | State schema + Plan models | 0.5 d |
| 3 | Planner node + plan validation | 1 d |
| 4 | Broker policy layer + MCP adapters + native tools | 2–3 d |
| 5 | Specialist factory with skill injection | 2 d |
| 6 | Graph wiring + Send fan-out + replan edge | 2 d |
| 7 | Checkpointing (deletion + thread ids) | 0.5 d |
| 8 | RetryPolicy + LangSmith tagging | 1 d |
| 9 | Three-layer test suite | 1–2 d |
| 10 | Docker / Studio deployment | 1 d |
| | **Total** | **~12–14 working days** |

Suggested milestone cuts: **M1** (Phases 0,2,3,5 with one hardcoded specialist, no
MCP) — a working planner→specialist→synthesizer graph in ~3 days; **M2** adds the
Broker + MCP allowlists; **M3** adds fan-out, replan, checkpoint resume; **M4**
hardening + deploy.

---

## 7. Migration Checklist (delta to your original checklist)

- [ ] All skills + loader copied over unchanged; activation tests green against the planner node
- [ ] `Plan`/`PlanStep` Pydantic models reject cyclic `depends_on` and unknown agents at parse time
- [ ] One `MultiServerMCPClient` config per agent; dedicated MCP appears in exactly one agent's config
- [ ] `BrokerPolicy` middleware attached to every `create_agent` instance — no specialist constructed without it
- [ ] Rate-limit counters are shared (and lock-guarded) across parallel `Send` branches
- [ ] All concurrent-write state keys use reducers (`operator.add` / `operator.or_`)
- [ ] Custom `CheckpointStore`, retry helper, and tool-use loop deleted (not merely unused)
- [ ] `thread_id` is a deterministic hash of the task; resume test passes
- [ ] `interrupt()` gate inserted before production-write steps (`db-writer`)
- [ ] Upstream context summarised to <4k tokens before specialist injection
- [ ] LangSmith traces tagged with `thread_id`, `agent`, `mcp_server`; MCP access audit saved as a filter
- [ ] Graph renders correctly in LangGraph Studio (`langgraph dev`) before first deploy
