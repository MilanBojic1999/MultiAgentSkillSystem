# Testing Guide — Multi-Agent Skills Pipeline

> **Status:** Plan / specification — no tests exist yet.
> **Audience:** Developers writing the test suite (item 2.3 of `IMPROVEMENT_PLAN.md`).
> **Created:** 2026-07-21
>
> This document specifies the test suite for this repository: layout, shared
> infrastructure, per-module test cases, and the CI workflow. It is written
> against the code as of the `cleaning-main` working tree; `file:line`
> references will drift as fixes land — trust the described behavior over the
> line number.

---

## Table of Contents

- [Ground rules](#ground-rules)
- [Known bugs the first tests must catch](#known-bugs-the-first-tests-must-catch)
- [The import-time side-effect problem](#the-import-time-side-effect-problem)
- [Test layout](#test-layout)
- [Shared infrastructure (`conftest.py`)](#shared-infrastructure-conftestpy)
- [Shared fixtures](#shared-fixtures)
- [Test module specifications](#test-module-specifications)
- [CI workflow](#ci-workflow)
- [Conventions](#conventions)
- [Definition of done](#definition-of-done)

---

## Ground rules

1. **No network, no `.env`, no live LLM.** Every test must pass on a fresh
   clone with `pip install -e ".[dev]"` and nothing else. If a test needs a
   model response, it uses a fake (see [Faking the LLM](#faking-the-llm)).
2. **Deterministic.** No sleeps, no timing assertions, no reliance on dict
   ordering beyond what Python guarantees.
3. **Test the contract, not the implementation.** The most bug-prone logic
   (routers, dependency scheduling, plan validation, skill parsing) is pure
   Python — test it as functions with plain-dict inputs and asserted outputs.
4. **Write the regression test before the fix.** For each known bug below,
   land a failing test first so the failure mode is pinned, then fix the code
   in the same PR.

---

## Known bugs the first tests must catch

Two bugs live in the current working tree. They are the strongest argument for
this suite: both would be caught by the first two tests below, and neither is
visible without either running a live pipeline or reading the code closely.

### Bug 1 — `paralel_pipeline_graph.py` does not import

`paralel_pipeline_graph.py:57-58` wires the scheduler with

```python
builder.add_conditional_edges("orchestrator", "scheduler")
builder.add_conditional_edges("parallel_sub_agent", "scheduler")
```

`add_conditional_edges` requires a **callable** path; passing the string
`"scheduler"` raises at import time in langgraph 1.2.4 (verified):

```
TypeError: Expected a Runnable, callable or dict. Instead got an unsupported type: <class 'str'>
```

These are plain edges and must be `builder.add_edge(...)` — which is also the
whole point of the 1.1 scheduler fix (plain edges dedupe). Additionally
`paralel_pipeline_graph.py:59` still uses the `Send` **class** as a path-map
key; per plan 1.1 the path map should be a plain list:
`["assemble", "parallel_sub_agent"]`.

**Caught by:** `test_dispatch_dedup.py::test_parallel_graph_module_imports`.

### Bug 2 — `validate_plan` called with wrong arity

`agents/orchestrator_node.py:85` calls

```python
plan = validate_plan(plan)
```

but the signature (`utils/plan_validator.py:25`) is
`validate_plan(plan, known_agents, known_skills)`. Every orchestrator run
raises `TypeError`, which the surrounding `except Exception` swallows and
re-raises as the misleading `"Failed to parse JSON response: ..."`, burning
the node's retry. The call must pass
`set(AGENT_ROSTER)` and `set(SKILL_INDEX)`.

**Caught by:** `test_orchestrator_node.py::test_valid_plan_is_validated_and_returned`.

Delete this section once both regression tests are green.

---

## The import-time side-effect problem

Most modules do real work at import, which is the main obstacle to testing.
Know these before writing any test:

| Module | Side effect at import | Consequence for tests |
|---|---|---|
| `llm_factory.py` | `load_dotenv()`; `create_llm()` raises `EnvironmentError` if `LLM_MODEL`/`LLM_URL` unset | conftest must set dummy env vars **before** any pipeline import |
| `agents/orchestrator_node.py`, `agents/sub_agents_nodes.py` | construct a `ChatOpenAI` client and call `load_skills()` at module level | importing anything from `agents/` needs env vars **and** a `skills/` dir in CWD |
| `config_loader.py` | reads `CONFIG_PATH` (or `agents/agent_config.json`) and validates it at import | error-path tests need a fresh import (`sys.modules.pop` + `importlib.import_module`) |
| `skill_loader.py` | none at import, but `root_dir = "skills"` is **CWD-relative** | tests must run with CWD = repo root, or monkeypatch `skill_loader.root_dir` |
| `pipeline_graph.py`, `paralel_pipeline_graph.py` | build **and compile** the graph with a `MemorySaver` at import | import the pure functions (`fan_out_router`, `should_continue`, `scheduler_node`); build test graphs from stubs rather than invoking the compiled module graph |
| `tools/agent_tools.py` | resolves the tool registry against `AGENT_CONFIG` at import | fine with the real config; don't try to fake it per-test |

Two mitigating facts:

- Constructing `ChatOpenAI` does **not** open a connection — dummy env values
  are enough to import everything; only `.invoke()` would hit the network.
- `load_dotenv()` never overrides variables that are already set, so values
  exported by `conftest.py` beat a developer's real `.env`.

`create_llm` is `lru_cache`d (`llm_factory.py:15`); tests that vary env vars
must call `create_llm.cache_clear()` (do it in a fixture).

> Long-term: this table is the test-facing symptom of the Phase 3 problem
> ("everything is wired at import time"). Don't fight it in the tests — pin
> behavior now, and Phase 3 removes the workarounds.

---

## Test layout

```
tests/
  conftest.py               # env defaults + CWD pinning + shared fixtures
  plans.py                  # canonical plan fixtures (linear, diamond, cyclic, ...)
  test_plan_validator.py    # plan 1.2 acceptance cases
  test_fan_out_router.py    # parallel-graph routing as a pure function
  test_should_continue.py   # sequential-graph routing
  test_dispatch_dedup.py    # plan 1.1 regression: exactly-once dispatch
  test_orchestrator_node.py # orchestrator with a fake LLM
  test_assemble_node.py     # output assembly formatting
  test_skill_loader.py      # frontmatter parsing against a tmp skills dir
  test_json_utils.py        # extract_json failure modes
  test_step_output_validator.py  # utils/validator.py
  test_sanitize.py          # utils/senitize.py injection patterns
  test_config_loader.py     # config errors need fresh-import machinery
  test_llm_factory.py       # env fallbacks, caching, missing-var errors
```

`tests/` is a plain directory (no `__init__.py`); pytest finds it from the
repo root. Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not integration'"
markers = ["integration: needs a live LLM endpoint; excluded by default"]
```

---

## Shared infrastructure (`conftest.py`)

The env vars must be set **before** the first pipeline import, and the CWD
must be the repo root **before** any module calls `load_skills()`. Both are
import-time concerns, so both happen at the top of `conftest.py`, not inside
fixtures:

```python
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Must precede every pipeline import. Port 9 (discard) makes any accidental
# live LLM call fail instantly instead of silently hitting a real server.
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_URL", "http://127.0.0.1:9/v1")
os.environ.setdefault("LLM_KEY", "test-key")
os.environ.setdefault("CONFIG_PATH", str(REPO_ROOT / "agents" / "agent_config.json"))

# skill_loader.root_dir is CWD-relative; agents/ modules call load_skills()
# at import, which can happen during collection — so pin CWD here.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))  # redundant after `pip install -e .`, harmless

import pytest  # noqa: E402


@pytest.fixture()
def fresh_llm_cache():
    from llm_factory import create_llm
    create_llm.cache_clear()
    yield
    create_llm.cache_clear()
```

---

## Shared fixtures

### Canonical plans (`tests/plans.py`)

Every plan-shaped test imports from one module so the shape is defined once:

```python
def step(n, agent="researcher", skills=(), deps=()):
    return {"step": n, "subtask": f"subtask {n}", "agent": agent,
            "skills_needed": list(skills), "depends_on": list(deps)}

LINEAR_PLAN  = [step(1), step(2, deps=[1]), step(3, deps=[2])]
DIAMOND_PLAN = [step(1), step(2), step(3, deps=[1, 2])]        # the 1.1 scenario
CYCLIC_PLAN  = [step(1, deps=[2]), step(2, deps=[1])]
KNOWN_AGENTS = {"researcher", "mathematician", "writer"}
KNOWN_SKILLS = {"yotta-researcher", "answer-writer", "roll-dice", "frontend-design"}
```

Use agent/skill names matching `agents/agent_config.json` and `skills/` so the
same fixtures also work in tests that touch the real config.

### Faking the LLM

Use `langchain_core.language_models.fake_chat_models.GenericFakeChatModel`
(available in the pinned langchain-core 1.4.0). It replays a scripted message
iterator through the full `BaseChatModel` interface, so it drops into any code
that expects a `ChatOpenAI`:

```python
import json
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

def fake_llm(payload: dict) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=json.dumps(payload))]))
```

Inject it with `monkeypatch.setattr("agents.orchestrator_node.llm", fake_llm({...}))` —
the orchestrator reads the module-global `llm`, so this swap is complete.

### Fake skills directory

`skill_loader` tests build their own tree under `tmp_path` and monkeypatch the
module global:

```python
@pytest.fixture()
def skills_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("skill_loader.root_dir", str(tmp_path))
    def make(name, frontmatter, body):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}")
    return make
```

---

## Test module specifications

### `test_plan_validator.py` — plan 1.2 acceptance criteria

Target: `utils/plan_validator.validate_plan`,
`utils/plan_validator.PlanValidationError`. Pure function, no mocking.

| Case | Input | Expected |
|---|---|---|
| valid linear plan | `LINEAR_PLAN` | returns list of plain dicts, sorted by `step` |
| normalization | valid plan given out of order, missing optional keys | `skills_needed`/`depends_on` default to `[]`; output sorted |
| not a list / empty list | `{}`, `[]` | `PlanValidationError` |
| schema violation | step missing `subtask`; `step: 0`; `subtask: ""` | `PlanValidationError` naming the index |
| duplicate step numbers | two steps with `step: 1` | `PlanValidationError` |
| unknown agent | `agent: "nonexistent"` | `PlanValidationError` naming the agent and listing known ones |
| unknown skill | one bogus name among real ones | **no raise**; bogus name dropped, real ones kept (soft failure) |
| self-dependency | `step 2, depends_on: [2]` | `PlanValidationError` |
| dangling dependency | `depends_on: [99]` | `PlanValidationError` naming `99` |
| cycle | `CYCLIC_PLAN` | `PlanValidationError` listing the cyclic steps |
| `PlanValidationError` is a `ValueError` | — | `issubclass` assert (the orchestrator `RetryPolicy` retries on `ValueError`) |

Use `pytest.raises(PlanValidationError, match=...)` so error messages stay
actionable — the message *is* part of the contract (it's what a developer sees
when the orchestrator re-plans).

### `test_fan_out_router.py` — routing as a pure function

Target: `paralel_pipeline_graph.fan_out_router`. Feed it plain state dicts;
assert on the returned `Send` list (check `s.node` and `s.arg["step"]["step"]`)
or the `"assemble"` string.

| Case | State | Expected |
|---|---|---|
| first layer | `DIAMOND_PLAN`, `results={}` | two `Send`s to `parallel_sub_agent`, steps 1 and 2 |
| second layer | `DIAMOND_PLAN`, `results={1:..., 2:...}` | one `Send`, step 3 |
| partial layer | `DIAMOND_PLAN`, `results={1:...}` | one `Send`, step 2 only (step 3 not ready) |
| all done | full results | `"assemble"` |
| `Send` payload | any ready step | payload carries `step`, `results`, `current_datetime` keys |
| blocked forever | plan with unsatisfiable dep, e.g. all remaining steps depend on a failed/absent step | **desired:** `RuntimeError` listing blocked steps (plan 1.2 guard); mark `xfail(strict=False)` until the guard lands — today it silently returns `"assemble"` |

### `test_should_continue.py` — sequential routing

Target: `pipeline_graph.should_continue` and
`agents.sub_agents_nodes.sub_agent_node`'s dead-state contract.

- results shorter than plan → `"sub_agent"`; equal → `"assemble"`.
- Blocked-forever guard: when no step is ready but results are incomplete,
  `sub_agent_node` currently returns `{}` — which makes `should_continue` loop
  until the recursion limit. Encode the desired `RuntimeError` guard as
  `xfail` like above. (Don't invoke the compiled sequential graph for this —
  asserting on the two functions is enough and avoids the recursion blow-up.)

### `test_dispatch_dedup.py` — the plan 1.1 regression test

The most important file in the suite. Two tests:

**1. `test_parallel_graph_module_imports`** — currently red (Bug 1):

```python
def test_parallel_graph_module_imports():
    import paralel_pipeline_graph  # noqa: F401
```

**2. `test_each_step_dispatched_exactly_once`** — builds the *real* topology
shape with stub nodes that count invocations. It must mirror the wiring in
`paralel_pipeline_graph.py` (plain edges into `scheduler`, single conditional
edge out of it); keep a comment in both files noting they must change together
— until Phase 3 extracts a `build(...)` factory, this duplication is the price
of testability.

```python
from collections import Counter
from langgraph.graph import StateGraph, END
from agent_states import AgentState
from assemble_node import assemble_node
from paralel_pipeline_graph import fan_out_router, scheduler_node
from tests.plans import DIAMOND_PLAN

def test_each_step_dispatched_exactly_once():
    calls = Counter()

    def stub_orchestrator(state):
        return {"plan": DIAMOND_PLAN, "results": {}, "current_step": 0}

    async def stub_worker(state):
        step_num = state["step"]["step"]
        calls[step_num] += 1
        return {"results": {step_num: f"out-{step_num}"}}

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", stub_orchestrator)
    builder.add_node("scheduler", scheduler_node)
    builder.add_node("parallel_sub_agent", stub_worker)
    builder.add_node("assemble", assemble_node)
    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")
    builder.add_conditional_edges("scheduler", fan_out_router,
                                  ["assemble", "parallel_sub_agent"])
    builder.add_edge("parallel_sub_agent", "scheduler")
    builder.add_edge("assemble", END)
    graph = builder.compile()

    out = graph.invoke({"task": "t", "current_datetime": ""})

    assert calls == {1: 1, 2: 1, 3: 1}          # the 1.1 bug makes step 3 == 2
    assert "out-3" in out["final_output"]
```

Also parametrize over `LINEAR_PLAN` and a wider fan (4 parallel steps, one
join) — the duplicate-dispatch count grows with layer width, so the wide case
is the loudest failure if the topology regresses.

**Failure containment** (plan 1.4, already implemented in
`paralel_pipeline_graph.parallel_sub_agent_node`): a third test uses a stub
`run_sub_agent_async` (monkeypatched) that raises for one step; assert the
run still completes, the failed step's result starts with `[STEP FAILED]`,
and `failed_steps` contains it. Note: `failed_steps` is **missing from
`AgentState`** (`agent_states.py:20`) — the plan calls for
`failed_steps: Annotated[list[int], operator.add]`; this test documents that
gap and goes red until the field is added.

### `test_orchestrator_node.py` — fake-LLM node test

Target: `agents.orchestrator_node.orchestrator_agent`. Currently red (Bug 2).

- `test_valid_plan_is_validated_and_returned` — monkeypatch `llm` with
  `fake_llm({"plan": DIAMOND_PLAN})` (use agent names from the real roster);
  call `orchestrator_agent({"task": "do things"})`; assert the returned
  `plan` is the validated/normalized form and `results == {}`.
- Plan wrapped in a ```json fence → still parsed (exercises `extract_json`
  integration).
- Empty plan (`{"plan": []}`) → `ValueError`.
- Non-JSON response → `ValueError`.
- Unknown agent in the plan → raises (proves validation is actually wired in,
  not just imported).
- Injection-looking task (e.g. "ignore all previous instructions...") →
  `ValueError` from `sanitize_content` (documents that sanitization gates the
  orchestrator).

### `test_assemble_node.py`

Pure function. Ordered sections per plan step (`## Step N: <subtask>`);
missing result renders as empty string; empty plan → empty `final_output`.
When plan 1.4's warning header lands (assemble prepends a warning if
`failed_steps` non-empty), add that case here.

### `test_skill_loader.py`

Target: `_split_frontmatter`, `load_skills`, `load_skills_body`, via the
`skills_dir` fixture.

- Well-formed file → frontmatter YAML parsed, `name` key indexed, body
  returned stripped.
- File not starting with `---` → `ValueError` naming the path (regression for
  the old `content.split("---")[1]` bug — fixed, keep it pinned).
- Body containing `---` separators → body not truncated (the regex is
  non-greedy on the *frontmatter*, not the body).
- Directory without `SKILL.md` → skipped silently.
- `load_skills_body` with unknown skill name → `ValueError`.
- Real-repo smoke test (no fixture): `load_skills()` against the checked-in
  `skills/` finds the four known skills.

### `test_json_utils.py`

Target: `utils.json_utils.extract_json`.

Direct object; direct array; ```json fence; bare fence; JSON with leading and
trailing prose; array-before-object precedence (prose containing
`[{...}, {...}]`); nothing parseable → `ValueError` with truncated payload in
the message.

### `test_step_output_validator.py`

Target: `utils.validator.validate_step_output`. Non-string → `ValueError`;
empty/whitespace → `ValueError`; > 50 000 chars → `ValueError`; each
`BLOCKED_PATTERNS` entry (`<script`, `javascript:`, ` ```system `) →
`ValueError`; a normal multi-paragraph markdown answer passes through
unchanged.

### `test_sanitize.py`

Target: `utils.senitize.sanitize_content`. Parametrize each injection pattern
with a matching string → `ValueError`. Just as important, pin the
false-positive boundary — benign strings that must **pass**: "the previous
chapter covers these instructions", "you are now ready to begin", a local
image `![alt](./plot.png)` (only `http(s)` URLs are blocked).

### `test_config_loader.py`

`config_loader` runs at import, so error paths need a fresh import:

```python
import importlib, sys

def fresh_config_loader(monkeypatch, path):
    monkeypatch.setenv("CONFIG_PATH", str(path))
    sys.modules.pop("config_loader", None)
    return importlib.import_module("config_loader")
```

Note: reloading `config_loader` orphans the already-imported `agents` package
(it keeps its old reference) — keep these tests self-contained and don't mix
them with graph tests in the same module.

- Missing file → `FileNotFoundError` whose message names the tried path and
  mentions `.env.example` (this message is 2.1's acceptance criterion).
- Duplicate MCP-server owner across two agents → `ValueError` naming both.
- Valid minimal config → `AGENT_CONFIG` keyed by agent name.
- Real-repo smoke test: default import exposes a non-empty `AGENT_CONFIG`.

### `test_llm_factory.py`

Target: `llm_factory.create_llm` (use `fresh_llm_cache`; constructing
`ChatOpenAI` is offline-safe).

- Env fallback: with conftest dummies, `create_llm()` returns a client with
  `model_name == "test-model"`.
- Missing `LLM_MODEL`/`LLM_URL` (`monkeypatch.delenv` + explicit
  `model=None`) → `EnvironmentError` naming exactly the missing variables.
  Caveat: `load_dotenv` already ran, so also monkeypatch-delete anything a
  real `.env` may have injected.
- Caching: same args twice → identical object (`is`); different temperature →
  different object.
- Defaults: temperature 0.9 / max_tokens 4096 when unspecified; explicit
  `temperature=0.0` is respected (guard against `or`-style falsy bugs).

---

## CI workflow

`.github/workflows/ci.yml` — lint + test on every push and PR. No secrets,
no services:

```yaml
name: CI
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"   # match the dev venv
          cache: pip
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest -q
```

Keep it to one job until it's slow. When ruff is first enabled expect a batch
of findings across the repo; fix or `# noqa` deliberately in a dedicated
commit, not mixed into a feature PR.

---

## Conventions

- One test module per production module; name `test_<module>.py`.
- `pytest.mark.parametrize` for input tables; one behavior per test function;
  test names state the expected behavior
  (`test_cycle_raises_plan_validation_error`).
- Always `pytest.raises(..., match=...)` — error *messages* are part of this
  codebase's contract (they drive re-planning and developer debugging).
- Desired-but-not-yet-implemented behavior (the two deadlock guards, the
  `failed_steps` state field) goes in as `@pytest.mark.xfail(strict=False,
  reason="plan 1.2/1.4 guard not yet implemented")` — the suite documents the
  target contract and flips to green as fixes land.
- Anything needing a live endpoint gets `@pytest.mark.integration` and is
  excluded by default via `addopts`. Nothing in this document needs it.
- No test writes outside `tmp_path`. No test depends on another test's state.

---

## Definition of done

Matches `IMPROVEMENT_PLAN.md` 2.3 acceptance criteria, expanded:

1. `pytest -q` passes from a fresh clone with no network, no `.env`, and no
   live services — including on a machine where port 9 is closed (it should
   be everywhere; that's the point).
2. Bug 1 and Bug 2 above each have a regression test that failed before the
   fix and passes after; the fixes land in the same PRs as their tests.
3. The dedup test (`test_each_step_dispatched_exactly_once`) is green on the
   corrected scheduler topology, including the wide-fan parametrization.
4. All `xfail` markers reference the plan item that will flip them.
5. CI is green on the PR that introduces the suite; `ruff check .` passes.
6. `README.md` gains a "Running the tests" section
   (`pip install -e ".[dev]" && pytest`) in the same PR — per the plan's rule
   that docs land with the feature.
