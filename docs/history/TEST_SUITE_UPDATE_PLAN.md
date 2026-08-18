# Test Suite Update Plan

> **Status:** Plan — no changes made yet.
> **Created:** 2026-07-23
> **Origin:** Review of the item-2.3 test suite (73 passed / 15 xfailed on
> `cleaning-main`). The suite's philosophy is sound — hermetic, red-first,
> contract-focused — this plan closes the holes the review found.
> **Relation to `docs/history/IMPROVEMENT_PLAN.md`:** Phases A and B are
> test-only. Phase C touches production code and is coordinated with roadmap
> items 1.1, 1.2 and Phase 3 — those fixes stay in their own PRs; this plan
> only says which tests must land with them.

---

## Phase A — Suite hygiene (test-only, do first)

These fix tests that lie or leak. No production code changes. Land as one PR
**before** the Bug 1 / Bug 2 fixes, so the bug-fix PRs flip honest markers.

### A.1 `test_unknown_agent_in_plan_raises` is green for the wrong reason

**Problem:** Bug 2 (`validate_plan` called with wrong arity,
`agents/orchestrator_node.py:85`) makes *every* call through `validate_plan`
raise `TypeError`, which the blanket `except` wraps into `ValueError`. The
bare `pytest.raises(ValueError)` in `tests/test_orchestrator_node.py:52`
catches that wrapper — validation never runs, yet the test passes while its
docstring claims it "proves validation is actually wired in".

**Change:** assert on the validation message —
`pytest.raises(ValueError, match="unknown agent 'nobody'")` — and mark the
test `xfail` with the same Bug 2 reason as its two siblings. It goes red-first
like the rest and flips when Bug 2 is fixed.

**Acceptance:** with Bug 2 still present the test reports XFAIL, not PASS;
after the Bug 2 fix it passes only if `validate_plan` actually rejected the
agent.

### A.2 Decide the xfail-strictness policy before the fixes land

**Problem:** all 15 markers are `strict=False`, so when Bugs 1/2 and the plan
1.2/1.4 guards land, the tests silently become XPASS and the stale markers
linger forever. Only `test_parallel_graph_module_imports` even has a comment
about removing its marker.

**Change:** deterministic failures get `strict=True` so a fix *breaks* the
suite until the marker is removed (that's the forcing function):

| Test(s) | Marker becomes |
|---|---|
| `test_parallel_graph_module_imports` (Bug 1) | `strict=True` |
| `test_orchestrator_node` Bug 2 trio (incl. A.1) | `strict=True` |
| Every `import_parallel_or_xfail` imperative xfail | unchanged (imperative xfail is inherently non-strict; these disappear naturally when Bug 1 is fixed) |
| Plan 1.2 / 1.4 guard tests (blocked-forever, `failed_steps`, assemble warning header) | `strict=True` — the guards are deterministic too |

Do **not** set a global `xfail_strict = true` in `pyproject.toml` — the
imperative helper xfails would fight it. Per-marker is explicit and greppable.

**Acceptance:** `grep -rn "xfail" tests/` shows every declarative marker with
an explicit `strict=` and a reason naming its plan item; fixing Bug 1 or Bug 2
without touching the tests makes `pytest` fail with XPASS errors.

### A.3 Conftest hardening

Two leaks:

1. **Real shell env beats the canary.** `conftest.py` uses
   `os.environ.setdefault`, so a developer who exports a real `LLM_URL` runs
   the whole suite against it, silently losing the port-9 protection. Change
   the four `setdefault` calls to hard assignment (`os.environ[...] = ...`).
   Tests are hermetic by ground rule 1, so nothing legitimate can depend on
   the real values.
2. **Tests write a log file into the repo root.** `utils/logger._build_logger`
   defaults `LOG_FILE` to `langgraph_smart_reasoning.log` in CWD (pinned to
   repo root by conftest), so any test that reaches `log_event` — the
   orchestrator tests do — appends to a real file. That violates the "no test
   writes outside `tmp_path`" convention. Add `os.environ["LOG_FILE"] = ""`
   to the conftest env block (must be set before the first
   `utils.logger` import, same as the `LLM_*` vars).

**Acceptance:** suite passes with `LLM_URL=http://real-server:8000/v1
pytest -q`; `git status` after a full run shows no new/modified
`langgraph_smart_reasoning.log`.

---

## Phase B — Coverage gaps (test-only, new modules)

One test module per production module, same conventions as the existing suite
(parametrized tables, `match=` on messages, fakes over mocks-of-internals).

### B.1 `tests/test_api_server.py` — FastAPI endpoints

**Gated on Bug 1:** `api_server.py:27` imports `paralel_pipeline_graph` at
module level, so the server is unimportable today. Reuse the
`import_parallel_or_xfail` pattern (module-level `pytest.importorskip`-style
helper for `api_server`) so these tests xfail cleanly until Bug 1 lands, then
flip green.

Test through `fastapi.testclient.TestClient`, monkeypatching
`api_server._run_pipeline` (single seam shared by both endpoints — no graph,
no LLM):

| Case | Expected |
|---|---|
| `GET /health` | 200, `{"status": "ok"}` |
| `POST /run` happy path (fake `_run_pipeline` returns text) | 200, `final_output` echoed |
| `POST /run` with `task: ""` | 422 (pydantic `min_length=1`) |
| `POST /run`, `_run_pipeline` raises, `DEBUG` unset | 500; detail contains an error id, **not** the traceback |
| `POST /run`, `_run_pipeline` raises, `DEBUG=true` | 500; detail contains the exception text |
| `POST /run-async` → poll `GET /status/{id}` | 202 with `task_id`; status reaches `completed` with the fake output (drive the background task deterministically — `TestClient` runs the app's event loop; a fake `_run_pipeline` that resolves immediately is enough, no sleeps) |
| async run whose fake raises | status reaches `failed`, `error` populated per `DEBUG` |
| `GET /status/unknown` | 404, message names the id |

Note `DEBUG` is read at import (`api_server.py:33`) — parametrizing it needs
the fresh-import machinery from `test_config_loader.py`, or monkeypatch
`api_server.DEBUG` directly (simpler; do that).

### B.2 `tests/test_sub_agents_nodes.py` — prompt assembly and step execution

The meat of `agents/sub_agents_nodes.py` is currently only touched by one
xfail. Three groups:

1. **`_build_system_prompt`** (pure): skill bodies joined with `---`
   separators; context block present only when `context` is non-empty;
   datetime line present only when given; agent name/description
   interpolated.
2. **`run_sub_agent_async`** with `create_react_agent` monkeypatched to a
   recording fake and `create_mcp_client` forced to `None`:
   - only the skills in `skills_needed` have their bodies loaded (assert
     against a `skills_dir` fixture tree);
   - upstream context contains exactly the `depends_on` results;
   - the returned output went through `validate_step_output` — a fake agent
     answering `<script>alert(1)</script>` makes the step raise;
   - MCP branch: a fake client returning one extra tool ends up in the
     `create_react_agent(tools=...)` call alongside native tools.
3. **`sub_agent_node`** (sequential) with `run_sub_agent_async`
   monkeypatched: picks the first step whose deps are met, skips completed
   steps, returns `{"results": {n: out}}`; keeps the existing xfail for the
   blocked-forever guard.

### B.3 `tests/test_tools.py` — calculator, plotting, registry

- **`calculate`** (call with `verbose=False` — see C.4): representative
  expression table (precedence `2+3*4`, `^`, unary minus, postfix `!`,
  constants, multi-arg `nCr(5,2)` / `atan2`, scientific notation); error
  table (`SyntaxError` on unknown identifier / unbalanced parens / trailing
  token, `ValueError` on `fact(-1)`, `ZeroDivisionError` on `1/0`).
- **`plot_function`**: writes the PNG it names (call with
  `output_file=tmp_path/...`; never the default — see the CWD note in C.4);
  invalid expression raises `ValueError` naming the expression; expression
  using a blocked name (`__import__`, `open`) raises `ValueError` — pin the
  `eval` sandbox boundary the same way `test_sanitize.py` pins its detector.
- **`tools/agent_tools`**: `AGENT_TOOLS` covers every agent in the real
  config with resolved (non-None) tool objects; a config naming an unknown
  tool yields an agent whose list simply omits it (fresh-import via the
  `test_config_loader.py` fixture pattern, since `AGENT_TOOLS` is built at
  import).

### B.4 `tests/test_agent_mcp_tools.py` — client factory

`create_mcp_client` against monkeypatched `AGENT_CONFIG` dicts (patch
`agent_mcp_tools.AGENT_CONFIG` — no fresh import needed since the module
reads it at call time):

- agent with no `mcp_servers` → `None` (this is what makes B.2's no-MCP path
  the default);
- agent with servers → client constructed with each server as
  `streamable_http` and the right URL;
- ownership conflict (two agents claiming one server) → `ValueError` naming
  both agents — the defence-in-depth path that config-loader validation
  normally makes unreachable.

### B.5 `tests/test_logger.py` — structured events

With `LOG_FILE` pointed at `tmp_path` (fresh logger via a new
`logging.getLogger` name or handler reset — `_build_logger` memoizes on
`logger.handlers`): `log_event("x", a=1)` writes exactly one line of valid
JSON containing `event`, `ts`, and the kwargs; non-serializable kwargs fall
back to `str` (the `default=str`) rather than raising.

---

## Phase C — Items that need production changes

Each lands **with** its production fix (red-first test in the same PR), per
the suite's existing convention. None of these are test-only.

### C.1 `extract_json` array-precedence is a live bug, not a contract

**Problem:** `test_json_utils.py::test_array_takes_precedence_over_object`
pins behavior that breaks the orchestrator: for unfenced prose around
`{"plan": [...]}` the array regex wins, `extract_json` returns the inner
array, and `plan_json.get("plan")` then explodes (`list.get`) — wrapped into
the misleading parse error, burning a retry. A plausible LLM output triggers
it.

**Change (production):** make `utils/json_utils.extract_json` prefer the
*outermost* structure — try fenced first (already the case), then object,
then array — so prose + object returns the object.
**Change (tests):** rewrite the precedence test to assert the wrapping object
is returned (red-first, `strict=True` xfail until the fix); add the
orchestrator-level regression: fake LLM answering
`'Here is the plan: {"plan": [...]}'` (unfenced, with prose) produces a valid
plan.

### C.2 Orchestrator error wrapping — fold into the Bug 2 fix (roadmap 1.2)

**Problem:** the blanket `except Exception → ValueError("Failed to parse JSON
response: ...")` at `agents/orchestrator_node.py:88-89` relabels validation
errors, empty-plan errors, and genuine parse errors identically. Several
tests can only match on the misleading wrapper text.

**Change (production, in the Bug 2 PR):** scope the `try` to
`extract_json` only; let `PlanValidationError` and the empty-plan
`ValueError` propagate with their own messages (both are `ValueError`s, so
the node's retry policy is unaffected).
**Change (tests):** in the same PR, update the `match=` patterns in
`test_orchestrator_node.py` — `test_empty_plan_raises` matches
`"empty or invalid"` *without* the `"Failed to parse"` prefix,
`test_non_json_response_raises` keeps the parse-error match, A.1's
unknown-agent test matches the validator's own message.

### C.3 Guard the real graph's topology (stopgap until Phase 3)

**Problem:** `test_dispatch_dedup.py` exercises a hand-built *copy* of the
wiring. After Bug 1 is fixed, the only check on the actual module graph is
"it imports" — its edges could regress silently.

**Change (test, in the Bug 1 PR):** add
`test_module_graph_topology_matches_spec` — introspect the compiled graph
(`paralel_pipeline_graph.graph.get_graph()`) and assert: node set is exactly
`{orchestrator, scheduler, parallel_sub_agent, assemble}` (plus start/end);
plain edges `orchestrator→scheduler`, `parallel_sub_agent→scheduler`,
`assemble→END`; conditional edges leave **only** `scheduler`, targeting
exactly `{assemble, parallel_sub_agent}`. This replaces the "must change
together" comment with an executable check. Retire it when Phase 3's
`build(...)` factory lets the dedup test run the real builder with stub
nodes.

### C.4 `print()` in pipeline-reachable code (logging convention)

Two violations of the "no `print()` in pipeline code" convention, both
reachable from a live agent run:

- `tools/calculator.py:318` — `calculate` defaults `verbose=True` and prints
  a step-by-step breakdown to stdout on every tool call. Flip the default to
  `False` and route the breakdown through `log_event` (keep the `steps` list
  — it's good debugging data).
- `tools/agent_tools.py` — the unknown-tool warning is a `print`; make it
  `log_event("unknown_tool_in_config", ...)`.
- (Adjacent, same PR): `plotting_tool` hard-codes CWD-relative
  `artifacts/plot.png`; make the output directory env-configurable
  (`ARTIFACTS_DIR`, default `artifacts/`) so tests — and containers — can
  redirect it.

B.3's tests assert the post-fix behavior (no stdout capture needed;
`ARTIFACTS_DIR=tmp_path`).

---

## Keeping `docs/history/TESTING_GUIDE.md` in sync

The guide is still the referenced spec (`CLAUDE.md`, `tests/conftest.py`, and
`tests/_helpers.py` all point at it), so it stays a **living document** and
each phase PR updates the sections it invalidates — per the repo rule that
docs land with the feature. Concrete deltas this plan causes:

| Guide section | Delta | Caused by |
|---|---|---|
| Status header ("no tests exist yet") | Already stale — rewrite to "implemented (73 passed / 15 xfailed); this doc is the spec of record". Do this in the Phase A PR. | — |
| Shared infrastructure (conftest listing) | `setdefault` → hard assignment; add `LOG_FILE = ""` to the env block | A.3 |
| Conventions (xfail bullet) | Blanket `strict=False` → per-marker policy: deterministic pins are `strict=True`, imperative helper xfails stay non-strict | A.2 |
| `test_orchestrator_node.py` spec (unknown-agent bullet) | Must specify `match=` on the validator's message and the Bug 2 xfail marker | A.1 |
| `test_json_utils.py` spec | "array-before-object precedence" is a bug, not a contract to pin — replace with outermost-object-first behavior | C.1 |
| Test layout tree | Add the five Phase B modules (`test_api_server`, `test_sub_agents_nodes`, `test_tools`, `test_agent_mcp_tools`, `test_logger`) with one-line purposes | B.1–B.5 |
| `test_dispatch_dedup.py` spec | Add the module-graph topology assertion; drop the "keep a comment in both files" instruction once the executable check exists | C.3 |
| Known-bugs section | Its own instruction ("delete once green") triggers when the Bug 1/2 fix PRs land | Bug PRs |
| Definition of done | Add: full run leaves `git status` clean (no stray log file) and passes with real `LLM_*` shell vars exported | A.3 |

## Ordering and dependencies

```
A.1–A.3  (one PR, test-only, land first)
   │
   ├─ B.2–B.5  (test-only, independent of the bug fixes; any order)
   │     └─ B.3 asserts C.4's post-fix behavior → land B.3 with or after C.4
   ├─ C.4      (small production PR; unblocks full B.3)
   ├─ C.1      (production + tests, independent)
   │
   ├─ Bug 1 PR (roadmap 1.1) ──> carries C.3; unblocks B.1 and flips the
   │                             import xfails
   └─ Bug 2 PR (roadmap 1.2) ──> carries C.2; flips A.1 and the fence test
```

## Definition of done

1. No test passes for a reason other than the one in its name (A.1 is the
   known offender; re-audit the orchestrator module after C.2).
2. Every declarative xfail has explicit `strict=` and a plan-item reference;
   fixing a pinned bug without updating its marker fails the suite.
3. A full run leaves `git status` clean and never contacts a socket even
   with real `LLM_*` vars exported in the shell.
4. `api_server.py`, `agents/sub_agents_nodes.py`, `agent_mcp_tools.py`,
   `tools/`, and `utils/logger.py` each have a test module; the only
   remaining untested entry point is `run_pipeline.py` (thin CLI — accepted
   gap, revisit at Phase 3).
5. CI stays green at every step; `ruff check .` passes on each PR.
