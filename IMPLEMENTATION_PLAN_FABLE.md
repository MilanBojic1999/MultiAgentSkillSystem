# Implementation Plan — Multi-Agent Pipeline Package

Roadmap for evolving this package from a working prototype into a maintainable,
extensible multi-agent framework that new graphs (and new developers) can build
on. Based on a full codebase review (2026-07-01).

Phases are ordered by dependency and leverage: each phase makes the next one
cheaper or safer. Within a phase, tasks are independent unless noted.

---

## Phase 0 — Security & repo hygiene *(blocking; ~half a day)*

Do this before sharing the repo with anyone.

- [ ] **Remove secrets from git.**
  - Move the hardcoded bearer token in `yotta_tool.py:8` to `.env`
    (`YOTTA_API_KEY`), read it via `os.getenv`.
  - Remove `.env` from git tracking; add a documented `.env.example` with
    every variable (`LLM_URL`, `LLM_MODEL`, `LLM_KEY`, `CONFIG_PATH`,
    `YOTTA_API_KEY`, LangSmith vars) and no values.
  - **Rotate both keys** — they are permanently in git history.
- [ ] **Add a proper `.gitignore`**: `venv/`, `__pycache__/`, `.env`,
  `debug_log.log`, `*.log`, `scraped_pages/`, `artifacts/`.
- [ ] **Clean stray files**: delete or move to `examples/` —
  `old_agent.py`, `testing_main.py`, `example_file.txt`,
  `example_output.txt`, `single_agent_pipeline.txt`, committed log files.
- [ ] **Fix the port mismatch**: Dockerfile exposes 8000, direct run uses
  8999 (`api_server.py:271`). Pick one (suggest 8000) everywhere.
- [ ] Add a `LICENSE` file if external adoption is intended.

**Done when:** `git clone` + `.env.example` → running server, with no secret
in the repo and no noise in `git status`.

---

## Phase 1 — Correctness fixes in the current graphs *(~1–2 days)*

Independent of the restructure; fixes real bugs in `yotta_graph.py`'s flow.

- [ ] **Fix replanning end-to-end** (currently broken in three ways):
  - Add `verifier_report: str` and `replan_count: int` to `AgentState`
    (`agent_states.py`). The orchestrator prompt already references both
    (`agents/orchestrator_node.py:39`) but never receives them.
  - Pass verifier notes + replan count into the orchestrator on the
    FAILED route; enforce the 3-pass cap in the router, not the prompt.
  - On replan, clear stale step results (or version them) — the `results`
    reducer only merges, so today `should_continue` sees
    `len(results) > len(plan)` and **skips the new plan entirely**
    (`yotta_graph.py:43`).
- [ ] **Replace count-based routing with set-based routing**:
  `remaining = {s["step"] for s in plan} - set(results)`. Route on
  emptiness, not `len()` arithmetic.
- [ ] **Detect deadlock**: if steps remain but none is ready
  (bad `depends_on` from the LLM), fail with a clear error instead of
  spinning to the recursion limit (`yotta_graph.py:26-35`).
- [ ] **Stop the verifier verdict leaking into the writer**: store it in a
  dedicated state field, not `results[len(results)]`
  (`yotta_graph.py:111`) — today it gets synthesized into the final
  document as if it were a research finding.
- [ ] **Validate the plan at the boundary** (Pydantic model for `PlanStep`):
  unique step ids, `depends_on` refers to existing steps and is acyclic,
  `agent` exists in `AGENT_CONFIG`, `skills_needed` exist in the skill
  index. Fail fast with a retryable message — today an unknown agent
  name surfaces as `StopIteration` deep in
  `agents/sub_agents_nodes.py:50`, and unknown skills are silently
  dropped.
- [ ] **Move search results out of the task string**: add
  `search_results: str` to state instead of embedding
  `"\n\n## Search results"` into `task` and re-splitting it in
  `writer_node` (`yotta_graph.py:153-186`).
- [ ] **Smaller fixes:**
  - Planner/verifier temperature 0.9 → ≤0.2 (strict-JSON roles);
    keep higher temperature for the writer only.
  - `max_tokens=8048` → 8192 (or config).
  - Bare `except:` in `_call_yotta_sync` (`yotta_tool.py:57-60`):
    catch specific errors, log, and don't reference `response` before
    assignment.
  - Remove dead `citatitaion_node` or wire it in properly (see Phase 4);
    fix the typo either way.
  - Bound checkpointer growth: per-request `MemorySaver` threads are
    never freed in the long-running server.

**Done when:** a forced-FAILED verification actually re-plans with feedback
and executes the new plan; a plan naming a bogus agent fails with one clear
error; verifier text no longer appears in final documents.

---

## Phase 2 — Test harness & CI *(~1–2 days; do before restructuring)*

The safety net that makes Phase 3 a refactor instead of a rewrite.

- [ ] **Unit tests for pure logic** (no LLM needed):
  `extract_json`, `parse_yotta_results`, routers (`should_continue`,
  `after_verify`, `fan_out_router`), plan validation, dependency
  scheduling, `validate_step_output`, file-truncation helpers.
- [ ] **Fake-LLM smoke test per graph**: stub `ChatOpenAI` / the agent
  runner, feed a canned plan + canned step outputs, assert the graph
  reaches `final_output` on: happy path, empty-plan direct route,
  FAILED-verdict replan route.
- [ ] **API tests** via `fastapi.testclient`: `/run`, `/run-async` +
  `/status`, `/run-stream` framing.
- [ ] **Tooling**: `pytest`, `ruff` (lint + format), optional `mypy` on
  `core/` once it exists.
- [ ] **CI** (GitHub Actions): install, lint, test on push/PR.
- [ ] **Evaluation harness** (can run later, scaffold now):
  `evals/` with golden question → expected-facts pairs and an
  LLM-as-judge script producing a score per graph. This is how every
  later graph change gets measured instead of eyeballed.

**Done when:** `pytest` passes locally and in CI; a deliberate break in a
router fails a test.

---

## Phase 3 — Package restructure: the node library *(~3–5 days)*

The core enabler for "create new graphs easily." Three graph files currently
copy-paste node logic with drift (two `sub_agent_node`s, two
`assemble_node`s, duplicated routing and `ChatOpenAI` blocks).

- [ ] **Adopt a package layout** (installable via `pyproject.toml`, pinned
  deps / lockfile):

  ```
  agent_pipeline/
  ├── core/
  │   ├── state.py          # AgentState + PlanStep (Pydantic), reducers
  │   ├── llm.py            # get_llm(role, streaming) — single factory
  │   └── config.py         # config_loader + agent roster
  ├── nodes/                # node FACTORIES: options in → node fn out
  │   ├── orchestrate.py
  │   ├── run_step.py       # sequential + Send-based fan-out variants
  │   ├── verify.py
  │   ├── write.py
  │   └── cite.py
  ├── graphs/               # thin composition files (~30 lines each)
  │   ├── quick.py          # today's pipeline_graph
  │   ├── parallel.py       # today's paralel_pipeline_graph
  │   └── deep_research.py  # today's yotta_graph
  ├── tools/                # unchanged (registry pattern already good)
  ├── skills/               # unchanged (SKILL.md pattern already good)
  ├── server/               # api_server + streaming
  └── tests/
  ```

- [ ] **One LLM factory** `get_llm(role, streaming)` reading per-role
  model/temperature from config — kills the duplicated `ChatOpenAI`
  blocks and enables per-agent model selection (Phase 5).
- [ ] **Node factories, not modules with side effects**: each node is
  produced by a function taking explicit options; no more import-time
  `load_skills()` / env reads scattered across modules.
- [ ] **Graph registry**: `graphs.get("deep-research")` → compiled graph;
  API accepts an optional `graph` parameter.
- [ ] **Rename typos while moving** (free during a restructure):
  `senitize.py` → `sanitize.py`, `paralel_…` → `parallel_…`,
  `citatitaion` → `citation`.
- [ ] **Structured logging**: replace `print("+"*50)` debugging with the
  logger, keyed by run id + step id.
- [ ] **Checkpointer strategy**: `SqliteSaver` for resumable runs, or no
  checkpointer on the stateless API path.

**Done when:** a new graph = one file in `graphs/` composing existing nodes;
all Phase 2 tests still pass; old entry points re-export or are deleted.

---

## Phase 4 — Product features on the solid base *(ordered by value)*

1. - [ ] **Plan approval (human-in-the-loop).** LangGraph `interrupt` after
     the orchestrator; API returns the plan, user approves/edits, run
     resumes via checkpointer. Highest-trust UX feature.
2. - [ ] **Structured progress events in the stream**: typed SSE events
     (`plan_created`, `step_started`, `step_finished`, `verdict`,
     `token`) alongside text. Prerequisite for streaming the parallel
     graph (tokens need step labels) and for any real frontend. Document
     the protocol (see Phase 5 docs).
3. - [ ] **The flagship graph: parallel fan-out + verify + writer.**
     Today's parallel graph has no quality gate; the quality graph is
     sequential. First new graph to build from the node library; compare
     against `deep_research` with the eval harness.
4. - [ ] **Citation pipeline.** Make research workers return the structured
     JSON their skill already specifies (findings + sources), thread a
     real `source_map` through state, wire the (renamed) citation node
     after the writer.
5. - [ ] **Token/cost accounting + budgets**: accumulate usage metadata per
     step; expose per-run totals in API responses; optional budget cap
     that short-circuits to the writer. Enforce the per-worker tool-call
     budget the yotta-researcher skill already documents.
6. - [ ] **Step timeouts + run cancellation**: `asyncio.wait_for` around
     step execution; `DELETE /run/{task_id}` cancels the background task.
7. - [ ] **Follow-up conversations**: accept a client `thread_id` to
     continue a completed run's state — the bridge to the chat-based
     learning platform.
8. - [ ] **Subquery caching** on `call_yotta` (question → result, TTL).
9. - [ ] **Platform agents** (learning-platform tie-in): quiz generator,
     lecture summarizer, project writing agent — each a config entry +
     SKILL.md once the plumbing is trusted.

---

## Phase 5 — Developer adoption polish *(parallel to Phase 4)*

- [ ] **README rewrite for accuracy**: `yotta_graph`/`deep_research` is the
  flagship — the architecture diagram must show yotta pre-search,
  verifier loop, and writer; fix the stale "parallel is default" claim.
- [ ] **Cookbook docs** (each a short recipe — the config-driven design is
  the selling point, show it off):
  - `docs/adding-an-agent.md` — edit `agent_config.json`, done.
  - `docs/adding-a-skill.md` — drop a `SKILL.md` folder, done.
  - `docs/adding-a-tool.md` — registry pattern.
  - `docs/building-a-graph.md` — compose nodes from the library.
  - `docs/streaming-protocol.md` — SSE event/marker spec for frontends.
- [ ] **One-command quickstart** that actually works: `docker compose up`
  or `make dev` → health check passes; curl examples for every endpoint.
- [ ] **Makefile/justfile**: `dev`, `test`, `lint`, `eval`.
- [ ] **Pre-commit hooks** (ruff, secret scan).
- [ ] **CONTRIBUTING.md** + versioning/changelog once external
  contributors exist.
- [ ] **Production hardening flags**: don't return tracebacks in HTTP 500
  bodies outside dev mode; tighten CORS.

---

## Suggested sequence at a glance

| Order | Phase | Effort | Unblocks |
|-------|-------|--------|----------|
| 1 | 0 — Security & hygiene | ~0.5 day | sharing the repo at all |
| 2 | 1 — Correctness fixes | 1–2 days | trustworthy output; replanning |
| 3 | 2 — Tests & CI | 1–2 days | safe refactoring; measurable changes |
| 4 | 3 — Node library restructure | 3–5 days | cheap new graphs; features 1–4 |
| 5 | 4 — Features | incremental | product value |
| 6 | 5 — Adoption polish | incremental | contributors |

Phases 0–2 are worth doing even if the restructure never happens. Phase 4
items 5, 6, 8 and all of Phase 5 don't strictly require Phase 3 — they can
be pulled forward if priorities shift.
