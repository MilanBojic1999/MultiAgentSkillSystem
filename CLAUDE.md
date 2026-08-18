# agent_skills — Multi-Agent Skills Pipeline

This directory is a standalone Python project. The parent directory's
`CLAUDE.md` describes a **different project** (an Angular learning platform) —
ignore it when working here.

## What this is

A plan-and-execute multi-agent pipeline on LangGraph 1.x:

1. **Orchestrator** (`agents/orchestrator_node.py`) — one LLM call that
   decomposes the user task into a JSON plan: a DAG of steps, each assigned to
   a specialist agent with a list of skills.
2. **Sub-agents** (`agents/sub_agents_nodes.py`) — each step runs a
   `create_react_agent` combining native tools (auto-discovered from `tools/`),
   MCP tools (`agent_mcp_tools.py`), and skill bodies injected into the system
   prompt (`skill_loader.py`).
3. **Assemble** (`assemble_node.py`) — concatenates step outputs into
   `final_output`.

Graphs live in `graphs/` and are **auto-discovered**: any module there defining
`build(*, checkpointer=None, orchestrator=None, sub_agent=None)` is registered
under its file name minus a trailing `_pipeline_graph`/`_graph`/`_pipeline`
suffix (`GRAPH_NAME` overrides it, `GRAPH_DESCRIPTION` documents it). Resolve
one with `graphs.build_graph(name)` — never import a graph module directly from
an entry point. Ships with `parallel` (via `Send`, the default) and
`sequential`. Entry points: `run_pipeline.py` (CLI, `--graph`/`--list-graphs`)
and `api_server.py` (FastAPI: `/graphs`, `/run`, `/run-async`,
`/status/{id}`; the `graph` request field selects the topology). Agent
definitions live in `agents/agent_config.json`, loaded
and validated by `config_loader.py`. Env config comes from `.env` (template:
`.env.example`): `LLM_URL`, `LLM_MODEL`, `LLM_KEY`, `CONFIG_PATH`,
`ORCHESTRATOR_TEMPERATURE`.

## Conventions

- **LLM clients** are constructed only through `llm_factory.create_llm(...)` —
  never instantiate `ChatOpenAI` directly.
- **Logging** goes through `utils/logger.log_event` (structured JSON) — no
  `print()` in pipeline code.
- **Extension is config/data, not code**: tools are auto-discovered from
  `tools/*.py` (`@tool` functions), agents are entries in
  `agents/agent_config.json`, skills are `skills/<name>/SKILL.md` files, and
  graphs are `graphs/*.py` modules exposing `build()`. See
  the "Contributor Recipes" section of `README.md` before adding code for any
  of these.
- **Import-time side effects**: modules load config and skills at import (CWD-
  relative), so `tests/conftest.py` pins the CWD and sets dummy `LLM_*` env vars
  *before* importing any pipeline module — preserve that ordering when adding
  tests. LLM clients are no longer built at import: `create_llm` is called
  inside the node factories, so importing `graphs`, `agents` or `api_server`
  needs no LLM configuration; building a graph does.
- **Graph topology rules**: workers route back through the no-op scheduler
  node via plain `add_edge`; the fan-out router is the only conditional edge
  (conditional edges run once per completed task and `Send` dispatches are not
  deduplicated). Path maps are plain lists of node names.
- **Known naming warts**: `utils/senitize.py` (sic) is kept for compatibility
  until item 1.5 of the plan lands. The graph module was renamed to
  `graphs/parallel_pipeline_graph.py`; the root `paralel_pipeline_graph.py`
  (sic) is only an import-compat shim and exposes no compiled `graph`.

## Tests

`pytest` from the repo root — hermetic, no network, no `.env`, no live LLM.
Desired-but-unimplemented behavior is marked `xfail(strict=False)`; live-endpoint
tests carry the `integration` marker and are excluded by default. Spec:
`docs/history/TESTING_GUIDE.md`.

## Roadmap and known bugs

`IMPROVEMENT_PLAN.md` (repo root) is the **active roadmap**; the files in
`docs/history/` are stale design history, except `TESTING_GUIDE.md` (the test
suite's spec). The two bugs the plan's item 2.3 called out — the parallel graph
failing at import, and `validate_plan` called with the wrong arity — are fixed
and their regression tests are green; don't reintroduce either. Phase 1, 2 and
Phase 3 items 3.1–3.3 have landed.
