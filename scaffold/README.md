# Scaffold — extension stub generator

`python -m scaffold <kind> <name>` generates well-formed, immediately-testable
stubs for the four extension points (graphs, tools, skills, agents). It is the
implementation of [plan item 4.17](../IMPROVEMENT_PLAN.md#417-scaffold-command-done-2026-08-06).

Every scaffold refuses to overwrite an existing file and prints the next step
(`pytest tests/test_extension_contracts.py`) after writing.

## Quick start

```bash
# Graph — generates graphs/<name>_graph.py with the correct topology rules
python -m scaffold graph my-topology -d "Custom DAG with a critique loop"

# Tool — generates tools/<name>.py with a @tool-decorated function
python -m scaffold tool word-count -d "Count the words in a piece of text."

# Skill — generates skills/<name>/SKILL.md with YAML frontmatter
python -m scaffold skill code-review -d "Review code for bugs and style issues."

# Agent — appends a validated entry to agents/agent_config.json
python -m scaffold agent coder -d "Software engineer skilled in writing and reviewing code."
```

The `-d` / `--description` flag is optional — a sensible default is used when
omitted.

After scaffolding, verify with the conformance suite:

```bash
pytest tests/test_extension_contracts.py
```

## Kinds reference

| Kind | Name validation | Output | Can re-scaffold? |
|---|---|---|---|
| `graph` | Python identifier (`[a-zA-Z_][a-zA-Z0-9_]*`) | `graphs/{name}_graph.py` | No — refuses if file exists |
| `tool` | Python identifier | `tools/{name}.py` | No — refuses if file exists |
| `skill` | Directory name (`[a-zA-Z0-9][-a-zA-Z0-9_]*`) | `skills/{name}/SKILL.md` | No — refuses if directory exists |
| `agent` | Directory name | Appends to `agents/agent_config.json` | No — refuses if key exists |

Graph and tool names use Python identifier rules because they become importable
modules. Skill and agent names allow hyphens (kebab-case) because skills become
directories and agents become JSON keys.

## Architecture

```
scaffold/
    __init__.py       # Re-exports the public API
    __main__.py       # Entry point for `python -m scaffold`
    _core.py          # Logic: templates, validation, CLI
    README.md         # This file
```

### Public API

```python
from scaffold import (
    scaffold_graph,   # (name, description, *, graphs_dir=None) -> Path
    scaffold_tool,    # (name, description, *, tools_dir=None) -> Path
    scaffold_skill,   # (name, description, *, skills_dir=None) -> Path
    scaffold_agent,   # (name, description, *, config_path=None) -> Path
    main,             # (argv=None, **overrides) -> None  — CLI entry point
)
```

Each `scaffold_*` function accepts directory overrides so tests can target
temporary paths. When an override is `None`, the production path (relative to
the repo root) is used.

### Template system

Each kind has a string template with `{name}` and `{description}` placeholders,
formatted via `str.format()`. Templates live as module-level constants in
`_core.py`:

- `_GRAPH_TEMPLATE` — a complete `.py` file with the Phase 3.2 topology rules
  (scheduler barrier, single conditional edge, blocked-forever guard, `build()`
  factory). The router, scheduler, and build function all use the graph's name
  so two scaffolded graphs never collide.
- `_TOOL_TEMPLATE` — a `@tool`-decorated function with typed signature and
  docstring. The `@tool` decorator auto-generates `args_schema`, satisfying
  the 4.16 conformance contract.
- `_SKILL_TEMPLATE` — YAML frontmatter (`name`, `description`) followed by
  a markdown body. The title is derived from the name by replacing hyphens and
  underscores with spaces and applying `.title()`.
- `_AGENT_STUB` — a JSON fragment inserted into `agent_config.json`. Minimal
  valid shape: `description`, empty `tools` list, empty `mcp_servers` dict.

### Name validation

Two regex patterns in `_core.py`:

- `_IDENTIFIER_RE` — `^[a-zA-Z_][a-zA-Z0-9_]*$` — for graphs and tools (they
  become Python modules / function names).
- `_DIR_NAME_RE` — `^[a-zA-Z0-9][-a-zA-Z0-9_]*$` — for skills and agents (they
  become directory names / JSON keys). Hyphens are allowed; the first character
  must be alphanumeric.

Validation happens before any filesystem access. Invalid names exit with a
message naming the kind, the rejected value, and the allowed pattern.

### Path resolution

The repo root is computed as `Path(__file__).resolve().parent.parent` (this
file is `scaffold/_core.py`, so two levels up). Production targets are:

| Kind | Target path |
|---|---|
| Graph | `{root}/graphs/{name}_graph.py` |
| Tool | `{root}/tools/{name}.py` |
| Skill | `{root}/skills/{name}/SKILL.md` |
| Agent | `{root}/agents/agent_config.json` |

Tests inject temporary paths via the `*_dir` / `config_path` keyword arguments,
which take precedence over the production defaults.

### The conformance contract (plan item 4.16)

Scaffolded output is designed to pass `tests/test_extension_contracts.py`
unmodified:

- **Graph**: imports cleanly, exposes `build()` and `GRAPH_DESCRIPTION`
- **Tool**: has non-empty `description` and a valid JSON-serializable
  `args_schema`
- **Skill**: parses with the same `^---\n(.*?)\n---` regex the real
  `skill_loader` uses; frontmatter has `name` and `description`; body is
  non-empty
- **Agent**: JSON entry has string `description`, list `tools`, dict
  `mcp_servers`

This is enforced by `tests/test_scaffold.py::TestConformance` and is the
recommended next step printed after every scaffold.

## Extending — adding a new scaffold kind

1. **Write a template** — add a module-level string constant in `_core.py`.
   Use `{name}` and `{description}` as format placeholders.

2. **Write a `_scaffold_<kind>` function** — signature:
   ```python
   def _scaffold_<kind>(name: str, description: str, *,
                        override_dir: Path | None = None) -> Path:
   ```
   Validate the name, resolve the target path (respecting the override),
   check it doesn't exist, format the template, write, and return the path.

3. **Register in `_KINDS`** — add an entry:
   ```python
   "<kind>": (_scaffold_<kind>, "name", {"override_dir": "override_dir"})
   ```
   The tuple is `(function, arg_name_for_name, {kwarg: cli_name})`. The CLI
   subparser is auto-generated from this registry.

4. **Expose in `__init__.py`** — add the function to the imports and `__all__`.
   If the kind has a regex export (needed by tests), add that too.

5. **Add a conformance test** — in `tests/test_scaffold.py`, add a
   `test_<kind>_has_required_shape` method to `TestConformance`.

That's it. The CLI discovers the new kind automatically from `_KINDS` — no
parser changes needed.

## Tests

```bash
# Scaffold-specific tests (unit + conformance)
pytest tests/test_scaffold.py

# Full extension contract suite (verifies every registered extension)
pytest tests/test_extension_contracts.py
```

`tests/test_scaffold.py` covers:

- **Creation** — each kind generates the expected file with correct content
- **Default descriptions** — omitting `-d` produces a sensible fallback
- **Overwrite refusal** — re-scaffolding the same name exits with an error
- **Name validation** — invalid identifiers/dir-names are rejected at the regex
  level (parametrized)
- **CLI integration** — `main()` is exercised with argv-style lists
- **Conformance** — scaffolded output passes the 4.16 contract (dynamically
  imports scaffolded modules to check `build()`, `args_schema`, frontmatter
  parsing, JSON shape)
