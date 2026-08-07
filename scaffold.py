"""Scaffold command — generate well-formed extension stubs (plan item 4.17).

Usage::

    python -m scaffold graph  my_graph     # -> graphs/my_graph_graph.py
    python -m scaffold tool   my_tool      # -> tools/my_tool.py
    python -m scaffold skill  my_skill     # -> skills/my-skill/SKILL.md
    python -m scaffold agent  my_agent     # appends to agents/agent_config.json

Every scaffold refuses to overwrite an existing file and prints the next
step (``pytest tests/test_extension_contracts.py``) after writing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Paths (relative to this module — the repo root)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
GRAPHS_DIR = _ROOT / "graphs"
TOOLS_DIR = _ROOT / "tools"
SKILLS_DIR = _ROOT / "skills"
AGENT_CONFIG_PATH = _ROOT / "agents" / "agent_config.json"

_NEXT_STEP = "pytest tests/test_extension_contracts.py"

# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

# Valid Python identifier — used for graph and tool names (they become modules).
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Valid directory / agent-key name — hyphenated-kebab plus snake_case.
_DIR_NAME_RE = re.compile(r"^[a-zA-Z0-9][-a-zA-Z0-9_]*$")


def _require_identifier(kind: str, name: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        sys.exit(
            f"error: '{kind}' name '{name}' is not a valid Python identifier. "
            f"Use letters, digits and underscores (e.g. 'my_{kind}')."
        )


def _require_dir_name(kind: str, name: str) -> None:
    if not _DIR_NAME_RE.match(name):
        sys.exit(
            f"error: '{kind}' name '{name}' is not a valid directory name. "
            f"Use letters, digits, hyphens and underscores (e.g. 'my-{kind}')."
        )


def _require_not_exists(path: Path) -> None:
    if path.exists():
        sys.exit(f"error: '{path}' already exists — refusing to overwrite.")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_GRAPH_TEMPLATE = '''\
"""{name} pipeline graph.

{description}
"""

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy
from agents.agent_states import AgentState, _transitive_dependents
from agents.orchestrator_node import make_orchestrator_agent
from agents.sub_agents_nodes import make_parallel_sub_agent_node
from assemble_node import assemble_node

GRAPH_DESCRIPTION = "{description}"


def {name}_router(state: dict):
    """Dispatch steps — one at a time in dependency order."""
    plan = state["plan"]
    results = state.get("results", {{}})
    current_datetime = state.get("current_datetime", "")

    ready = [
        s for s in plan
        if s["step"] not in results
        and all(d in results for d in s.get("depends_on", []))
    ]

    if not ready:
        unfinished = [s["step"] for s in plan if s["step"] not in results]
        if unfinished:
            raise RuntimeError(
                f"No step is ready to execute, but {{len(unfinished)}} step(s) "
                f"remain unfinished and are permanently blocked: {{unfinished}}."
            )
        return "assemble"

    next_step = ready[0]
    return Send(
        "sub_agent",
        {{"step": next_step, "results": results, "current_datetime": current_datetime}},
    )


def scheduler_node(state: dict) -> dict:
    """Synchronisation barrier: also propagates skips from failed steps."""
    failed = set(state.get("failed_steps", []))
    if not failed:
        return {{}}
    blocked = _transitive_dependents(state["plan"], failed)
    results = state.get("results", {{}})
    return {{
        "results": {{s: "[SKIPPED — dependency failed]" for s in blocked if s not in results}}
    }}


def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
    """Compile the {name} pipeline."""
    orchestrator = orchestrator or make_orchestrator_agent()
    sub_agent = sub_agent or make_parallel_sub_agent_node()

    builder = StateGraph(AgentState)
    builder.add_node(
        "orchestrator", orchestrator,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(ValueError,)),
    )
    builder.add_node(
        "sub_agent", sub_agent,
        retry_policy=RetryPolicy(max_attempts=2, retry_on=(Exception,)),
    )
    builder.add_node("assemble", assemble_node)
    builder.add_node("scheduler", scheduler_node)

    builder.set_entry_point("orchestrator")
    builder.add_edge("orchestrator", "scheduler")
    builder.add_edge("sub_agent", "scheduler")
    builder.add_conditional_edges(
        "scheduler", {name}_router, ["assemble", "sub_agent"]
    )
    builder.add_edge("assemble", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())
'''


_TOOL_TEMPLATE = '''\
from langchain_core.tools import tool


@tool
def {name}(query: str) -> str:
    """{description}

    Args:
        query: The input to process.
    """
    # TODO: implement the tool logic here.
    return f"{name}({{query}})"
'''


_SKILL_TEMPLATE = '''\
---
name: {name}
description: {description}
---

# {title}

TODO: Write the skill body — instructions the agent will follow when this
skill is active.  The body is injected into the system prompt of any sub-agent
whose step lists this skill.
'''


# Minimal valid agent entry that will pass test_every_agent_resolves.
_AGENT_STUB = """\
    "{name}": {{
        "description": "{description}",
        "tools": [],
        "mcp_servers": {{}}
    }}"""


# ---------------------------------------------------------------------------
# Scaffold functions
# ---------------------------------------------------------------------------

def _scaffold_graph(name: str, description: str, *,
                    graphs_dir: Path | None = None) -> Path:
    _require_identifier("graph", name)
    target_dir = graphs_dir if graphs_dir is not None else GRAPHS_DIR
    module_name = f"{name}_graph"
    path = target_dir / f"{module_name}.py"
    _require_not_exists(path)

    desc = description or f"Pipeline graph for {name}."
    content = _GRAPH_TEMPLATE.format(name=name, description=desc)
    path.write_text(content, encoding="utf-8")
    return path


def _scaffold_tool(name: str, description: str, *,
                   tools_dir: Path | None = None) -> Path:
    _require_identifier("tool", name)
    target_dir = tools_dir if tools_dir is not None else TOOLS_DIR
    path = target_dir / f"{name}.py"
    _require_not_exists(path)

    desc = description or f"A tool for {name}."
    content = _TOOL_TEMPLATE.format(name=name, description=desc)
    path.write_text(content, encoding="utf-8")
    return path


def _scaffold_skill(name: str, description: str, *,
                    skills_dir: Path | None = None) -> Path:
    _require_dir_name("skill", name)
    target_dir = skills_dir if skills_dir is not None else SKILLS_DIR
    skill_dir = target_dir / name
    path = skill_dir / "SKILL.md"
    if skill_dir.exists():
        sys.exit(f"error: '{skill_dir}' already exists — refusing to overwrite.")
    _require_not_exists(path)

    title = name.replace("-", " ").replace("_", " ").title()
    desc = description or f"Use this skill for {name} tasks."
    content = _SKILL_TEMPLATE.format(name=name, description=desc, title=title)
    skill_dir.mkdir(parents=True, exist_ok=False)
    path.write_text(content, encoding="utf-8")
    return path


def _scaffold_agent(name: str, description: str, *,
                    config_path: Path | None = None) -> Path:
    _require_dir_name("agent", name)
    target = config_path if config_path is not None else AGENT_CONFIG_PATH
    if not target.is_file():
        sys.exit(
            f"error: agent config not found at '{target}'. "
            f"Run from the repo root or set CONFIG_PATH."
        )

    # Read existing config, preserving formatting as much as possible.
    with open(target, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    if name in config:
        sys.exit(
            f"error: agent '{name}' already exists in '{target}'."
        )

    desc = description or f"Agent for {name} tasks."
    config[name] = {"description": desc, "tools": [], "mcp_servers": {}}

    with open(target, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4, ensure_ascii=False)
        fh.write("\n")

    return target


# Registry of scaffold kinds.
# Each entry: (func, arg_name, extra_kwargs)
_KINDS: dict[str, tuple[Callable[..., Path], str, dict[str, str]]] = {
    "graph": (_scaffold_graph, "name", {"graphs_dir": "graphs_dir"}),
    "tool": (_scaffold_tool, "name", {"tools_dir": "tools_dir"}),
    "skill": (_scaffold_skill, "name", {"skills_dir": "skills_dir"}),
    "agent": (_scaffold_agent, "name", {"config_path": "config_path"}),
}

# Public API — exported for test injection.
scaffold_graph = _scaffold_graph
scaffold_tool = _scaffold_tool
scaffold_skill = _scaffold_skill
scaffold_agent = _scaffold_agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scaffold",
        description="Generate well-formed extension stubs.",
    )
    sub = parser.add_subparsers(dest="kind", required=True)

    for kind, (func, arg_name, _extra) in _KINDS.items():
        p = sub.add_parser(kind, help=f"Scaffold a new {kind}")
        p.add_argument(arg_name, help=f"Name for the new {kind}")
        p.add_argument(
            "--description", "-d",
            default="",
            help="One-line description (uses a sensible default if omitted).",
        )

    return parser


def main(argv: list[str] | None = None, **overrides: Path) -> None:
    """Entry point — ``python -m scaffold <kind> <name>``.

    Keyword-only arguments override the target directories (for tests):
    ``graphs_dir``, ``tools_dir``, ``skills_dir``, ``config_path``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    kind: str = args.kind
    func, name_attr, extra_keys = _KINDS[kind]
    name: str = getattr(args, name_attr)
    description: str = getattr(args, "description", "")

    # Inject overrides that match the function's keyword names.
    kwargs = {kw: overrides[kw] for kw, ov in extra_keys.items() if ov in overrides}

    path = func(name, description, **kwargs)
    print(f"✅  Created {path}")
    print(f"    Next step: {_NEXT_STEP}")


if __name__ == "__main__":
    main()
