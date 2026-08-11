"""Extension-point conformance tests (plan item 4.16).

Parametrized over the live registries so each test grows automatically as a
fork adds extensions. A deliberately broken skill, agent, tool or graph turns
exactly one parametrized case red, with a message naming the offender.

This file *is* the extension contract for the baseline. Run it after adding
any extension to verify it is well-formed before the failure surfaces deep
inside a graph run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from config_loader import AGENT_CONFIG
from tools import TOOL_REGISTRY


# ============================================================================
# Graphs — every registered graph must compile
# ============================================================================

def _fake_orchestrator(state: dict) -> dict:
    return {"plan": [], "results": {}, "current_step": 0}


async def _fake_worker(state: dict) -> dict:
    return {}


_FAKE_OVERRIDES = {
    "orchestrator": _fake_orchestrator,
    "sub_agent": _fake_worker,
}


@pytest.mark.parametrize("graph_name", sorted(__import__("graphs").available_graphs()))
def test_every_graph_compiles(graph_name: str):
    """Every shipped graph must compile with fake nodes and no network.

    A graph that fails to import or whose ``build()`` raises will turn exactly
    this parametrized case red, naming the graph.
    """
    from graphs import build_graph

    built = build_graph(graph_name, checkpointer=MemorySaver(), **_FAKE_OVERRIDES)
    assert built is not None, f"build_graph({graph_name!r}) returned None"


# ============================================================================
# Agents — every agent in the config must resolve
# ============================================================================

def _valid_mcp_shape(server_conf: Any, agent_name: str, server_name: str) -> None:
    """Assert a single MCP server entry has a recognised shape (4.6)."""
    if isinstance(server_conf, str):
        # Legacy — plain-string URL, accepted for backward compatibility.
        return
    if not isinstance(server_conf, dict):
        raise AssertionError(
            f"Agent '{agent_name}', MCP server '{server_name}': "
            f"expected dict or string, got {type(server_conf).__name__}"
        )
    has_url = "url" in server_conf
    has_command = "command" in server_conf
    if not has_url and not has_command:
        raise AssertionError(
            f"Agent '{agent_name}', MCP server '{server_name}': "
            f"config dict must have 'url' or 'command'. "
            f"Got keys: {sorted(server_conf)}"
        )
    if has_url and has_command:
        raise AssertionError(
            f"Agent '{agent_name}', MCP server '{server_name}': "
            f"config dict has both 'url' and 'command' — pick one."
        )


# The set of accepted keys in an agent's "llm" block (mirrors config_loader).
_LLM_KEYS = frozenset({"model", "url", "api_key_env", "temperature", "max_tokens"})


def _valid_llm_block(llm_block: Any, agent_name: str) -> None:
    """Assert the optional llm block only contains recognised keys (4.3)."""
    if llm_block is None:
        return
    if not isinstance(llm_block, dict):
        raise AssertionError(
            f"Agent '{agent_name}': 'llm' must be a dict, "
            f"got {type(llm_block).__name__}"
        )
    bad = set(llm_block) - _LLM_KEYS
    if bad:
        raise AssertionError(
            f"Agent '{agent_name}': unknown key(s) in 'llm' block: "
            f"{sorted(bad)}. Accepted keys: {sorted(_LLM_KEYS)}."
        )


@pytest.mark.parametrize("agent_name", sorted(AGENT_CONFIG))
def test_every_agent_resolves(agent_name: str):
    """Every agent must have a non-empty description, and every tool,
    MCP server and optional LLM block it references must be valid.

    A missing tool, a malformed MCP server config, or a typo in the
    ``llm`` block turns exactly this parametrized case red.
    """
    cfg: dict[str, Any] = AGENT_CONFIG[agent_name]

    # --- description ---------------------------------------------------------
    desc = cfg.get("description", "")
    assert isinstance(desc, str) and desc.strip(), (
        f"Agent '{agent_name}': 'description' must be a non-empty string, "
        f"got {desc!r}"
    )

    # --- tools ---------------------------------------------------------------
    for tool_name in cfg.get("tools", []):
        assert tool_name in TOOL_REGISTRY, (
            f"Agent '{agent_name}' references tool '{tool_name}' which is not "
            f"in TOOL_REGISTRY. Available tools: {sorted(TOOL_REGISTRY)}"
        )

    # --- mcp_servers ---------------------------------------------------------
    servers = cfg.get("mcp_servers", {})
    assert isinstance(servers, dict), (
        f"Agent '{agent_name}': 'mcp_servers' must be a dict, "
        f"got {type(servers).__name__}"
    )
    for server_name, server_conf in servers.items():
        _valid_mcp_shape(server_conf, agent_name, server_name)

    # --- llm block -----------------------------------------------------------
    _valid_llm_block(cfg.get("llm"), agent_name)


# ============================================================================
# Skills — every skill must parse
# ============================================================================

def _skill_names_and_dirs():
    """Return ``(name, skill_dir_name)`` pairs for every shipped skill.

    The ``name`` comes from the YAML frontmatter; ``skill_dir_name`` is the
    name of the directory the SKILL.md lives in.
    """
    from pathlib import Path
    from skill_loader import load_skills

    # load_skills() returns (index dict, pairs dict).
    index, pairs = load_skills()
    # pairs maps skill frontmatter name -> directory path
    for fm_name, dir_path in sorted(pairs.items()):
        dir_name = Path(dir_path).name
        yield fm_name, dir_name, index[fm_name]


def _skill_ids():
    """Human-readable parametrize ids for the skill test."""
    return [f"{fm_name} (dir: {dir_name})" for fm_name, dir_name, _ in _skill_names_and_dirs()]


@pytest.mark.parametrize(
    ("skill_name", "dir_name", "frontmatter"),
    sorted(_skill_names_and_dirs(), key=lambda x: x[0]),
    ids=_skill_ids(),
)
def test_every_skill_parses(skill_name: str, dir_name: str, frontmatter: dict):
    """Every skill must have valid frontmatter and a non-empty body.

    Checks that:
    - The frontmatter ``name`` matches the directory name.
    - The ``description`` is present and non-empty.
    - The body is non-empty.
    """
    from skill_loader import load_skills, load_skills_body

    _, pairs = load_skills()

    # Name matches its directory
    assert skill_name == dir_name, (
        f"Skill '{skill_name}': frontmatter name '{skill_name}' does not "
        f"match its directory '{dir_name}'. The ``name`` key in the SKILL.md "
        f"frontmatter must equal the enclosing folder name."
    )

    # Description is present and non-empty
    desc = frontmatter.get("description", "")
    assert isinstance(desc, str) and desc.strip(), (
        f"Skill '{skill_name}': 'description' in frontmatter must be a "
        f"non-empty string, got {desc!r}"
    )

    # Body is non-empty
    body = load_skills_body(pairs, skill_name)
    assert body.strip(), (
        f"Skill '{skill_name}': body must be non-empty (the markdown content "
        f"after the closing '---' frontmatter delimiter)"
    )


# ============================================================================
# Tools — every tool must be usable
# ============================================================================

def _tool_ids():
    """Human-readable ids: tool name + type."""
    return [f"{name} ({type(tool).__name__})" for name, tool in sorted(TOOL_REGISTRY.items())]


@pytest.mark.parametrize(
    ("tool_name", "tool"),
    sorted(TOOL_REGISTRY.items(), key=lambda x: x[0]),
    ids=_tool_ids(),
)
def test_every_tool_is_usable(tool_name: str, tool: Any):
    """Every tool must have a non-empty description and a JSON-serialisable
    ``args_schema``.
    """
    # Non-empty description
    desc = getattr(tool, "description", None)
    assert isinstance(desc, str) and desc.strip(), (
        f"Tool '{tool_name}': description must be a non-empty string, "
        f"got {desc!r}"
    )

    # JSON-serialisable args_schema
    schema = getattr(tool, "args_schema", None)
    assert schema is not None, (
        f"Tool '{tool_name}': has no 'args_schema'. Every @tool-decorated "
        f"function must have type annotations on its parameters."
    )

    # Verify the schema is a Pydantic model serialisable to JSON Schema.
    try:
        json_schema = schema.model_json_schema()
    except Exception as exc:
        raise AssertionError(
            f"Tool '{tool_name}': args_schema.model_json_schema() raised "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    assert isinstance(json_schema, dict), (
        f"Tool '{tool_name}': model_json_schema() returned "
        f"{type(json_schema).__name__}, expected dict"
    )

    # Round-trip through json to confirm it really is serialisable.
    try:
        json.dumps(json_schema)
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"Tool '{tool_name}': args_schema JSON Schema is not "
            f"JSON-serialisable: {exc}"
        ) from exc
