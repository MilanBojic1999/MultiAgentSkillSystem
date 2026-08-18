"""Tests for the scaffold command (plan item 4.17).

Verifies that each scaffold kind produces a well-formed stub that passes
the corresponding 4.16 conformance test, that overwrites are refused, and
that name validation catches invalid inputs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Import the scaffold functions directly.
from scaffold import (
    scaffold_graph,
    scaffold_tool,
    scaffold_skill,
    scaffold_agent,
    main as scaffold_main,
    _IDENTIFIER_RE,
    _DIR_NAME_RE,
)


def _make_agent_config(path: Path, agents: dict | None = None) -> Path:
    """Write a minimal agent_config.json to *path* and return *path*."""
    path.write_text(json.dumps(agents or {}, indent=4, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


# ============================================================================
# Graph
# ============================================================================

class TestGraphScaffold:
    def test_creates_valid_module(self, tmp_path: Path):
        path = scaffold_graph("test_graph", "A test graph.", graphs_dir=tmp_path)
        assert path == tmp_path / "test_graph_graph.py"
        content = path.read_text()
        assert "def test_graph_router(" in content
        assert "def build(*, checkpointer=None, orchestrator=None, sub_agent=None):" in content
        assert 'GRAPH_DESCRIPTION = "A test graph."' in content

    def test_default_description(self, tmp_path: Path):
        path = scaffold_graph("mygraph", "", graphs_dir=tmp_path)
        content = path.read_text()
        assert "Pipeline graph for mygraph." in content

    def test_refuses_overwrite(self, tmp_path: Path):
        scaffold_graph("g1", "", graphs_dir=tmp_path)
        with pytest.raises(SystemExit, match="already exists"):
            scaffold_graph("g1", "", graphs_dir=tmp_path)

    def test_refuses_invalid_identifier(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="not a valid Python identifier"):
            scaffold_graph("bad-name", "", graphs_dir=tmp_path)

    def test_refuses_numeric_start(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="not a valid Python identifier"):
            scaffold_graph("123bad", "", graphs_dir=tmp_path)


# ============================================================================
# Tool
# ============================================================================

class TestToolScaffold:
    def test_creates_valid_module(self, tmp_path: Path):
        path = scaffold_tool("test_tool", "A test tool.", tools_dir=tmp_path)
        assert path == tmp_path / "test_tool.py"
        content = path.read_text()
        assert "@tool" in content
        assert "def test_tool(" in content
        assert "A test tool." in content

    def test_default_description(self, tmp_path: Path):
        path = scaffold_tool("mytool", "", tools_dir=tmp_path)
        content = path.read_text()
        assert "A tool for mytool." in content

    def test_refuses_overwrite(self, tmp_path: Path):
        scaffold_tool("t1", "", tools_dir=tmp_path)
        with pytest.raises(SystemExit, match="already exists"):
            scaffold_tool("t1", "", tools_dir=tmp_path)

    def test_refuses_invalid_identifier(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="not a valid Python identifier"):
            scaffold_tool("bad-name", "", tools_dir=tmp_path)


# ============================================================================
# Skill
# ============================================================================

class TestSkillScaffold:
    def test_creates_valid_skill(self, tmp_path: Path):
        path = scaffold_skill("test-skill", "A test skill.", skills_dir=tmp_path)
        assert path == tmp_path / "test-skill" / "SKILL.md"
        content = path.read_text()
        assert content.startswith("---")
        assert "name: test-skill" in content
        assert "description: A test skill." in content
        assert "# Test Skill" in content

    def test_default_description(self, tmp_path: Path):
        path = scaffold_skill("my-skill", "", skills_dir=tmp_path)
        content = path.read_text()
        assert "Use this skill for my-skill tasks." in content

    def test_refuses_overwrite_directory(self, tmp_path: Path):
        scaffold_skill("dup", "", skills_dir=tmp_path)
        with pytest.raises(SystemExit, match="already exists"):
            scaffold_skill("dup", "", skills_dir=tmp_path)

    def test_refuses_invalid_dir_name(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="not a valid directory name"):
            scaffold_skill("bad name!", "", skills_dir=tmp_path)

    def test_underscore_skill_name_accepted(self, tmp_path: Path):
        path = scaffold_skill("my_skill", "", skills_dir=tmp_path)
        assert path == tmp_path / "my_skill" / "SKILL.md"
        content = path.read_text()
        assert "name: my_skill" in content

    def test_title_is_derived_from_name(self, tmp_path: Path):
        path = scaffold_skill("hello-world", "", skills_dir=tmp_path)
        content = path.read_text()
        assert "# Hello World" in content


# ============================================================================
# Agent
# ============================================================================

class TestAgentScaffold:
    def test_creates_valid_entry(self, tmp_path: Path):
        config_path = _make_agent_config(tmp_path / "agent_config.json")
        result = scaffold_agent("test_agent", "A test agent.", config_path=config_path)
        assert result == config_path
        config = json.loads(config_path.read_text())
        assert "test_agent" in config
        assert config["test_agent"]["description"] == "A test agent."
        assert config["test_agent"]["tools"] == []
        assert config["test_agent"]["mcp_servers"] == {}

    def test_default_description(self, tmp_path: Path):
        config_path = _make_agent_config(tmp_path / "agent_config.json")
        scaffold_agent("myagent", "", config_path=config_path)
        config = json.loads(config_path.read_text())
        assert config["myagent"]["description"] == "Agent for myagent tasks."

    def test_refuses_duplicate_agent(self, tmp_path: Path):
        config_path = _make_agent_config(tmp_path / "agent_config.json")
        scaffold_agent("dup", "", config_path=config_path)
        with pytest.raises(SystemExit, match="already exists"):
            scaffold_agent("dup", "", config_path=config_path)

    def test_refuses_invalid_agent_name(self, tmp_path: Path):
        config_path = _make_agent_config(tmp_path / "agent_config.json")
        with pytest.raises(SystemExit, match="not a valid directory name"):
            scaffold_agent("bad name!", "", config_path=config_path)

    def test_preserves_existing_entries(self, tmp_path: Path):
        config_path = _make_agent_config(
            tmp_path / "agent_config.json",
            {"existing": {"description": "Already here.", "tools": ["calculate"], "mcp_servers": {}}},
        )
        scaffold_agent("new_agent", "New one.", config_path=config_path)
        config = json.loads(config_path.read_text())
        assert "existing" in config
        assert config["existing"]["tools"] == ["calculate"]
        assert "new_agent" in config

    def test_raises_when_config_missing(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.json"
        with pytest.raises(SystemExit, match="agent config not found"):
            scaffold_agent("test", "", config_path=missing)


# ============================================================================
# CLI integration
# ============================================================================

class TestCLI:
    def test_graph_via_main(self, tmp_path: Path):
        scaffold_main(["graph", "cli_graph", "-d", "CLI test."],
                      graphs_dir=tmp_path)
        path = tmp_path / "cli_graph_graph.py"
        assert path.is_file()
        assert "CLI test." in path.read_text()

    def test_tool_via_main(self, tmp_path: Path):
        scaffold_main(["tool", "cli_tool", "-d", "CLI tool test."],
                      tools_dir=tmp_path)
        path = tmp_path / "cli_tool.py"
        assert path.is_file()
        assert "CLI tool test." in path.read_text()

    def test_skill_via_main(self, tmp_path: Path):
        scaffold_main(["skill", "cli-skill", "-d", "CLI skill test."],
                      skills_dir=tmp_path)
        path = tmp_path / "cli-skill" / "SKILL.md"
        assert path.is_file()
        assert "CLI skill test." in path.read_text()

    def test_agent_via_main(self, tmp_path: Path):
        config_path = _make_agent_config(tmp_path / "agent_config.json")
        scaffold_main(["agent", "cli_agent", "-d", "CLI agent test."],
                      config_path=config_path)
        config = json.loads(config_path.read_text())
        assert config["cli_agent"]["description"] == "CLI agent test."

    def test_missing_kind_is_rejected(self):
        with pytest.raises(SystemExit):
            scaffold_main([])

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(SystemExit):
            scaffold_main(["badkind", "x"])


# ============================================================================
# Name validation (unit-level — no filesystem)
# ============================================================================

class TestNameValidation:
    @pytest.mark.parametrize("name", ["my_graph", "MyGraph", "_private", "x1", "X"])
    def test_valid_identifiers(self, name: str):
        assert _IDENTIFIER_RE.match(name)

    @pytest.mark.parametrize("name", ["bad-name", "123abc", "has space", "a.b", ""])
    def test_invalid_identifiers(self, name: str):
        assert not _IDENTIFIER_RE.match(name)

    @pytest.mark.parametrize("name", ["my-skill", "my_skill", "skill1", "a"])
    def test_valid_dir_names(self, name: str):
        assert _DIR_NAME_RE.match(name)

    @pytest.mark.parametrize("name", ["bad name", "a.b", "-starthyphen", ""])
    def test_invalid_dir_names(self, name: str):
        assert not _DIR_NAME_RE.match(name)


# ============================================================================
# Conformance — scaffolded output passes the 4.16 contract
# ============================================================================

class TestConformance:
    """Smoke: scaffold each kind and assert it would satisfy test_extension_contracts."""

    def test_graph_passes_compile_check(self, tmp_path: Path):
        """A scaffolded graph must import and expose ``build()``."""
        path = scaffold_graph("conf_graph", "Conformance graph.", graphs_dir=tmp_path)

        # Add tmp_path to sys.path so the import resolves.
        sys.path.insert(0, str(tmp_path.parent))
        try:
            # Import the module dynamically.
            import importlib
            mod = importlib.import_module(f"{tmp_path.name}.conf_graph_graph")
            assert callable(mod.build)
            assert mod.GRAPH_DESCRIPTION == "Conformance graph."
        finally:
            sys.path.pop(0)

    def test_tool_has_required_attributes(self, tmp_path: Path):
        """A scaffolded tool must have description and args_schema."""
        path = scaffold_tool("conf_tool", "Conformance tool.", tools_dir=tmp_path)

        sys.path.insert(0, str(tmp_path.parent))
        try:
            import importlib
            mod = importlib.import_module(f"{tmp_path.name}.conf_tool")
            tool = getattr(mod, "conf_tool")
            assert tool.description is not None
            assert "Conformance tool." in tool.description
            assert tool.args_schema is not None
            # JSON-serialisable
            schema = tool.args_schema.model_json_schema()
            json.dumps(schema)  # must not raise
        finally:
            sys.path.pop(0)

    def test_skill_has_valid_frontmatter(self, tmp_path: Path):
        """A scaffolded skill must parse the same way skill_loader does."""
        path = scaffold_skill("conf-skill", "Conformance skill.", skills_dir=tmp_path)

        import re
        import yaml
        content = path.read_text()
        m = re.match(r"^---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
        assert m is not None, "frontmatter block missing"
        fm = yaml.safe_load(m.group(1))
        assert fm["name"] == "conf-skill"
        assert fm["description"] == "Conformance skill."
        body = m.group(2).strip()
        assert body, "body must be non-empty"

    def test_agent_has_required_shape(self, tmp_path: Path):
        """A scaffolded agent must have non-empty description + valid tools/mcp."""
        config_path = _make_agent_config(tmp_path / "agent_config.json")
        scaffold_agent("conf_agent", "Conformance agent.", config_path=config_path)

        config = json.loads(config_path.read_text())
        entry = config["conf_agent"]
        assert isinstance(entry["description"], str) and entry["description"].strip()
        assert isinstance(entry["tools"], list)
        assert isinstance(entry["mcp_servers"], dict)
        for _sn, sc in entry["mcp_servers"].items():
            # Each MCP server must be a dict with url or command, or a plain string.
            if isinstance(sc, str):
                continue
            assert isinstance(sc, dict)
            assert "url" in sc or "command" in sc
