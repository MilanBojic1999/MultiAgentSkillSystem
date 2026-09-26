"""Tests for ``agent_mcp_tools`` — transport detection, server-map building,
ownership validation, client construction (Phase 4.6), and per-attempt MCP
call serialization (``serialize_tool_calls``).

No live MCP server is required — we assert on the dict passed to
``MultiServerMCPClient``.
"""

import asyncio
import inspect
from typing import Annotated

import pytest
from langchain_core.tools import InjectedToolArg, StructuredTool

from agent_mcp_tools import (
    _build_server_map,
    _check_mcp_ownership,
    create_mcp_client,
    serialize_tool_calls,
)


# ---------------------------------------------------------------------------
# _build_server_map — per-server transport detection
# ---------------------------------------------------------------------------

def test_url_server_maps_to_streamable_http():
    """A server config with ``url`` gets ``transport: "streamable_http"``."""
    result = _build_server_map({"yotta": {"url": "http://host:8001/mcp"}})
    assert result == {"yotta": {"transport": "streamable_http", "url": "http://host:8001/mcp"}}


def test_command_server_maps_to_stdio():
    """A server config with ``command`` gets ``transport: "stdio"``."""
    result = _build_server_map({"fs": {"command": "npx", "args": ["-y", "server"]}})
    assert result == {"fs": {"transport": "stdio", "command": "npx", "args": ["-y", "server"]}}


def test_command_server_with_optional_keys():
    """Optional stdio keys (env, cwd) pass through unchanged."""
    result = _build_server_map(
        {"fs": {"command": "python", "args": ["-m", "srv"], "env": {"FOO": "1"}, "cwd": "/tmp"}}
    )
    assert result["fs"]["transport"] == "stdio"
    assert result["fs"]["command"] == "python"
    assert result["fs"]["env"] == {"FOO": "1"}
    assert result["fs"]["cwd"] == "/tmp"


def test_mixed_servers_detect_transport_independently():
    """One agent with both url and command servers — each gets its own transport."""
    result = _build_server_map({
        "web": {"url": "http://host/mcp"},
        "fs":  {"command": "npx", "args": ["-y", "srv"]},
    })
    assert result["web"]["transport"] == "streamable_http"
    assert result["fs"]["transport"] == "stdio"


def test_empty_server_map_returns_empty_dict():
    assert _build_server_map({}) == {}


def test_missing_both_keys_raises_naming_server_and_keys():
    """Neither ``url`` nor ``command`` → ValueError."""
    with pytest.raises(ValueError, match="must have 'url' or 'command'"):
        _build_server_map({"bad": {"transport": "stdio"}})  # transport alone is not enough


def test_both_keys_raises():
    """Both ``url`` and ``command`` → ValueError (ambiguous)."""
    with pytest.raises(ValueError, match="both 'url' and 'command'"):
        _build_server_map({"bad": {"url": "http://h", "command": "x"}})


def test_non_dict_value_raises_type_error():
    """A list or int where a dict config is expected."""
    with pytest.raises(TypeError, match="expected a dict"):
        _build_server_map({"bad": ["not", "a", "dict"]})


# ---------------------------------------------------------------------------
# _check_mcp_ownership — defence-in-depth duplicate check
# ---------------------------------------------------------------------------

def test_no_conflict_when_server_unique(monkeypatch):
    """No error when the server belongs only to the requesting agent."""
    monkeypatch.setattr(
        "agent_mcp_tools.AGENT_CONFIG",
        {"alice": {"mcp_servers": {"s1": {}}}, "bob": {"mcp_servers": {}}},
    )
    _check_mcp_ownership("alice", {"s1"})  # does not raise


def test_conflict_raises_naming_both_agents(monkeypatch):
    """Conflict → ValueError naming the server and both agents."""
    monkeypatch.setattr(
        "agent_mcp_tools.AGENT_CONFIG",
        {"alice": {"mcp_servers": {"shared": {}}}, "bob": {"mcp_servers": {"shared": {}}}},
    )
    with pytest.raises(ValueError, match=r"\['shared'\]"):
        _check_mcp_ownership("alice", {"shared"})


def test_ownership_check_skips_self(monkeypatch):
    """The requesting agent's own servers are not flagged as conflicts."""
    monkeypatch.setattr(
        "agent_mcp_tools.AGENT_CONFIG",
        {"alice": {"mcp_servers": {"s1": {}}}},
    )
    _check_mcp_ownership("alice", {"s1"})  # does not raise


# ---------------------------------------------------------------------------
# create_mcp_client — integration of the above
# ---------------------------------------------------------------------------

def test_returns_none_when_agent_has_no_mcp_servers(monkeypatch):
    monkeypatch.setattr(
        "agent_mcp_tools.AGENT_CONFIG",
        {"alice": {"mcp_servers": {}}},
    )
    assert create_mcp_client("alice") is None


def test_returns_none_when_agent_missing_from_config(monkeypatch):
    monkeypatch.setattr("agent_mcp_tools.AGENT_CONFIG", {})
    assert create_mcp_client("no_such_agent") is None


def test_returns_client_with_correct_server_map(monkeypatch):
    """create_mcp_client builds a MultiServerMCPClient whose ``connections``
    attribute holds the per-server transport maps."""
    monkeypatch.setattr(
        "agent_mcp_tools.AGENT_CONFIG",
        {"alice": {"mcp_servers": {"web": {"url": "http://h/mcp"}, "fs": {"command": "x", "args": []}}}},
    )
    client = create_mcp_client("alice")
    assert client is not None
    # MultiServerMCPClient stores connections under ``self.connections``
    assert client.connections["web"] == {"transport": "streamable_http", "url": "http://h/mcp"}
    assert client.connections["fs"] == {"transport": "stdio", "command": "x", "args": []}


def test_create_mcp_client_rejects_duplicate_ownership(monkeypatch):
    """The runtime ownership check still fires inside create_mcp_client."""
    monkeypatch.setattr(
        "agent_mcp_tools.AGENT_CONFIG",
        {"alice": {"mcp_servers": {"shared": {}}}, "bob": {"mcp_servers": {"shared": {}}}},
    )
    with pytest.raises(ValueError, match=r"\['shared'\].*declared by both"):
        create_mcp_client("alice")


# ---------------------------------------------------------------------------
# serialize_tool_calls — one MCP call at a time per agent attempt
# ---------------------------------------------------------------------------

def _mcp_like_tool(name="search", tracker=None):
    """A StructuredTool shaped like a langchain_mcp_adapters tool: dict
    ``args_schema``, ``content_and_artifact``, and a coroutine taking an
    injected ``runtime`` plus ``**arguments``."""

    async def call_tool(
        runtime: Annotated[object | None, InjectedToolArg()] = None,
        **arguments,
    ):
        if tracker is not None:
            tracker["in_flight"] += 1
            tracker["max_in_flight"] = max(tracker["max_in_flight"], tracker["in_flight"])
            await asyncio.sleep(0.01)
            tracker["in_flight"] -= 1
        return "ok", None

    return StructuredTool(
        name=name,
        description="Search.",
        args_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata={"server": "yotta"},
    )


def test_serialize_tool_calls_keeps_tool_and_leaves_original_untouched():
    original = _mcp_like_tool()
    original_coroutine = original.coroutine

    [copy] = serialize_tool_calls([original])

    assert copy is not original
    assert copy.name == "search"
    assert copy.args_schema == original.args_schema
    assert copy.response_format == "content_and_artifact"
    assert copy.metadata == {"server": "yotta"}
    assert original.coroutine is original_coroutine
    assert copy.coroutine is not original_coroutine
    # functools.wraps keeps the original signature visible, so the injected
    # ``runtime`` argument is still detected through the wrapper.
    assert copy.coroutine.__wrapped__ is original_coroutine
    assert "runtime" in inspect.signature(copy.coroutine).parameters


def test_serialize_tool_calls_passes_tools_without_coroutine_through():
    def echo(q: str) -> str:
        """Echo."""
        return q

    sync_tool = StructuredTool.from_function(func=echo)
    assert serialize_tool_calls([sync_tool])[0] is sync_tool


def test_serialize_tool_calls_runs_one_call_at_a_time_across_tools():
    """All copies from one call share one lock — even different tools."""
    tracker = {"in_flight": 0, "max_in_flight": 0}
    tools = serialize_tool_calls(
        [_mcp_like_tool("a", tracker), _mcp_like_tool("b", tracker)]
    )

    async def run():
        return await asyncio.gather(
            *(t.coroutine(q="x") for t in tools for _ in range(2))
        )

    results = asyncio.run(run())

    assert results == [("ok", None)] * 4
    assert tracker["max_in_flight"] == 1
