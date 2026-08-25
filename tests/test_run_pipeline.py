"""Direct CLI and HTTP-client boundaries of the effort slider (plan step 2).

``run_pipeline`` (direct CLI) and ``api_client`` (HTTP client) propagate the
same per-run effort through the shared policy module; the server is the
authoritative normalizer, so the client only mirrors the preset names.
"""

import asyncio

import pytest

import run_pipeline
from execution_policy import DEFAULT_EFFORT


class _FakeGraph:
    """Records the run config of the last invoke — no LLM involved."""

    def __init__(self, state=None):
        self._state = state or {"final_output": "ok", "status": "completed",
                                "failed_steps": [], "step_stats": []}
        self.last_config = None

    def invoke(self, payload, config=None):
        self.last_config = config
        return self._state

    async def ainvoke(self, payload, config=None):
        self.last_config = config
        return self._state


@pytest.fixture()
def fake_build(monkeypatch):
    fake = _FakeGraph()
    monkeypatch.setattr(run_pipeline, "build_graph", lambda name: fake)
    return fake


# ---------------------------------------------------------------------------
# Argument parsing and config building
# ---------------------------------------------------------------------------

def test_parse_args_effort_is_case_insensitive_with_choices():
    args = run_pipeline._parse_args(["--effort", "Instant", "some task"])
    assert args.effort == "instant"
    args = run_pipeline._parse_args(["--graph", "yotta", "--effort", "THOROUGH", "t"])
    assert args.effort == "thorough"


def test_parse_args_omitted_effort_is_none():
    assert run_pipeline._parse_args(["task"]).effort is None


def test_parse_args_rejects_unknown_effort():
    with pytest.raises(SystemExit):
        run_pipeline._parse_args(["--effort", "extreme", "task"])


def test_run_config_carries_resolved_serializable_policy():
    config = run_pipeline._run_config("standard")
    configurable = config["configurable"]
    assert configurable["effort"] == "standard"
    policy = configurable["execution_policy"]
    assert policy["preset"] == "standard"
    assert policy["max_plan_steps"] == 8
    assert policy["max_replans"] == 1
    assert isinstance(policy["deadline"], float)
    assert configurable["task_id"]
    assert configurable["thread_id"]


def test_run_config_omitted_effort_resolves_unlimited():
    configurable = run_pipeline._run_config(None)["configurable"]
    assert configurable["effort"] == DEFAULT_EFFORT
    assert configurable["execution_policy"]["preset"] == "unlimited"


# ---------------------------------------------------------------------------
# run() / run_async() propagation and compatibility
# ---------------------------------------------------------------------------

def test_run_propagates_effort_into_graph_invoke_config(fake_build):
    result = run_pipeline.run("task", "yotta", "light")
    assert result["final_output"] == "ok"
    assert fake_build.last_config["configurable"]["effort"] == "light"


def test_run_async_propagates_effort_into_graph_invoke_config(fake_build):
    result = asyncio.run(run_pipeline.run_async("task", "yotta", "thorough"))
    assert result["final_output"] == "ok"
    assert fake_build.last_config["configurable"]["effort"] == "thorough"


def test_run_omitted_effort_resolves_unlimited(fake_build):
    run_pipeline.run("task", "yotta")
    assert fake_build.last_config["configurable"]["effort"] == "unlimited"


def test_run_rejects_verification_effort_on_legacy_graph(fake_build):
    with pytest.raises(ValueError, match="verification"):
        run_pipeline.run("task", "parallel", "standard")


def test_run_unlimited_on_legacy_graph_is_permitted(fake_build):
    run_pipeline.run("task", "parallel", "unlimited")
    assert fake_build.last_config["configurable"]["effort"] == "unlimited"


def test_run_instant_always_executes_on_yotta(fake_build):
    result = run_pipeline.run("task", "parallel", "instant")
    assert result["final_output"] == "ok"
    assert fake_build.last_config["configurable"]["effort"] == "instant"


# ---------------------------------------------------------------------------
# HTTP client body
# ---------------------------------------------------------------------------

def test_client_run_body_includes_effort_only_when_selected():
    from api_client import _run_body

    assert _run_body("t", None, "yotta", "standard") == {
        "task": "t", "graph": "yotta", "effort": "standard",
    }
    # omission intentionally lets the server resolve Unlimited
    assert _run_body("t", None, None, None) == {"task": "t"}
    assert _run_body("t", [{"filename": "a.md", "content": "x"}], None, "instant") == {
        "task": "t", "effort": "instant",
        "files": [{"filename": "a.md", "content": "x"}],
    }


def test_client_effort_choices_mirror_the_server_presets():
    from api_client import EFFORT_CHOICES
    from execution_policy import EFFORT_PRESETS

    assert list(EFFORT_CHOICES) == list(EFFORT_PRESETS)
