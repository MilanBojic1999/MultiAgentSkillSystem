"""Plan 4.5: the artifact-path convention for file-producing tools.

Pins three things:

  * ``get_artifact_path`` keys the output directory by the run config
    (task id, thread-id fallback, root fallback) and rejects path traversal;
  * ``plotting_tool`` writes into the run's directory with a unique filename
    per invocation — the regression for two plotting steps silently
    overwriting each other's ``artifacts/plot.png``;
  * the worker nodes thread the run config into the production sub-agent
    runner (so tools can read ``configurable``) while test-injected stubs
    keep their 3-argument signature.
"""

import asyncio
from pathlib import Path

import pytest

from utils.artifacts import artifacts_root, get_artifact_path

CFG = {"configurable": {"task_id": "task-1", "thread_id": "thr-1"}}

STEP = {
    "step": {"step": 1, "agent": "researcher", "subtask": "t",
             "skills_needed": [], "depends_on": []},
    "results": {},
    "current_datetime": "",
}


# ---------------------------------------------------------------------------
# get_artifact_path
# ---------------------------------------------------------------------------

def test_task_id_keys_the_run_directory():
    assert get_artifact_path("plot.png", CFG) == artifacts_root() / "task-1" / "plot.png"


def test_thread_id_is_the_fallback_key():
    assert (get_artifact_path("plot.png", {"configurable": {"thread_id": "thr-9"}})
            == artifacts_root() / "thr-9" / "plot.png")


def test_without_run_config_files_land_under_the_root():
    assert get_artifact_path("plot.png") == artifacts_root() / "plot.png"


def test_root_is_env_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    assert get_artifact_path("plot.png", CFG) == tmp_path / "task-1" / "plot.png"


@pytest.mark.parametrize("bad", ["../plot.png", "a/b.png", "a\\b.png", "..", "-x.png"])
def test_invalid_artifact_names_raise(bad):
    with pytest.raises(ValueError, match="artifact name"):
        get_artifact_path(bad, CFG)


@pytest.mark.parametrize("bad", ["../evil", "a/b"])
def test_invalid_run_ids_raise(bad):
    with pytest.raises(ValueError, match="run id"):
        get_artifact_path("plot.png", {"configurable": {"task_id": bad}})


# ---------------------------------------------------------------------------
# plotting_tool — the collision regression
# ---------------------------------------------------------------------------

def _invoke_plot(config, tmp_path):
    from tools.plotting import plotting_tool

    return plotting_tool.invoke({"expression": "sin(x)"}, config=config)


def test_plotting_tool_writes_into_the_per_run_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    result = _invoke_plot(CFG, tmp_path)
    path = Path(result)
    assert path.parent == tmp_path / "task-1"
    assert path.name.startswith("plot-") and path.name.endswith(".png")
    assert path.is_file()


def test_two_plotting_invocations_do_not_collide(monkeypatch, tmp_path):
    """4.5 regression: two plotting calls must never overwrite each other."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    first = _invoke_plot(CFG, tmp_path)
    second = _invoke_plot(CFG, tmp_path)
    assert first != second
    assert Path(first).is_file() and Path(second).is_file()


# ---------------------------------------------------------------------------
# Worker nodes thread the run config into the real sub-agent runner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory_name", [
    "make_sub_agent_node", "make_parallel_sub_agent_node",
])
def test_worker_node_passes_run_config_to_the_real_runner(monkeypatch, factory_name):
    from agents import sub_agents_nodes as san

    seen = {}

    async def spy_run(step, results, current_datetime="", llm=None, config=None,
                      policy=None):
        seen["config"] = config
        return step["step"], f"out-{step['step']}", {
            "input_tokens": 0, "output_tokens": 0, "tool_calls": 0,
        }

    monkeypatch.setattr(san, "run_sub_agent_async", spy_run)
    node = getattr(san, factory_name)()
    cfg = {"configurable": {"task_id": "task-42"}}

    out = asyncio.run(node.afunc(STEP, config=cfg))
    print(seen)
    assert seen["config"] is cfg, "the run config was not threaded to the sub-agent"
    assert out["results"] == {1: "out-1"}
