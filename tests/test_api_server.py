"""Slice 4: structured execution result and partial API contract.

Covers the HTTP surface with a fake graph injected via ``api_server._get_graph``
— no live LLM, no network. Pins:

  * sync ``completed``/``partial`` both return HTTP 200 with typed fields;
  * a contained failure never discards independent successful output;
  * fatal errors are HTTP 500 (sync) / terminal ``failed`` (async) — never
    ``partial``;
  * async polling transitions ``running`` → terminal while preserving output,
    failed/skipped lists and step statistics;
  * response schemas reject malformed status/statistics values.
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api_server


def _row(step, status, **overrides):
    row = {
        "step": step,
        "agent": "researcher",
        "status": status,
        "duration_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tool_calls": 0,
    }
    row.update(overrides)
    return row


COMPLETED_STATE = {
    "status": "completed",
    "final_output": "full output",
    "failed_steps": [],
    "step_stats": [_row(1, "completed")],
}

PARTIAL_STATE = {
    "status": "partial",
    "final_output": "useful partial output",
    "failed_steps": [2],
    "step_stats": [
        _row(1, "completed", input_tokens=10, output_tokens=5, tool_calls=1),
        _row(2, "failed"),
        _row(3, "skipped"),
    ],
}


class _FakeGraph:
    """Graph stand-in: returns ``state`` after optionally waiting on a gate.

    Records the config of the last ``ainvoke`` so tests can assert what the
    server put into ``configurable`` (plan 4.5).
    """

    def __init__(self, state=None, exc=None, release=None):
        self._state = state
        self._exc = exc
        self._release = release
        self.last_config = None

    async def ainvoke(self, payload, config=None):
        self.last_config = config
        if self._release is not None:
            await self._release.wait()
        if self._exc is not None:
            raise self._exc
        return self._state


@pytest.fixture()
def client(monkeypatch):
    """A TestClient whose graph lookup is a fake — no LLM is ever built."""
    monkeypatch.setattr(api_server, "DEBUG", False)
    monkeypatch.setattr(
        api_server, "_get_graph", lambda name=None: _FakeGraph(COMPLETED_STATE)
    )
    with TestClient(api_server.app) as c:
        yield c


def _wait_terminal(client: TestClient, task_id: str, timeout: float = 5.0) -> dict:
    """Poll /status until the task leaves ``running``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/status/{task_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} never reached a terminal state")


# ---------------------------------------------------------------------------
# Synchronous /run
# ---------------------------------------------------------------------------

def test_run_all_success_returns_200_completed(client):
    resp = client.post("/run", json={"task": "t"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["final_output"] == "full output"
    assert body["failed_steps"] == []
    assert body["skipped_steps"] == []
    assert body["step_stats"][0]["status"] == "completed"


def test_run_contained_failure_returns_200_partial(client, monkeypatch):
    """A contained step failure is not a transport failure (Slice 4)."""
    monkeypatch.setattr(
        api_server, "_get_graph", lambda name=None: _FakeGraph(PARTIAL_STATE)
    )
    resp = client.post("/run", json={"task": "t"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "partial"
    # Independent successful output is preserved, not discarded
    assert body["final_output"] == "useful partial output"
    # Failed and skipped step lists are accurate
    assert body["failed_steps"] == [2]
    assert body["skipped_steps"] == [3]
    # Stats are typed and step-ordered
    assert [s["step"] for s in body["step_stats"]] == [1, 2, 3]
    assert body["step_stats"][0]["status"] == "completed"
    assert body["step_stats"][1]["status"] == "failed"
    assert body["step_stats"][2]["status"] == "skipped"


def test_run_fatal_error_returns_500_not_partial(client, monkeypatch):
    monkeypatch.setattr(
        api_server, "_get_graph", lambda name=None: _FakeGraph(exc=ValueError("boom"))
    )
    resp = client.post("/run", json={"task": "t"})
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "Pipeline failed" in detail
    # DEBUG is off — no raw exception or traceback leaks to the client
    assert "boom" not in detail
    assert "Traceback" not in detail


def test_run_fatal_error_debug_mode_includes_details(client, monkeypatch):
    monkeypatch.setattr(api_server, "DEBUG", True)
    monkeypatch.setattr(
        api_server, "_get_graph", lambda name=None: _FakeGraph(exc=ValueError("boom"))
    )
    resp = client.post("/run", json={"task": "t"})
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


def test_run_unknown_graph_returns_400(client, monkeypatch):
    def _unknown(name=None):
        raise ValueError("Unknown graph 'nope'. Available: parallel, sequential")

    monkeypatch.setattr(api_server, "_get_graph", _unknown)
    resp = client.post("/run", json={"task": "t", "graph": "nope"})
    assert resp.status_code == 400
    assert "nope" in resp.json()["detail"]


def test_sync_run_returns_task_id_and_keys_artifacts(client, monkeypatch):
    """Plan 4.5: the sync response carries the id of the run's artifact directory."""
    fake = _FakeGraph(COMPLETED_STATE)
    monkeypatch.setattr(api_server, "_get_graph", lambda name=None: fake)
    resp = client.post("/run", json={"task": "t"})
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    assert task_id
    assert fake.last_config["configurable"]["task_id"] == task_id


# ---------------------------------------------------------------------------
# Asynchronous /run-async + /status
# ---------------------------------------------------------------------------

def test_async_accepted_with_202(client):
    resp = client.post("/run-async", json={"task": "t"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["task_id"]
    assert body["status"] == "started"


def test_async_polling_running_to_partial_preserves_output(client, monkeypatch):
    """Polling transitions ``running`` → ``partial`` with output and stats intact."""
    release = asyncio.Event()
    monkeypatch.setattr(
        api_server, "_get_graph", lambda name=None: _FakeGraph(PARTIAL_STATE, release=release)
    )

    resp = client.post("/run-async", json={"task": "t"})
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    # The background task is gated, so the first poll is deterministically running
    first = client.get(f"/status/{task_id}").json()
    assert first["status"] == "running"
    assert first["final_output"] is None

    release.set()
    status = _wait_terminal(client, task_id)
    assert status["status"] == "partial"
    assert status["final_output"] == "useful partial output"
    assert status["failed_steps"] == [2]
    assert status["skipped_steps"] == [3]
    assert status["step_stats"] == PARTIAL_STATE["step_stats"]
    assert status["error"] is None


def test_async_fatal_error_produces_terminal_failed(client, monkeypatch):
    monkeypatch.setattr(
        api_server, "_get_graph", lambda name=None: _FakeGraph(exc=ValueError("boom"))
    )
    resp = client.post("/run-async", json={"task": "t"})
    task_id = resp.json()["task_id"]

    status = _wait_terminal(client, task_id)
    assert status["status"] == "failed"   # fatal — never "partial"
    assert status["final_output"] is None
    assert status["step_stats"] == []
    assert status["error"]
    assert "boom" not in status["error"]  # safe public message, DEBUG off


def test_async_unknown_graph_returns_400(client, monkeypatch):
    def _unknown(name=None):
        raise ValueError("Unknown graph 'nope'")

    monkeypatch.setattr(api_server, "_get_graph", _unknown)
    resp = client.post("/run-async", json={"task": "t", "graph": "nope"})
    assert resp.status_code == 400


def test_status_unknown_task_returns_404(client):
    resp = client.get("/status/does-not-exist")
    assert resp.status_code == 404


def test_async_run_passes_task_id_as_artifact_key(client, monkeypatch):
    """Plan 4.5: async runs key artifacts by the task id the client already has."""
    fake = _FakeGraph(COMPLETED_STATE)
    monkeypatch.setattr(api_server, "_get_graph", lambda name=None: fake)
    resp = client.post("/run-async", json={"task": "t"})
    task_id = resp.json()["task_id"]
    _wait_terminal(client, task_id)
    assert fake.last_config["configurable"]["task_id"] == task_id


# ---------------------------------------------------------------------------
# GET /artifacts/{task_id}/{filename} (plan 4.5)
# ---------------------------------------------------------------------------

def test_artifacts_endpoint_serves_existing_file(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    (tmp_path / "task-1").mkdir()
    (tmp_path / "task-1" / "plot-abcd1234.png").write_bytes(b"fake-png")
    resp = client.get("/artifacts/task-1/plot-abcd1234.png")
    assert resp.status_code == 200
    assert resp.content == b"fake-png"


def test_artifacts_endpoint_404_for_missing_file(client, monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    resp = client.get("/artifacts/task-1/nope.png")
    assert resp.status_code == 404
    assert "nope.png" in resp.json()["detail"]


def test_artifacts_endpoint_rejects_invalid_segments(client, monkeypatch, tmp_path):
    """A filename failing segment validation is a 400, never a file read.

    The traversal cases ("..", "../x") are pinned at the helper level in
    ``tests/test_artifacts.py``; here an unambiguously invalid segment
    (a space) exercises the endpoint's 400 branch without depending on how
    HTTP clients normalize dot segments in URLs.
    """
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    resp = client.get("/artifacts/task-1/bad%20name.png")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Response-schema validation
# ---------------------------------------------------------------------------

def test_response_models_reject_malformed_status_and_stats():
    """Typed schemas reject malformed status/statistics values (Slice 4)."""
    with pytest.raises(ValidationError):
        api_server.RunResponse(status="bogus", final_output="x")

    with pytest.raises(ValidationError):
        api_server.RunResponse(status="completed", final_output="x", step_stats="nope")

    with pytest.raises(ValidationError):
        api_server.RunResponse(
            status="completed",
            final_output="x",
            step_stats=[{"step": "one", "agent": "a", "status": "completed",
                         "duration_s": 0, "input_tokens": 0, "output_tokens": 0,
                         "tool_calls": 0}],
        )

    with pytest.raises(ValidationError):
        api_server.RunResponse(
            status="completed",
            final_output="x",
            step_stats=[{"step": 1, "agent": "a", "status": "bogus",
                         "duration_s": 0, "input_tokens": 0, "output_tokens": 0,
                         "tool_calls": 0}],
        )

    with pytest.raises(ValidationError):
        api_server.StatusResponse(task_id="t", status="bogus")


def test_response_models_accept_typed_partial_result():
    resp = api_server.RunResponse(**{
        "status": "partial",
        "final_output": "x",
        "failed_steps": [2],
        "skipped_steps": [3],
        "step_stats": PARTIAL_STATE["step_stats"],
    })
    assert resp.status == "partial"
    assert resp.failed_steps == [2]
    assert resp.skipped_steps == [3]
    assert resp.step_stats[0].input_tokens == 10
    assert resp.step_stats[2].status == "skipped"
