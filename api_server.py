"""
FastAPI REST API server for the multi-agent LangGraph pipeline.

Endpoints:
    GET  /health          — health check
    GET  /graphs          — list the graphs discovered in graphs/
    POST /run             — run the pipeline synchronously (blocking)
    POST /run-async       — start a pipeline run in the background
    GET  /status/{task_id} — check status of an async run
    GET  /artifacts/{task_id}/{filename} — serve a generated artifact file
"""

import asyncio
import os
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load env before importing pipeline modules (they read os.getenv at import time)
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver

from graphs import build_graph, graph_descriptions
from agents.agent_states import ExecutionStatus, StepStatus, get_current_datetime_str
from assemble_node import pipeline_result
from utils.artifacts import get_artifact_path
from utils.logger import log_event

from pipeline_entry import (
    build_task_string,
    build_files_state,
    UnsupportedFileTypeError,
    EmptyExtractedTextError,
)

from streaming import stream_pipeline
from yotta_tool import call_yotta, parse_yotta_results

# Graph used when a request does not name one.
DEFAULT_GRAPH = os.getenv("DEFAULT_GRAPH", "parallel").strip() or "parallel"

# When true, API error responses include the full traceback; otherwise clients
# get a generic message + id and the traceback stays in the server log.
DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class FileInput(BaseModel):
    """A file attached as input context for the pipeline."""
    filename: str = Field(..., description="Name of the file (e.g. 'notes.md')")
    content: str = Field(..., description="File contents as plain UTF-8 text, or base64-encoded when encoding='base64'")
    encoding: str | None = Field(
        default=None,
        description="'base64' if the content is base64-encoded binary; omit for plain UTF-8 text",
    )


class RunRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        description="The natural-language task to run through the pipeline",
        examples=["Calculate sin(pi/4) + cos(pi/4) and plot both functions"],
    )
    
    files: list[FileInput] | None = Field(
        default=None,
        description="Optional list of files to include as input context",
    )

    graph: str | None = Field(
        default=None,
        description=f"Which graph in graphs/ to run (default: {DEFAULT_GRAPH}). "
                    f"See GET /graphs.",
        examples=["parallel", "sequential"],
    )


class StepStatsModel(BaseModel):
    """One typed per-step statistics row (Slice 4)."""
    step: int
    agent: str
    status: StepStatus
    duration_s: float
    input_tokens: int
    output_tokens: int
    tool_calls: int
    files: list[FileInput] | None = Field(
        default=None,
        description="Optional list of files to include as input context",
    )


class RunResponse(BaseModel):
    """Typed synchronous execution result.

    ``status`` is ``completed`` or ``partial`` on HTTP 200; a fatal error
    before assembly is an HTTP 500 error response instead (Slice 4).
    ``task_id`` keys the run's artifact directory (plan 4.5) — the client
    fetches generated files from ``GET /artifacts/{task_id}/{filename}``.
    """
    status: ExecutionStatus
    final_output: str
    failed_steps: list[int] = []
    skipped_steps: list[int] = []
    step_stats: list[StepStatsModel] = []
    task_id: str = ""


class GraphInfo(BaseModel):
    name: str
    description: str = ""
    default: bool = False


class GraphsResponse(BaseModel):
    graphs: list[GraphInfo]


class AsyncRunResponse(BaseModel):
    task_id: str
    status: Literal["started"] = "started"


class StatusResponse(BaseModel):
    """Typed asynchronous poll response: ``running`` until terminal.

    Terminal states are ``completed``, ``partial`` (output and stats are
    preserved and retrievable) and ``failed`` (``error`` carries the safe
    public message) — Slice 4.
    """
    task_id: str
    status: ExecutionStatus
    final_output: str | None = None
    failed_steps: list[int] = []
    skipped_steps: list[int] = []
    step_stats: list[StepStatsModel] = []
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# In-memory async-task store
# ---------------------------------------------------------------------------

_task_store: dict[str, dict[str, Any]] = {}
_task_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Compiled-graph cache
# ---------------------------------------------------------------------------
# Graphs are compiled once per process and reused across requests. The
# checkpointer is created here rather than inside each graph's build() so a
# durable backend (SqliteSaver/PostgresSaver — plan item 4.7) can be swapped in
# at one place.

_graph_cache: dict[str, Any] = {}
_checkpointer: Any = None


def _make_checkpointer():
    """The checkpointer shared by every graph in this process."""
    return MemorySaver()


def _get_graph(name: str | None):
    """Return the compiled graph for ``name``, building it on first use.

    Raises ``ValueError`` (naming the available graphs) for an unknown name.
    """
    name = (name or DEFAULT_GRAPH).strip() or DEFAULT_GRAPH
    if name not in _graph_cache:
        _graph_cache[name] = build_graph(name, checkpointer=_checkpointer)
        log_event("api_graph_compiled", graph=name)
    print("graph name:", name)
    return _graph_cache[name]


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    global _checkpointer
    _checkpointer = _make_checkpointer()
    # Compile the default graph up front so a broken graph module surfaces at
    # startup instead of on the first request. Other graphs compile on demand.
    _get_graph(DEFAULT_GRAPH)
    yield
    # Clean up any lingering tasks
    async with _task_lock:
        _task_store.clear()
    _graph_cache.clear()
    _checkpointer = None


app = FastAPI(
    title="Agent Skills Pipeline",
    description="Multi-agent LangGraph pipeline API — orchestrates LLM sub-agents with tools and skills",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow all origins (containerised service; tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pipeline runner (shared by sync and async endpoints)
# ---------------------------------------------------------------------------

def _resolve_graph_or_400(graph_name: str | None) -> None:
    """Compile the requested graph early so a bad name is a 400, not a 500."""
    try:
        _get_graph(graph_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


async def _run_pipeline(task: str, files: list[FileInput] | None, graph_name: str | None = None,
                        task_id: str | None = None) -> RunResponse:
    """Run the selected LangGraph pipeline and return its typed result.

    ``status`` is ``completed`` when every planned step finished and
    ``partial`` when containment failed or skipped steps while assembly still
    produced usable output. Fatal planner, graph or configuration errors
    raise here — they are transport-level failures, never partial results.

    ``task_id`` keys the run's artifact directory (plan 4.5); a fresh one is
    generated for synchronous runs and returned in the response so the client
    can fetch generated files from ``GET /artifacts/{task_id}/{filename}``.
    """
    graph = _get_graph(graph_name)
    run_id = task_id or uuid.uuid4().hex[:12]
    config = {"configurable": {
        "thread_id": f"api-{uuid.uuid4().hex[:8]}",
        "task_id": run_id,
    }}

    files_state = build_files_state(files)

    result = await graph.ainvoke(
        {"task": task, "current_datetime": get_current_datetime_str(), "files": files_state,},
        config=config,
    )
    return RunResponse(task_id=run_id, **pipeline_result(result))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Return ok if the service is alive."""
    return HealthResponse(status="ok")


@app.get("/graphs", response_model=GraphsResponse)
async def list_graphs():
    """List every graph discovered in ``graphs/`` — any module there defining
    ``build()`` shows up here without further registration."""
    return GraphsResponse(
        graphs=[
            GraphInfo(name=name, description=description, default=(name == DEFAULT_GRAPH))
            for name, description in graph_descriptions().items()
        ]
    )


@app.post("/run", response_model=RunResponse)
async def run_pipeline(req: RunRequest):
    """
    Run the full multi-agent pipeline **synchronously** (the HTTP call blocks
    until the pipeline finishes).

    Contained step failures return HTTP 200 with ``status: partial`` and the
    usable assembled output; only a fatal planner/graph/configuration error
    is an HTTP 500 (Slice 4).
    """
    _resolve_graph_or_400(req.graph)
    try:
        return await _run_pipeline(req.task, req.files, req.graph)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EmptyExtractedTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        error_id = uuid.uuid4().hex[:12]
        log_event("api_run_failed", error_id=error_id, error=str(exc),
                  traceback=traceback.format_exc())
        detail = (
            f"Pipeline failed: {exc}\n\n{traceback.format_exc()}"
            if DEBUG
            else f"Pipeline failed. See server logs (error id: {error_id})."
        )
        raise HTTPException(status_code=500, detail=detail)


@app.post("/run-async", response_model=AsyncRunResponse, status_code=202)
async def run_pipeline_async(req: RunRequest):
    """
    Start the pipeline **asynchronously** and return immediately with a task ID.

    Poll ``GET /status/{task_id}`` to check progress and retrieve the result.
    """
    _resolve_graph_or_400(req.graph)
    task_id = uuid.uuid4().hex[:12]

    async with _task_lock:
        _task_store[task_id] = {
            "status": "running",
            "final_output": None,
            "failed_steps": [],
            "skipped_steps": [],
            "step_stats": [],
            "error": None,
        }

    async def _background():
        try:
            # The task id also keys the run's artifact directory (plan 4.5),
            # so the client can fetch generated files from
            # GET /artifacts/{task_id}/{filename} with the id it already has.
            result = await _run_pipeline(req.task, req.files, req.graph, task_id)
            async with _task_lock:
                # Terminal "completed" or "partial" — both preserve the
                # assembled output and per-step statistics (Slice 4).
                _task_store[task_id] = {
                    "status": result.status,
                    "final_output": result.final_output,
                    "failed_steps": result.failed_steps,
                    "skipped_steps": result.skipped_steps,
                    "step_stats": [s.model_dump() for s in result.step_stats],
                    "error": None,
                }
        except Exception as exc:
            log_event("api_async_task_failed", task_id=task_id, error=str(exc),
                      traceback=traceback.format_exc())
            error = (
                f"{exc}\n{traceback.format_exc()}"
                if DEBUG
                else f"Pipeline failed. See server logs (task id: {task_id})."
            )
            async with _task_lock:
                _task_store[task_id] = {
                    "status": "failed",
                    "final_output": None,
                    "failed_steps": [],
                    "skipped_steps": [],
                    "step_stats": [],
                    "error": error,
                }

    asyncio.create_task(_background())
    return AsyncRunResponse(task_id=task_id, status="started")


@app.get("/status/{task_id}", response_model=StatusResponse)
async def task_status(task_id: str):
    """
    Retrieve the current status and, once terminal, the result of an async
    pipeline run. Terminal ``partial`` results keep their output and stats.
    """
    async with _task_lock:
        entry = _task_store.get(task_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    return StatusResponse(
        task_id=task_id,
        status=entry["status"],
        final_output=entry.get("final_output"),
        failed_steps=entry.get("failed_steps", []),
        skipped_steps=entry.get("skipped_steps", []),
        step_stats=entry.get("step_stats", []),
        error=entry.get("error"),
    )


@app.get("/artifacts/{task_id}/{filename}")
async def get_artifact(task_id: str, filename: str):
    """
    Serve a generated artifact file for a run (plan 4.5).

    File-producing tools write into ``<ARTIFACTS_DIR>/<task_id>/`` and return
    the relative path, so the client fetches it with the task id from
    ``/run``, ``/run-async`` or ``/status``. Both path segments are validated
    as single safe segments — traversal attempts get a 400.
    """
    try:
        path = get_artifact_path(filename, config={"configurable": {"task_id": task_id}})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{filename}' not found for task '{task_id}'.",
        )
    return FileResponse(path)



@app.post("/run-stream")
async def run_pipeline_stream(req: RunRequest):
    """Stream the pipeline using the marker protocol as Server-Sent Events."""
    async def event_source():
        try:
            async for token in stream_pipeline(req.task, req.files):
                token = token.replace('\n','\\n')
                yield f"data: {token}\n\n"   # SSE frame; client strips "data: "
        except Exception as exc:
            print(traceback.format_exc())
            yield f"data: [error] {exc}\n\n"
        finally:
            yield "data: <stop>\n\n"          # replaces the old None sentinel

    return StreamingResponse(event_source(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Direct runner (for ``python api_server.py`` without uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8999, reload=True)
