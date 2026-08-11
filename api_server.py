"""
FastAPI REST API server for the multi-agent LangGraph pipeline.

Endpoints:
    GET  /health          — health check
    GET  /graphs          — list the graphs discovered in graphs/
    POST /run             — run the pipeline synchronously (blocking)
    POST /run-async       — start a pipeline run in the background
    GET  /status/{task_id} — check status of an async run
"""

import asyncio
import os
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load env before importing pipeline modules (they read os.getenv at import time)
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver

from graphs import build_graph, graph_descriptions
from agents.agent_states import get_current_datetime_str
from utils.logger import log_event

# Graph used when a request does not name one.
DEFAULT_GRAPH = os.getenv("DEFAULT_GRAPH", "parallel").strip() or "parallel"

# When true, API error responses include the full traceback; otherwise clients
# get a generic message + id and the traceback stays in the server log.
DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=1,
        description="The natural-language task to run through the pipeline",
        examples=["Calculate sin(pi/4) + cos(pi/4) and plot both functions"],
    )
    graph: str | None = Field(
        default=None,
        description=f"Which graph in graphs/ to run (default: {DEFAULT_GRAPH}). "
                    f"See GET /graphs.",
        examples=["parallel", "sequential"],
    )


class RunResponse(BaseModel):
    final_output: str
    step_stats: list[dict] | None = None


class GraphInfo(BaseModel):
    name: str
    description: str = ""
    default: bool = False


class GraphsResponse(BaseModel):
    graphs: list[GraphInfo]


class AsyncRunResponse(BaseModel):
    task_id: str
    status: str = "started"


class StatusResponse(BaseModel):
    task_id: str
    status: str  # "running" | "completed" | "failed"
    final_output: str | None = None
    step_stats: list[dict] | None = None
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


async def _run_pipeline(task: str, graph_name: str | None = None) -> tuple[str, list[dict]]:
    """Run the selected LangGraph pipeline.

    Returns ``(final_output, step_stats)`` — the assembled output string and
    the per-step execution statistics collected during the run (Phase 4.9).
    """
    graph = _get_graph(graph_name)
    config = {"configurable": {"thread_id": f"api-{uuid.uuid4().hex[:8]}"}}
    result = await graph.ainvoke(
        {"task": task, "current_datetime": get_current_datetime_str()},
        config=config,
    )
    return (
        result.get("final_output", "No final output produced."),
        result.get("step_stats", []),
    )


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

    Suitable for most use-cases where the task completes within a few seconds
    to a couple of minutes.
    """
    _resolve_graph_or_400(req.graph)
    try:
        output, step_stats = await _run_pipeline(req.task, req.graph)
        return RunResponse(final_output=output, step_stats=step_stats)
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
        _task_store[task_id] = {"status": "running", "final_output": None, "error": None}

    async def _background():
        try:
            output, step_stats = await _run_pipeline(req.task, req.graph)
            async with _task_lock:
                _task_store[task_id] = {
                    "status": "completed",
                    "final_output": output,
                    "step_stats": step_stats,
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
                    "error": error,
                }

    asyncio.create_task(_background())
    return AsyncRunResponse(task_id=task_id, status="started")


@app.get("/status/{task_id}", response_model=StatusResponse)
async def task_status(task_id: str):
    """
    Retrieve the current status and result (if completed) of an async pipeline run.
    """
    async with _task_lock:
        entry = _task_store.get(task_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")

    return StatusResponse(
        task_id=task_id,
        status=entry["status"],
        final_output=entry.get("final_output"),
        step_stats=entry.get("step_stats"),
        error=entry.get("error"),
    )


# ---------------------------------------------------------------------------
# Direct runner (for ``python api_server.py`` without uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
