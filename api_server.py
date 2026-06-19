"""
FastAPI REST API server for the multi-agent LangGraph pipeline.

Endpoints:
    GET  /health          — health check
    POST /run             — run the pipeline synchronously (blocking)
    POST /run-async       — start a pipeline run in the background
    GET  /status/{task_id} — check status of an async run
"""

import asyncio
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load env before importing pipeline modules (they read os.getenv at import time)
load_dotenv()

from yotta_graph import graph
from agent_states import get_current_datetime_str
from streaming import stream_pipeline

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


class RunResponse(BaseModel):
    final_output: str


class AsyncRunResponse(BaseModel):
    task_id: str
    status: str = "started"


class StatusResponse(BaseModel):
    task_id: str
    status: str  # "running" | "completed" | "failed"
    final_output: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# In-memory async-task store
# ---------------------------------------------------------------------------

_task_store: dict[str, dict[str, Any]] = {}
_task_lock: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    # Nothing special to initialise — the graph is already compiled at import
    yield
    # Clean up any lingering tasks
    async with _task_lock:
        _task_store.clear()


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

async def _run_pipeline(task: str) -> str:
    """Run the parallel LangGraph pipeline and return the assembled output."""
    config = {"configurable": {"thread_id": f"api-{uuid.uuid4().hex[:8]}"}}
    result = await graph.ainvoke(
        {"task": task, "current_datetime": get_current_datetime_str()},
        config=config,
    )
    return result.get("final_output", "No final output produced.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Return ok if the service is alive."""
    return HealthResponse(status="ok")


@app.post("/run", response_model=RunResponse)
async def run_pipeline(req: RunRequest):
    """
    Run the full multi-agent pipeline **synchronously** (the HTTP call blocks
    until the pipeline finishes).

    Suitable for most use-cases where the task completes within a few seconds
    to a couple of minutes.
    """
    try:
        output = await _run_pipeline(req.task)
        return RunResponse(final_output=output)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {exc}\n\n{traceback.format_exc()}",
        )


@app.post("/run-async", response_model=AsyncRunResponse, status_code=202)
async def run_pipeline_async(req: RunRequest):
    """
    Start the pipeline **asynchronously** and return immediately with a task ID.

    Poll ``GET /status/{task_id}`` to check progress and retrieve the result.
    """
    task_id = uuid.uuid4().hex[:12]

    async with _task_lock:
        _task_store[task_id] = {"status": "running", "final_output": None, "error": None}

    async def _background():
        try:
            output = await _run_pipeline(req.task)
            async with _task_lock:
                _task_store[task_id] = {"status": "completed", "final_output": output, "error": None}
        except Exception as exc:
            async with _task_lock:
                _task_store[task_id] = {
                    "status": "failed",
                    "final_output": None,
                    "error": f"{exc}\n{traceback.format_exc()}",
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
        error=entry.get("error"),
    )


@app.post("/run-stream")
async def run_pipeline_stream(req: RunRequest):
    """Stream the pipeline using the marker protocol as Server-Sent Events."""
    async def event_source():
        try:
            async for token in stream_pipeline(req.task):
                yield f"data: {token}\n\n"   # SSE frame; client strips "data: "
        except Exception as exc:
            yield f"data: [error] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"          # replaces the old None sentinel

    return StreamingResponse(event_source(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Direct runner (for ``python api_server.py`` without uvicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
