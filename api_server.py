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

from yotta_graph import builder as yotta_builder
from agent_states import get_current_datetime_str
from streaming import stream_pipeline
from yotta_tool import call_yotta, parse_yotta_results

# Compile without a checkpointer to prevent unbounded MemorySaver growth
# in the long-running server.  Each request is stateless and uses a unique
# thread_id — there's no need to persist checkpoints across calls.
api_graph = yotta_builder.compile()

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

# Server-side safety cap — client should truncate first, but this is a backstop.
_SERVER_MAX_FILE_CHARS = 100_000


def _truncate_file_content(filename: str, content: str, max_chars: int) -> str:
    """Truncate *content* to *max_chars*, keeping head + tail with a notice."""
    if len(content) <= max_chars:
        return content
    half = max_chars // 2
    return (
        content[:half]
        + f"\n\n... [TRUNCATED — {len(content) - max_chars:,} chars omitted] ...\n\n"
        + content[-half:]
    )


def _build_task_with_files(task: str, files: list[FileInput] | None) -> str:
    """Prepend attached file contents to the task string as context."""
    if not files:
        return task
    file_blocks: list[str] = []
    for f in files:
        content = _truncate_file_content(f.filename, f.content, _SERVER_MAX_FILE_CHARS)
        if f.encoding == "base64":
            file_blocks.append(
                f"### File: {f.filename} [binary — base64-encoded]\n"
                f"```\n{content}\n```"
            )
        else:
            file_blocks.append(f"### File: {f.filename}\n```\n{content}\n```")
    return "## Attached files (input context)\n\n" + "\n\n".join(file_blocks) + f"\n\n## Task\n{task}"


async def _run_pipeline(task: str, files: list[FileInput] | None = None) -> str:
    """Run the multi-agent LangGraph pipeline and return the assembled output."""
    task_with_files = _build_task_with_files(task, files)
    # Pre-search with yotta, same as the streaming path
    yotta_results = await call_yotta(task_with_files)
    clean_findings = parse_yotta_results(yotta_results)

    # Pass search results as a dedicated state field instead of embedding
    # them in the task string, so the writer node doesn't have to parse.
    config = {"configurable": {"thread_id": f"api-{uuid.uuid4().hex[:8]}"}}
    result = await api_graph.ainvoke(
        {
            "task": f"Query: {task_with_files}",
            "search_results": clean_findings,
            "current_datetime": get_current_datetime_str(),
        },
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
        output = await _run_pipeline(req.task, req.files)
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
            output = await _run_pipeline(req.task, req.files)
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
