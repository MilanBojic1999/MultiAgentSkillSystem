# streaming.py
import json
import uuid
from agent_states import get_current_datetime_str

# Stream the SEQUENTIAL graph — parallel interleaves tokens (see §4).
from yotta_graph import graph

# Nodes that represent an "agent turn"; each opens a new <thinking_step>.
_AGENT_NODES = {"orchestrator", "sub_agent", "verify", "assemble"}


def _reasoning_delta(chunk) -> str | None:
    """Vendor reasoning tokens live in additional_kwargs, NOT chunk.content.
    Returns the raw token delta string (may be a single token or a few tokens).
    Confirm the exact key against your endpoint (see §6.1)."""
    ak = getattr(chunk, "additional_kwargs", {}) or {}
    val = ak.get("reasoning_content") or ak.get("reasoning")
    if isinstance(val, dict):                 # some providers nest it
        val = val.get("text") or val.get("content")
    if isinstance(val, str) and val:
        return val
    return None


def _visible_delta(chunk) -> str:
    """Visible answer text — raw token delta from the chunk.
    chunk.content is usually a str but can be a list."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):             # content-block form
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return ""


async def stream_pipeline(task: str):
    """
    Async generator yielding the old marker protocol:
    <thinking_step>, <think>/<non_think>, token text, TOOL name(args), stop.

    Each call uses a unique thread_id so the MemorySaver checkpointer never
    resumes a previous run — every request starts fresh.
    """
    # Unique thread_id per invocation — prevents checkpoint collision across calls
    config = {"configurable": {"thread_id": f"stream-{uuid.uuid4().hex}"}}
    state_in = {
        "task": task,
        "current_datetime": get_current_datetime_str(),
        "streaming": True,                    # see §5.3 — must reach the LLM constructors
    }

    is_thinking = None                        # None = undecided for this step
    open_step = False

    async for event in graph.astream_events(state_in, config=config, version="v2"):
        kind = event["event"]
        agent = event.get("name", "unknown_agent")
        # --- New agent turn -> <thinking_step> ---------------------------
        if kind == "on_chain_start" and agent in _AGENT_NODES:
            if open_step:
                yield "\n\n"                  # close previous iteration (old l. 220)
            yield "<thinking_step>\n"
            open_step = True
            is_thinking = None                # reset toggle each step
            continue

        if kind == "on_tool_start":
            tool_name = event.get("name", "unknown_tool")
            tool_args = event.get("data", {})
            if tool_args:
                tool_args = tool_args.get("input", tool_args)  # some providers nest it
            yield "TOOL "
            yield f"{tool_name}"
            yield f"({json.dumps(tool_args)})\n"
            continue

        # --- Streaming LLM tokens ----------------------------------------
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            reasoning = _reasoning_delta(chunk)
            if reasoning:
                if is_thinking is not True:
                    is_thinking = True
                    yield "<think>\n"
                yield reasoning        # raw token delta — yield as-is
                continue
            visible = _visible_delta(chunk)
            if visible:
                if is_thinking is not False and agent == "assemble":
                    is_thinking = False
                    yield "<non_think>\n"
                yield visible          # raw token delta — yield as-is
            continue