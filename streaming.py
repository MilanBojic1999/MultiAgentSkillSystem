# streaming.py
"""Marker-protocol streaming for the ``yotta`` graph.

``astream_events`` reports every LLM token of every node, including the
sub-agents the yotta graph fans out in parallel — raw, that is an unreadable
interleaving of several workers, the planner's JSON and the verifier's report.
This module translates that event firehose into a small, ordered frame
protocol so a client can render *what is happening* and read only the writer's
answer.

Frames (each one is a single ``yield``, so one SSE ``data:`` line carries one
complete frame; payloads are JSON, since a subtask may contain any character):

.. code-block:: text

    <plan>{"plan":[{"step":1,"agent":"research-worker","subtask":"…",…}]}</plan>
    <agent_start>{"step":1,"agent":"research-worker","subtask":"…"}</agent_start>
    <agent_end>{"step":1,"agent":"…","status":"completed","duration_s":12.4,…}</agent_end>
    <verification>…raw verifier report…</verification>
    <answer>                                ← writer node started
    …visible tokens, raw, one yield per chunk…
    <answer_restart>                        ← writer attempt restarted; drop what came before
    </answer>                               ← writer node ended
    <tool>{"name":"…","args":{…}}</tool>    ← the writer's tool calls only
    [error] <message>                       ← one frame, newlines escaped
    <stop>                                  ← always the last frame

Per node:

- ``orchestrator`` — the plan, whole, once, when the node ends. Its raw LLM
  text is not retained anywhere (``agents/orchestrator_node.py`` parses it and
  returns the validated plan), so the frame carries that validated plan.
- ``sub_agent`` — lifecycle only: one ``<agent_start>`` per dispatched step and
  one ``<agent_end>`` when it finishes (or fails). No tokens, no tool frames.
- ``verify`` — the verifier's raw report, whole, once, when the node ends.
- ``writer`` — visible answer tokens, streamed live. Reasoning tokens are
  suppressed unless ``STREAM_WRITER_REASONING`` is turned on.

Node attribution cannot use ``metadata["langgraph_node"]`` alone: every worker
and the writer run a nested ``create_react_agent`` Pregel whose own tasks
overwrite that key with the inner graph's node names. A node's *own* chain
event is the one whose runnable name equals it (``_node_of`` — the same check
langgraph makes in ``pregel/_messages.py``), and token gating rides on the
writer node's own start/end because the writer never shares a super-step with
another node (``fan_out_router`` and ``after_verify`` each return a single
target, and only ``sub_agent`` tasks run concurrently).
"""

import json
import uuid
from agents.agent_states import get_current_datetime_str
from execution_policy import normalize_effort, resolve_execution_policy
from pipeline_entry import build_task_string, build_files_state
from utils.logger import log_event

# ---------------------------------------------------------------------------
# Monkey-patch: ChatOpenAI._convert_delta_to_message_chunk drops
# `reasoning_content` from the delta (it only targets the official OpenAI
# spec — see langchain_openai/chat_models/base.py:5-11).  Recover it so
# _reasoning_delta() can find it in additional_kwargs.
# ---------------------------------------------------------------------------
import langchain_openai.chat_models.base as _lc_base

from yotta_tool import call_yotta, parse_yotta_results

_original_convert = _lc_base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(_dict, default_class):
    chunk = _original_convert(_dict, default_class)
    reasoning = _dict.get("reasoning_content")
    if reasoning and isinstance(reasoning, str) and hasattr(chunk, "additional_kwargs"):
        chunk.additional_kwargs["reasoning_content"] = reasoning
    return chunk


_lc_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk
# ---------------------------------------------------------------------------

# The yotta graph is built lazily on first stream — building needs an LLM
# configuration, and importing this module must not (repo convention:
# imports carry no import-time side effects). ``build_graph`` resolves the
# auto-registered "yotta" graph (graphs/yotta_graph.py).
from graphs import build_graph

_graph_cache: dict = {}


def _get_graph():
    if "yotta" not in _graph_cache:
        _graph_cache["yotta"] = build_graph("yotta")
    return _graph_cache["yotta"]


# Graph node names this protocol reacts to (graphs/yotta_graph.py's build()).
# ``citatitaion`` is deliberately absent: the node exists but is not wired.
ORCHESTRATOR_NODE = "orchestrator"
WORKER_NODE = "sub_agent"
VERIFY_NODE = "verify"
WRITER_NODE = "writer"

# The writer's reasoning tokens are suppressed by default (the reader wants the
# answer, not the deliberation). Flip this to stream them inside <think> frames
# — the monkeypatch above is what makes them visible at all.
STREAM_WRITER_REASONING = False


def _reasoning_delta(chunk) -> str | None:
    """Vendor reasoning tokens live in additional_kwargs, NOT chunk.content.
    Returns the raw token delta string (may be a single token or a few tokens).
    Confirm the exact key against your endpoint."""
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


def _node_of(event) -> str | None:
    """The graph node an event belongs to — but only for the node's **own**
    chain event.

    Nested runnables inherit ``langgraph_node`` from the task config, and the
    inner ReAct graph overwrites it with its own node names, so the metadata
    key alone attributes nothing. The runnable-name match is what makes this
    exact (langgraph's own message streaming makes the same check).
    Returns ``None`` for every nested/framework event.
    """
    metadata = event.get("metadata") or {}
    node = metadata.get("langgraph_node")
    if node and event.get("name") == node:
        return node
    return None


def _frame(tag: str, body: str) -> str:
    return f"<{tag}>{body}</{tag}>"


def _json_frame(tag: str, payload) -> str:
    return _frame(tag, json.dumps(payload, ensure_ascii=False, default=str))


class _ProtocolTranslator:
    """One ``astream_events`` event in, zero or more protocol frames out.

    Pure and synchronous — no I/O, no graph, no LLM — so the protocol is unit
    testable against hand-written event dicts (tests/test_streaming_protocol.py).
    All state is the writer gate: whether the writer node is running, and
    whether it has already emitted answer text.
    """

    def __init__(self):
        self.writer_active = False
        self.answer_chars = 0
        self.in_think = False

    # -- public -------------------------------------------------------------

    def handle(self, event) -> list[str]:
        kind = event.get("event")
        if kind == "on_chain_start":
            return self._node_start(_node_of(event), event)
        if kind == "on_chain_end":
            return self._node_end(_node_of(event), event)
        if kind == "on_chat_model_start":
            return self._model_start()
        if kind == "on_chat_model_stream":
            return self._model_stream(event)
        if kind == "on_tool_start":
            return self._tool_start(event)
        return []

    def finish(self) -> list[str]:
        """Close an ``<answer>`` the writer never closed (error, cancellation)."""
        if self.writer_active:
            return self._close_answer()
        return []

    # -- node lifecycle -----------------------------------------------------

    def _node_start(self, node, event) -> list[str]:
        if node == WORKER_NODE:
            step = self._input(event).get("step")
            if not isinstance(step, dict):
                step = {}
            return [_json_frame("agent_start", {
                "step": step.get("step"),
                "agent": step.get("agent", ""),
                "subtask": step.get("subtask", ""),
                "depends_on": step.get("depends_on", []),
            })]
        if node == WRITER_NODE:
            self.writer_active = True
            self.answer_chars = 0
            self.in_think = False
            return ["<answer>"]
        return []

    def _node_end(self, node, event) -> list[str]:
        if node == ORCHESTRATOR_NODE:
            plan = self._output(event).get("plan")
            if not isinstance(plan, list):
                plan = []
            return [_json_frame("plan", {"plan": plan})]
        if node == WORKER_NODE:
            stats = self._output(event).get("step_stats")
            entry = stats[0] if isinstance(stats, list) and stats else {}
            if not isinstance(entry, dict):
                entry = {}
            return [_json_frame("agent_end", entry)]
        if node == VERIFY_NODE:
            output = self._output(event)
            report = output.get("verifier_report") or output.get("verification_notes") or ""
            return [_frame("verification", str(report))]
        if node == WRITER_NODE:
            if not self.writer_active:
                # No start event was seen (never expected) — still deliver the
                # document rather than swallowing the whole answer.
                final = self._output(event).get("final_output") or ""
                return ["<answer>", str(final), "</answer>"] if final else []
            frames: list[str] = []
            if self.answer_chars == 0:
                # The writer produced no streamed token — a non-streaming
                # client, or the budget-exhaustion finalize pass. Emit the
                # assembled document itself so the reader is never left with
                # an empty answer.
                final = self._output(event).get("final_output") or ""
                if final:
                    frames.append(str(final))
            frames.extend(self._close_answer())
            return frames
        return []

    # -- token / tool stream ------------------------------------------------

    def _model_start(self) -> list[str]:
        # A second LLM call while the writer is streaming means the bounded
        # attempt loop re-invoked it: everything sent so far is a dead draft.
        if self.writer_active and self.answer_chars:
            frames = self._close_think()
            frames.append("<answer_restart>")
            self.answer_chars = 0
            return frames
        return []

    def _model_stream(self, event) -> list[str]:
        if not self.writer_active:
            return []
        chunk = (event.get("data") or {}).get("chunk")
        if chunk is None:
            return []
        if STREAM_WRITER_REASONING:
            reasoning = _reasoning_delta(chunk)
            if reasoning:
                if self.in_think:
                    return [reasoning]
                self.in_think = True
                return ["<think>", reasoning]
        visible = _visible_delta(chunk)
        if not visible:
            return []
        frames = self._close_think()
        self.answer_chars += len(visible)
        frames.append(visible)
        return frames

    def _tool_start(self, event) -> list[str]:
        # Sub-agent tool calls are deliberately invisible: a step reports only
        # that it started and that it finished.
        if not self.writer_active:
            return []
        data = event.get("data") or {}
        args = data.get("input", data)
        return [_json_frame("tool", {
            "name": event.get("name", "unknown_tool"),
            "args": args,
        })]

    # -- helpers ------------------------------------------------------------

    def _close_think(self) -> list[str]:
        if self.in_think:
            self.in_think = False
            return ["</think>"]
        return []

    def _close_answer(self) -> list[str]:
        frames = self._close_think()
        frames.append("</answer>")
        self.writer_active = False
        self.answer_chars = 0
        return frames

    @staticmethod
    def _input(event) -> dict:
        data = (event.get("data") or {}).get("input")
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _output(event) -> dict:
        data = (event.get("data") or {}).get("output")
        return data if isinstance(data, dict) else {}


async def stream_pipeline(task: str, files: list | None = None, effort: str | None = None):
    """
    Async generator yielding the marker protocol documented at module level:
    ``<plan>``, ``<agent_start>``/``<agent_end>``, ``<verification>``,
    ``<answer>`` + the writer's visible tokens, ``<tool>``, ``<stop>``.

    Every run ends with ``<stop>``: on success after any open ``<answer>`` is
    closed, and on failure after a single-line ``[error] <message>`` frame. The
    per-stream checkpointer thread is deleted when the generator finishes,
    whatever the outcome — including a client disconnect (the SSE endpoint
    acalls ``aclose``).

    Each call uses a unique thread_id so the MemorySaver checkpointer never
    resumes a previous run — every request starts fresh.

    If ``files`` is provided, their text content is carried in ``state["files"]``
    (filename -> content) rather than embedded in the task string — the planner
    routes filenames to the steps that need them (same as the non-streaming
    path — see ``pipeline_entry.build_task_string`` / ``build_files_state``).

    ``effort`` resolves through the shared policy module exactly like the
    CLI (``run_pipeline.py``) and API (``/run``, ``/run-async``) boundaries:
    the normalized preset and its resolved ``execution_policy`` travel under
    ``config["configurable"]``, so the graph's entry router enforces the same
    budgets (plan steps, worker attempts, tool calls, verification, replans,
    wall-clock deadline) on streamed runs. Omitted effort resolves to
    ``unlimited`` (legacy behavior).
    """
    thread_id = f"stream-{uuid.uuid4().hex}"
    try:
        preset = normalize_effort(effort)
        policy = resolve_execution_policy(preset)
        log_event("execution_policy_resolved", effort=preset,
                  execution_policy=policy.as_dict())

        # Decode/extract file text first — fail fast on a bad upload before
        # spending a search call on it.
        files_state = build_files_state(files)

        # Search on the bare task — file contents never reach the search call.
        yotta_results = await call_yotta(task)
        clean_findings = parse_yotta_results(yotta_results)

        task_string = build_task_string(task, files)

        # Unique thread_id per invocation — prevents checkpoint collision across calls
        config = {"configurable": {"thread_id": thread_id,
                                   "effort": preset,
                                   "execution_policy": policy.as_dict()},
                  "recursion_limit": 64}
        state_in = {
            "task": task_string,
            "search_results": clean_findings,
            "current_datetime": get_current_datetime_str(),
            "streaming": True,                # must reach the LLM constructors
            "files": files_state,
        }

        translator = _ProtocolTranslator()
        async for event in _get_graph().astream_events(state_in, config=config, version="v2"):
            for frame in translator.handle(event):
                yield frame

        # Normal completion: close anything still open, then terminate.
        for frame in translator.finish():
            yield frame
        yield "<stop>"
    except Exception as exc:
        # One frame per error; newlines are escaped so the SSE transport (and
        # any line-based parser) keeps the message inside a single frame.
        log_event("stream_error", error=str(exc))
        yield "[error] " + str(exc).replace("\n", "\\n")
        yield "<stop>"
    finally:
        # Drop this run's checkpoints — MemorySaver never evicts, so a
        # finished/aborted stream would otherwise leak its super-steps forever.
        graph = _graph_cache.get("yotta")
        checkpointer = getattr(graph, "checkpointer", None) if graph is not None else None
        if checkpointer is not None:
            try:
                await checkpointer.adelete_thread(thread_id)
            except Exception as exc:
                log_event("stream_cleanup_failed", error=str(exc))
