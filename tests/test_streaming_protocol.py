"""streaming._ProtocolTranslator — the /run-stream marker protocol.

Hermetic: the translator is a pure event->frames mapper, so every case here is
a hand-written ``astream_events`` dict. No graph, no LLM, no network.

What the protocol must guarantee (see streaming.py's module docstring):
the orchestrator's plan and the verifier's report arrive whole, sub-agents
report only that they started and finished, and the only tokens on the wire
are the writer's visible answer text.
"""

import json

from streaming import _ProtocolTranslator, _node_of


# ---------------------------------------------------------------------------
# Event builders — shaped like langgraph's astream_events(version="v2") output
# ---------------------------------------------------------------------------

def node_event(kind: str, node: str, data: dict) -> dict:
    """A node's OWN chain event: runnable name == metadata langgraph_node."""
    return {"event": kind, "name": node, "data": data,
            "metadata": {"langgraph_node": node, "langgraph_step": 1}}


def nested_event(kind: str, name: str, node: str, data: dict) -> dict:
    """An event from a runnable nested inside a node (the ReAct agent, a tool,
    the chat model) — it inherits/overwrites ``langgraph_node`` but its
    runnable name never matches it."""
    return {"event": kind, "name": name, "data": data,
            "metadata": {"langgraph_node": node}}


class Chunk:
    """Minimal stand-in for an AIMessageChunk."""

    def __init__(self, content="", reasoning=None):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}


def token(text: str, node: str = "model") -> dict:
    return nested_event("on_chat_model_stream", "ChatOpenAI", node,
                        {"chunk": Chunk(text)})


def frames(translator, events) -> list[str]:
    out: list[str] = []
    for event in events:
        out.extend(translator.handle(event))
    return out


def body_of(frame: str, tag: str) -> str:
    assert frame.startswith(f"<{tag}>") and frame.endswith(f"</{tag}>"), frame
    return frame[len(tag) + 2: -(len(tag) + 3)]


PLAN = [
    {"step": 1, "agent": "research-worker", "subtask": "Find X",
     "skills_needed": ["web-researcher"], "depends_on": []},
    {"step": 2, "agent": "document-reader-worker", "subtask": "Read Y",
     "skills_needed": ["document-reader-worker"], "depends_on": [1]},
]


# ---------------------------------------------------------------------------
# Node attribution
# ---------------------------------------------------------------------------

def test_node_of_matches_only_a_nodes_own_event():
    assert _node_of(node_event("on_chain_start", "writer", {})) == "writer"


def test_node_of_ignores_nested_runnables():
    # The inner ReAct graph overwrites langgraph_node with its own node names;
    # only the name match keeps attribution exact.
    assert _node_of(nested_event("on_chain_start", "RunnableSequence", "writer", {})) is None
    assert _node_of(token("hi")) is None
    assert _node_of({"event": "on_chain_start", "name": "LangGraph", "data": {}}) is None


# ---------------------------------------------------------------------------
# Orchestrator — the plan, whole, once
# ---------------------------------------------------------------------------

def test_orchestrator_end_emits_the_plan_as_one_frame():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_start", "orchestrator", {"input": {"task": "t"}}),
        node_event("on_chain_end", "orchestrator",
                   {"output": {"plan": PLAN, "results": {}, "current_step": 0}}),
    ])
    assert len(out) == 1
    payload = json.loads(body_of(out[0], "plan"))
    assert [s["step"] for s in payload["plan"]] == [1, 2]
    assert [s["agent"] for s in payload["plan"]] == ["research-worker",
                                                     "document-reader-worker"]
    assert payload["plan"][1]["subtask"] == "Read Y"


def test_orchestrator_empty_plan_still_emits_a_frame():
    """The direct route: the search results sufficed, no sub-agents will run."""
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_end", "orchestrator", {"output": {"plan": []}}),
    ])
    assert out == ['<plan>{"plan": []}</plan>']


def test_orchestrator_tokens_are_not_streamed():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_start", "orchestrator", {"input": {}}),
        token('{"plan": ['),
        token('{"step": 1'),
    ])
    assert out == []


# ---------------------------------------------------------------------------
# Sub-agents — lifecycle only, even when they overlap
# ---------------------------------------------------------------------------

def _worker_start(step):
    return node_event("on_chain_start", "sub_agent",
                      {"input": {"step": step, "results": {}, "files": {}}})


def _worker_end(step_num, agent, status="completed", duration=1.5):
    return node_event("on_chain_end", "sub_agent", {"output": {
        "results": {step_num: "output text"},
        "step_stats": [{"step": step_num, "agent": agent, "status": status,
                        "duration_s": duration, "input_tokens": 10,
                        "output_tokens": 20, "tool_calls": 3}],
        "pending_retries": [],
    }})


def test_parallel_workers_report_start_and_finish_only():
    out = frames(_ProtocolTranslator(), [
        _worker_start(PLAN[0]),
        _worker_start(PLAN[1]),
        token("worker one thinking"),
        token("worker two thinking"),
        _worker_end(2, "document-reader-worker"),
        _worker_end(1, "research-worker"),
    ])
    assert len(out) == 4
    first = json.loads(body_of(out[0], "agent_start"))
    assert first == {"step": 1, "agent": "research-worker",
                     "subtask": "Find X", "depends_on": []}
    assert json.loads(body_of(out[1], "agent_start"))["step"] == 2
    assert json.loads(body_of(out[2], "agent_end"))["step"] == 2
    last = json.loads(body_of(out[3], "agent_end"))
    assert last["step"] == 1
    assert last["status"] == "completed"
    assert last["duration_s"] == 1.5


def test_failed_step_reports_its_status():
    out = frames(_ProtocolTranslator(), [
        _worker_end(1, "research-worker", status="failed", duration=0.2),
    ])
    assert json.loads(body_of(out[0], "agent_end"))["status"] == "failed"


def test_sub_agent_tool_calls_are_suppressed():
    out = frames(_ProtocolTranslator(), [
        _worker_start(PLAN[0]),
        nested_event("on_tool_start", "web_search", "tools",
                     {"input": {"query": "x"}}),
    ])
    assert len(out) == 1 and out[0].startswith("<agent_start>")


# ---------------------------------------------------------------------------
# Verifier — the raw report, whole, once
# ---------------------------------------------------------------------------

def test_verify_end_emits_the_raw_report():
    report = '[{"step": 1, "verification_result": "PASSED", "notes": "ok"}]'
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_end", "verify", {"output": {
            "verifier_report": report,
            "verification_result": "PASSED",
            "verification_notes": "### Step 1  [PASSED]",
            "verification_route": "proceed",
        }}),
    ])
    assert out == [f"<verification>{report}</verification>"]


def test_verify_falls_back_to_notes_without_a_report():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_end", "verify",
                   {"output": {"verification_notes": "### Step 1  [PASSED]"}}),
    ])
    assert body_of(out[0], "verification") == "### Step 1  [PASSED]"


# ---------------------------------------------------------------------------
# Writer — the only node whose tokens reach the client
# ---------------------------------------------------------------------------

def test_writer_tokens_stream_between_answer_markers():
    translator = _ProtocolTranslator()
    out = frames(translator, [
        node_event("on_chain_start", "writer", {"input": {}}),
        token("# Title"),
        token("\n\nbody"),
        node_event("on_chain_end", "writer", {"output": {"final_output": "# Title\n\nbody"}}),
    ])
    assert out == ["<answer>", "# Title", "\n\nbody", "</answer>"]
    assert translator.writer_active is False


def test_writer_reasoning_is_suppressed_by_default():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_start", "writer", {"input": {}}),
        nested_event("on_chat_model_stream", "ChatOpenAI", "model",
                     {"chunk": Chunk("", reasoning="deliberating...")}),
        token("answer"),
    ])
    assert out == ["<answer>", "answer"]


def test_writer_without_streamed_tokens_emits_the_final_output():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_start", "writer", {"input": {}}),
        node_event("on_chain_end", "writer",
                   {"output": {"final_output": "assembled document"}}),
    ])
    assert out == ["<answer>", "assembled document", "</answer>"]


def test_writer_retry_marks_the_partial_answer_stale():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_start", "writer", {"input": {}}),
        nested_event("on_chat_model_start", "ChatOpenAI", "model", {}),
        token("first attempt"),
        nested_event("on_chat_model_start", "ChatOpenAI", "model", {}),
        token("second attempt"),
        node_event("on_chain_end", "writer", {"output": {"final_output": "second attempt"}}),
    ])
    assert out == ["<answer>", "first attempt", "<answer_restart>",
                   "second attempt", "</answer>"]


def test_writer_tool_calls_are_streamed():
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_start", "writer", {"input": {}}),
        nested_event("on_tool_start", "write_file", "tools",
                     {"input": {"path": "out.md"}}),
    ])
    assert json.loads(body_of(out[1], "tool")) == {
        "name": "write_file", "args": {"path": "out.md"}}


def test_finish_closes_an_open_answer():
    translator = _ProtocolTranslator()
    frames(translator, [node_event("on_chain_start", "writer", {"input": {}}),
                        token("partial")])
    assert translator.finish() == ["</answer>"]
    assert translator.finish() == []


# ---------------------------------------------------------------------------
# Robustness — a malformed event must never break a run
# ---------------------------------------------------------------------------

def test_malformed_events_are_ignored():
    translator = _ProtocolTranslator()
    assert translator.handle({"event": "on_chain_end", "name": "sub_agent",
                              "metadata": {"langgraph_node": "sub_agent"},
                              "data": {}}) == ['<agent_end>{}</agent_end>']
    assert translator.handle({"event": "on_chain_start", "name": "sub_agent",
                              "metadata": {"langgraph_node": "sub_agent"},
                              "data": {"input": None}})[0].startswith("<agent_start>")
    assert translator.handle({"event": "on_custom_event", "name": "x"}) == []


def test_full_run_frame_order():
    """One end-to-end shape: plan -> two steps -> verification -> answer."""
    out = frames(_ProtocolTranslator(), [
        node_event("on_chain_end", "orchestrator", {"output": {"plan": PLAN}}),
        _worker_start(PLAN[0]),
        token("noise from a worker"),
        _worker_end(1, "research-worker"),
        _worker_start(PLAN[1]),
        _worker_end(2, "document-reader-worker"),
        node_event("on_chain_end", "verify", {"output": {"verifier_report": "[]"}}),
        node_event("on_chain_start", "writer", {"input": {}}),
        token("final answer"),
        node_event("on_chain_end", "writer", {"output": {"final_output": "final answer"}}),
    ])
    tags = [f.split(">")[0].lstrip("<") for f in out]
    assert tags == ["plan", "agent_start", "agent_end", "agent_start",
                    "agent_end", "verification", "answer", "final answer",
                    "/answer"]
