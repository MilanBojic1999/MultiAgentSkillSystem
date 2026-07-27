"""Small test helpers shared across modules."""

import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


def fake_llm(payload) -> GenericFakeChatModel:
    """A chat model that replays a single scripted message.

    ``payload`` may be a dict/list (serialized to JSON, the orchestrator's
    expected format) or a raw string (to script a malformed response).
    """
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return GenericFakeChatModel(messages=iter([AIMessage(content=content)]))


def import_parallel_or_xfail():
    """Return the parallel-graph module, or xfail if it cannot be imported.

    Since item 1.1 the module imports and compiles cleanly, so this is normally
    a passthrough — the dedicated ``test_parallel_graph_module_builds`` is what
    guards import regressions loudly. The xfail branch remains only as a safety
    net for tests that assert on the pure router/scheduler functions.
    """
    try:
        from graphs import parallel_pipeline_graph as pg
    except Exception as e:  # pragma: no cover - safety net only
        pytest.xfail(f"graphs.parallel_pipeline_graph does not import: {e}")
    return pg
