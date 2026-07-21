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
    """Import ``paralel_pipeline_graph`` or xfail the calling test.

    The module builds *and compiles* its graph at import time, and that wiring
    is currently broken (Bug 1 in TESTING_GUIDE.md: string path passed to
    ``add_conditional_edges``). The pure functions we want to test are defined
    before the failing lines, but a failed import removes the module from
    ``sys.modules`` entirely — so there is no way to reach them until Bug 1 is
    fixed. Calling this marks the test xfail imperatively; once the wiring is
    corrected the import succeeds and the real assertions run.
    """
    try:
        import paralel_pipeline_graph as pg
    except Exception as e:  # pragma: no cover - path exercised only while red
        pytest.xfail(f"Bug 1: paralel_pipeline_graph does not import: {e}")
    return pg
