"""graphs/__init__.py — the auto-discovering graph registry (plan item 3.3).

The contract under test: a module in ``graphs/`` **is** a graph as soon as it
defines ``build()``. Nothing else registers it, so the tests that matter here
write a throwaway module into ``graphs/`` at runtime and assert it shows up.
"""

import importlib
import sys
from pathlib import Path

import pytest

from graphs import (
    GRAPH_REGISTRY,
    available_graphs,
    build_graph,
    clear_registry_cache,
    graph_descriptions,
)
from graphs import _derive_name

GRAPHS_DIR = Path(__file__).resolve().parent.parent / "graphs"

# Every generated module is prefixed with this so a crashed run leaves obvious
# droppings rather than something mistakable for a real graph.
PROBE_PREFIX = "zz_probe"


@pytest.fixture()
def graph_file():
    """Write a module into graphs/ for the duration of one test."""
    created: list[Path] = []

    def _write(module_name: str, source: str) -> str:
        assert module_name.startswith(PROBE_PREFIX)
        path = GRAPHS_DIR / f"{module_name}.py"
        path.write_text(source, encoding="utf-8")
        created.append(path)
        importlib.invalidate_caches()
        clear_registry_cache()
        return module_name

    yield _write

    for path in created:
        path.unlink(missing_ok=True)
        sys.modules.pop(f"graphs.{path.stem}", None)
    importlib.invalidate_caches()
    clear_registry_cache()


SENTINEL_GRAPH = '''
"""A probe graph."""

GRAPH_DESCRIPTION = "probe graph for the registry tests"


def build(*, checkpointer=None, orchestrator=None, sub_agent=None):
    return {"compiled": True, "checkpointer": checkpointer,
            "orchestrator": orchestrator, "sub_agent": sub_agent}
'''


# ---------------------------------------------------------------------------
# Shipped graphs
# ---------------------------------------------------------------------------

def test_shipped_graphs_are_discovered():
    assert {"parallel", "sequential"} <= set(available_graphs())


def test_descriptions_come_from_the_modules():
    described = graph_descriptions()
    assert described["parallel"]
    assert described["sequential"]


def test_build_graph_compiles_with_injected_nodes():
    def stub_orchestrator(state):
        return {"plan": [], "results": {}, "current_step": 0}

    async def stub_worker(state):
        return {}

    graph = build_graph("parallel", orchestrator=stub_orchestrator, sub_agent=stub_worker)
    assert graph is not None
    # No LLM env beyond conftest's dummies and no network were needed.
    assert "parallel_sub_agent" in graph.get_graph().nodes


def test_registry_mapping_exposes_build_callables():
    from graphs import parallel_pipeline_graph

    assert GRAPH_REGISTRY["parallel"] is parallel_pipeline_graph.build
    assert "parallel" in list(GRAPH_REGISTRY)
    with pytest.raises(KeyError):
        GRAPH_REGISTRY["definitely-not-a-graph"]


def test_unknown_graph_error_lists_what_is_available():
    with pytest.raises(ValueError, match=r"Unknown graph 'nope'.*parallel"):
        build_graph("nope")


# ---------------------------------------------------------------------------
# Auto-registration: a new file is all it takes
# ---------------------------------------------------------------------------

def test_new_module_registers_itself(graph_file):
    graph_file(f"{PROBE_PREFIX}_new_graph", SENTINEL_GRAPH)

    assert f"{PROBE_PREFIX}_new" in available_graphs()
    built = build_graph(f"{PROBE_PREFIX}_new", checkpointer="cp")
    assert built["checkpointer"] == "cp"
    assert graph_descriptions()[f"{PROBE_PREFIX}_new"] == "probe graph for the registry tests"


def test_removing_the_file_deregisters_the_graph(graph_file):
    name = f"{PROBE_PREFIX}_gone"
    path = GRAPHS_DIR / f"{graph_file(f'{name}_graph', SENTINEL_GRAPH)}.py"
    assert name in available_graphs()

    path.unlink()
    sys.modules.pop(f"graphs.{name}_graph", None)
    importlib.invalidate_caches()
    clear_registry_cache()
    assert name not in available_graphs()


def test_graph_name_overrides_the_filename(graph_file):
    graph_file(
        f"{PROBE_PREFIX}_renamed_graph",
        'GRAPH_NAME = "aliased-probe"\n' + SENTINEL_GRAPH,
    )

    assert "aliased-probe" in available_graphs()
    assert f"{PROBE_PREFIX}_renamed" not in available_graphs()
    assert build_graph("aliased-probe")["compiled"] is True


def test_module_without_build_is_not_a_graph(graph_file):
    graph_file(f"{PROBE_PREFIX}_helper_graph", "CONSTANT = 1\n")

    assert f"{PROBE_PREFIX}_helper" not in available_graphs()
    with pytest.raises(ValueError, match="defines no build"):
        build_graph(f"{PROBE_PREFIX}_helper")


def test_private_modules_are_ignored(graph_file):
    # Files named with a leading underscore are graph-package internals.
    path = GRAPHS_DIR / "_zz_probe_private_graph.py"
    path.write_text(SENTINEL_GRAPH, encoding="utf-8")
    try:
        clear_registry_cache()
        assert "_zz_probe_private" not in available_graphs()
    finally:
        path.unlink(missing_ok=True)
        clear_registry_cache()


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------

def test_broken_module_does_not_break_the_registry(graph_file):
    graph_file(f"{PROBE_PREFIX}_broken_graph", "raise RuntimeError('boom')\n")

    # Listing survives and still reports the healthy graphs...
    assert {"parallel", "sequential"} <= set(available_graphs())
    # ...while asking for the broken one names the file and the real cause.
    with pytest.raises(ImportError, match="boom"):
        build_graph(f"{PROBE_PREFIX}_broken")


def test_name_collision_is_reported(graph_file):
    graph_file(f"{PROBE_PREFIX}_clash_graph", SENTINEL_GRAPH)
    graph_file(f"{PROBE_PREFIX}_clash_pipeline_graph", SENTINEL_GRAPH)

    with pytest.raises(ValueError, match="claimed by several modules"):
        build_graph(f"{PROBE_PREFIX}_clash")


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("module_name", "expected"),
    [
        ("parallel_pipeline_graph", "parallel"),
        ("sequential_pipeline_graph", "sequential"),
        ("reflection_graph", "reflection"),
        ("reflection_pipeline", "reflection"),
        ("reflection", "reflection"),
        ("_graph", "_graph"),  # suffix-only name is left alone
    ],
)
def test_derive_name(module_name, expected):
    assert _derive_name(module_name) == expected
