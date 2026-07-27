"""Import-compat shim (Phase 1.5 / 3.3 transition).

The parallel graph now lives in ``graphs/parallel_pipeline_graph.py`` (note the
fixed spelling) and is reached through the registry::

    from graphs import build_graph
    graph = build_graph("parallel")

This root module re-exports the graph's public names under the old misspelled
path so external importers keep working for one release. The pre-compiled
module-level ``graph`` singleton is gone: graphs are compiled by ``build()``,
which lets callers pass their own checkpointer.
"""

from graphs.parallel_pipeline_graph import (  # noqa: F401
    build,
    fan_out_router,
    scheduler_node,
)
