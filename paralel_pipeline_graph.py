"""Import-compat shim (Phase 1.5 transition).

The parallel graph now lives in ``graphs/paralel_pipeline_graph.py``. This
root module re-exports it so existing importers — ``api_server``,
``run_pipeline``, and the tests — keep working while callers migrate to the
``graphs`` package (Phase 3.3). Remove once nothing imports the root path.
"""

from graphs.paralel_pipeline_graph import (  # noqa: F401
    graph,
    fan_out_router,
    scheduler_node,
    parallel_sub_agent_node,
)
