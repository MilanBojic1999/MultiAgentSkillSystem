"""Artifact-path convention for file-producing tools (plan item 4.5).

Every tool that writes a file resolves its destination through
:func:`get_artifact_path` instead of inventing its own location:

    <ARTIFACTS_DIR (default ``artifacts``)>/<run id>/<unique name>

The run id is read from the run config (``configurable["task_id"]``, falling
back to ``configurable["thread_id"]``), so separate runs never share a
directory. Callers must still choose a *unique* filename per invocation
(e.g. a short uuid prefix) — two invocations inside one run share the
directory. Without a run config the artifact lands directly under the root,
which keeps direct tool use (REPL, tests, demos) working.

``name`` and the run id are validated as single path segments, so a path
built here can be handed to the API's ``GET /artifacts/{task_id}/{filename}``
endpoint without traversal risk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.runnables import RunnableConfig

# One path segment: no separators, no leading dot — blocks ``..`` traversal.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def artifacts_root() -> Path:
    """Root directory for generated artifacts; ``ARTIFACTS_DIR`` env-overridable."""
    return Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


def _segment(value: str, what: str) -> str:
    """Validate a single path segment, raising ``ValueError`` naming ``what``."""
    if not _SEGMENT_RE.match(value):
        raise ValueError(
            f"Invalid {what} {value!r}: must be a single path segment "
            "(letters, digits, '.', '_', '-'; no separators)."
        )
    return value


def get_artifact_path(name: str, config: RunnableConfig | None = None) -> Path:
    """Resolve where a generated artifact should be written.

    Keys the directory by the run id from ``config`` (task id, falling back to
    thread id) so separate runs never collide; without either, the artifact
    lands directly under the root. ``name`` must be a single path segment —
    tools pick a unique name per invocation themselves.

    Returns a path relative to the current working directory when
    ``ARTIFACTS_DIR`` is relative, so a tool can hand it straight back to the
    LLM as the artifact's location.
    """
    cfg = (config or {}).get("configurable", {})
    run_id = cfg.get("task_id") or cfg.get("thread_id") or None
    root = artifacts_root()
    if run_id is not None:
        root = root / _segment(str(run_id), "run id")
    return root / _segment(name, "artifact name")
