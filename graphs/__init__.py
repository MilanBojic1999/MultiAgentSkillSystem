"""Graph registry — graph modules register themselves by existing.

A module in this package **is** a graph as soon as it defines a module-level
``build(...)`` callable. There is no registry line to edit, no decorator to
apply and no config entry: dropping ``graphs/reflection_graph.py`` with a
``build()`` in it makes ``build_graph("reflection")`` work immediately.

Naming
------
The registry name is the module name with a trailing ``_pipeline_graph`` /
``_graph`` / ``_pipeline`` stripped::

    graphs/parallel_pipeline_graph.py   -> "parallel"
    graphs/sequential_pipeline_graph.py -> "sequential"
    graphs/reflection_graph.py          -> "reflection"

A module may override that with ``GRAPH_NAME = "..."`` and document itself for
``--list-graphs`` / ``GET /graphs`` with ``GRAPH_DESCRIPTION = "..."`` (the
module docstring's first line is used when no description is declared).
Modules whose name starts with ``_`` are ignored.

Laziness
--------
Discovery scans filenames; modules are imported only when their graph is
actually requested. So ``build_graph("parallel")`` imports exactly one graph
module, and a graph file that fails to import breaks only the callers asking
for *that* graph. Listing (``available_graphs`` / ``graph_descriptions``) does
have to import every candidate — it tolerates and reports broken modules
instead of propagating their errors.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterator, Mapping
from types import ModuleType
from typing import Any, Callable

from utils.logger import log_event

__all__ = [
    "GRAPH_REGISTRY",
    "available_graphs",
    "build_graph",
    "clear_registry_cache",
    "graph_descriptions",
]

# Longest first: "x_pipeline_graph" must not be stripped down to "x_pipeline".
_NAME_SUFFIXES = ("_pipeline_graph", "_graph", "_pipeline")

# Populated by the full scan (see _scan_all); reset by clear_registry_cache().
_scanned: dict[str, ModuleType] | None = None
_broken: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _derive_name(module_name: str) -> str:
    """``"parallel_pipeline_graph"`` -> ``"parallel"``."""
    for suffix in _NAME_SUFFIXES:
        if module_name.endswith(suffix) and len(module_name) > len(suffix):
            return module_name[: -len(suffix)]
    return module_name


def _candidates() -> dict[str, list[str]]:
    """Map derived name -> module names, from filenames alone (no imports)."""
    found: dict[str, list[str]] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        found.setdefault(_derive_name(info.name), []).append(info.name)
    return {name: sorted(mods) for name, mods in found.items()}


def _import(module_name: str) -> ModuleType:
    return importlib.import_module(f"{__name__}.{module_name}")


def _graph_name(module: ModuleType, module_name: str) -> str:
    declared = getattr(module, "GRAPH_NAME", None)
    return declared if isinstance(declared, str) and declared else _derive_name(module_name)


def _describe(module: ModuleType) -> str:
    declared = getattr(module, "GRAPH_DESCRIPTION", None)
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    doc = (module.__doc__ or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def _is_graph(module: ModuleType) -> bool:
    return callable(getattr(module, "build", None))


def _scan_all() -> dict[str, ModuleType]:
    """Import every candidate module and map its true graph name -> module.

    Broken modules are recorded in ``_broken`` and skipped, so one unimportable
    graph file cannot take the whole registry down with it.
    """
    global _scanned
    if _scanned is not None:
        return _scanned

    graphs: dict[str, ModuleType] = {}
    _broken.clear()
    for module_names in _candidates().values():
        for module_name in module_names:
            try:
                module = _import(module_name)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                _broken[module_name] = f"{type(exc).__name__}: {exc}"
                log_event("graph_module_import_failed", module=module_name, error=str(exc))
                continue
            if not _is_graph(module):
                continue
            name = _graph_name(module, module_name)
            if name in graphs and graphs[name] is not module:
                log_event(
                    "graph_name_collision",
                    name=name,
                    kept=graphs[name].__name__,
                    ignored=module.__name__,
                )
                continue
            graphs[name] = module

    _scanned = graphs
    return graphs


def clear_registry_cache() -> None:
    """Forget the full-scan cache — call after adding/removing a graph module."""
    global _scanned
    _scanned = None
    _broken.clear()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _resolve(name: str) -> ModuleType:
    """Return the graph module registered under ``name``.

    Fast path: the name matches a filename, so exactly one module is imported —
    and a module that is there but unusable gets a diagnosis naming the file
    rather than a generic "unknown graph".
    Slow path (a ``GRAPH_NAME`` override): fall back to the full scan.
    """
    if _scanned is not None and name in _scanned:
        return _scanned[name]

    candidates = _candidates().get(name, [])
    if len(candidates) > 1:
        raise ValueError(
            f"Graph name '{name}' is claimed by several modules in graphs/: "
            f"{candidates}. Rename one file or set GRAPH_NAME in it."
        )
    if candidates:
        module_name = candidates[0]
        try:
            module = _import(module_name)
        except Exception as exc:
            raise ImportError(
                f"Graph '{name}' lives in graphs/{module_name}.py but that "
                f"module failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        if not _is_graph(module):
            raise ValueError(
                f"graphs/{module_name}.py defines no build(...) function, so "
                f"it is not a graph. Add `def build(*, checkpointer=None, "
                f"orchestrator=None, sub_agent=None): ...` to register it."
            )
        if _graph_name(module, module_name) == name:
            return module
        # The module renamed itself via GRAPH_NAME — let the full scan decide.

    graphs = _scan_all()
    if name in graphs:
        return graphs[name]

    detail = ""
    if _broken:
        detail = " (modules that failed to import: " + ", ".join(
            f"{mod} — {err}" for mod, err in sorted(_broken.items())
        ) + ")"
    raise ValueError(
        f"Unknown graph '{name}'. Available: {sorted(graphs)}{detail}"
    )


class _GraphRegistry(Mapping):
    """Read-only mapping of graph name -> that graph's ``build`` callable.

    Kept for introspection and backwards compatibility with the hardcoded
    ``GRAPH_REGISTRY`` dict it replaces; ``build_graph`` is the normal entry
    point. Iteration triggers the full scan, item access does not.
    """

    def __getitem__(self, name: str) -> Callable[..., Any]:
        try:
            return _resolve(name).build
        except ValueError as exc:
            raise KeyError(str(exc)) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(_scan_all()))

    def __len__(self) -> int:
        return len(_scan_all())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<GraphRegistry {sorted(_scan_all())}>"


GRAPH_REGISTRY = _GraphRegistry()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def available_graphs() -> list[str]:
    """Names of every importable graph in ``graphs/``, sorted."""
    return sorted(_scan_all())


def graph_descriptions() -> dict[str, str]:
    """Graph name -> one-line description, for CLI/API listings."""
    return {name: _describe(module) for name, module in sorted(_scan_all().items())}


def build_graph(name: str, *, checkpointer=None, **overrides):
    """Build and compile the graph registered under ``name``.

    ``checkpointer`` and any node overrides (``orchestrator=...``,
    ``sub_agent=...``) are forwarded to the graph module's ``build()``.
    Raises ``ValueError`` naming the available graphs if ``name`` is unknown.
    """
    module = _resolve(name)
    log_event("build_graph", graph=name, module=module.__name__)
    return module.build(checkpointer=checkpointer, **overrides)
