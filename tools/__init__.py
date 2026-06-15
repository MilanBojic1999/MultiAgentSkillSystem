"""
Auto-discovery tool registry.

Scans all .py files in this directory (excluding this file and agent_tools.py),
imports them, and collects every @tool-decorated function into TOOL_REGISTRY.

To add a new tool, drop a .py file here that defines one or more @tool functions.
No manual registration needed — it's picked up automatically at import time.
"""

import importlib
import os
from pathlib import Path
from langchain_core.tools import BaseTool

_TOOLS_DIR = Path(__file__).parent

# Files excluded from auto-discovery (they're infrastructure, not tool definitions)
_EXCLUDE_FILES = {"__init__.py", "agent_tools.py"}

TOOL_REGISTRY: dict[str, BaseTool] = {}


def _discover_tools() -> dict[str, BaseTool]:
    """Scan the tools/ directory and collect every @tool-decorated callable."""
    registry: dict[str, BaseTool] = {}

    for fname in sorted(os.listdir(_TOOLS_DIR)):
        # Skip non-Python files, directories, and infrastructure modules
        if not fname.endswith(".py") or fname in _EXCLUDE_FILES:
            continue

        module_name = fname[:-3]  # strip ".py"

        try:
            mod = importlib.import_module(f"tools.{module_name}")
        except Exception as e:
            print(f"Warning: failed to import tools/{fname}: {e}")
            continue

        for name, obj in mod.__dict__.items():
            if isinstance(obj, BaseTool) and not name.startswith("_"):
                registry[name] = obj

    return registry


# Build the registry at import time
TOOL_REGISTRY = _discover_tools()

# Expose every discovered tool at the package level so "from tools import
# <tool_name>" continues to work without manual export lists.
globals().update(TOOL_REGISTRY)

__all__ = sorted(TOOL_REGISTRY.keys()) + ["TOOL_REGISTRY"]
