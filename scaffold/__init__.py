"""Scaffold — generate well-formed extension stubs (plan item 4.17).

Usage::

    python -m scaffold graph  my_graph     # -> graphs/my_graph_graph.py
    python -m scaffold tool   my_tool      # -> tools/my_tool.py
    python -m scaffold skill  my_skill     # -> skills/my-skill/SKILL.md
    python -m scaffold agent  my_agent     # appends to agents/agent_config.json

Every scaffold refuses to overwrite an existing file and prints the next
step after writing.
"""

from scaffold._core import (
    _DIR_NAME_RE,
    _IDENTIFIER_RE,
    scaffold_graph,
    scaffold_tool,
    scaffold_skill,
    scaffold_agent,
    main,
)

__all__ = [
    "scaffold_graph",
    "scaffold_tool",
    "scaffold_skill",
    "scaffold_agent",
    "main",
    "_IDENTIFIER_RE",
    "_DIR_NAME_RE",
]
