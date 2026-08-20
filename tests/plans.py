"""Canonical plan fixtures — one module so the shape is defined once.

Agent and skill names match ``agents/agent_config.json`` and ``skills/`` so the
same fixtures work in tests that touch the real config.
"""


def step(n, agent="researcher", skills=(), deps=(), files=()):
    return {
        "step": n,
        "subtask": f"subtask {n}",
        "agent": agent,
        "skills_needed": list(skills),
        "depends_on": list(deps),
        "files": list(files),
    }


LINEAR_PLAN = [step(1), step(2, deps=[1]), step(3, deps=[2])]
DIAMOND_PLAN = [step(1), step(2), step(3, deps=[1, 2])]        # the 1.1 scenario
WIDE_PLAN = [step(1), step(2), step(3), step(4), step(5, deps=[1, 2, 3, 4])]
CYCLIC_PLAN = [step(1, deps=[2]), step(2, deps=[1])]

KNOWN_AGENTS = {"researcher", "mathematician", "writer"}
KNOWN_SKILLS = {"yotta-researcher", "answer-writer", "roll-dice", "frontend-design"}
