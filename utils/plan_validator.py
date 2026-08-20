from pydantic import BaseModel, Field, ValidationError

from utils.logger import log_event

'''
      "step": 1,
      "subtask": "<concise description>",
      "agent": "<agent_name>",
      "skills_needed": ["<skill-name>"],
      "depends_on": [],
      "files": ["<attached-filename>"]
'''

class PlanStepModel(BaseModel):
    step: int = Field(ge=1) # step can be >= 1
    subtask: str = Field(min_length=1) # specifc task text in this step
    agent: str = Field()
    skills_needed: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)
    # Attached documents assigned to this step. Kept on the model so the field
    # survives model_dump(): it is the only way a worker gets a document's full
    # text (the orchestrator sees 300-char previews only).
    files: list[str] = Field(default_factory=list)


class PlanValidationError(ValueError):
    """Raised when an orchestrator-produced plan is structurally or semantically invalid."""


def validate_plan(plan, known_agents, known_skills) -> list[dict]:
    if not isinstance(plan, list) or len(plan) == 0:
        raise PlanValidationError(f"Plan must be a non-empty list of steps, got: {plan!r}")

    steps: list[PlanStepModel] = []
    for index, raw_step in enumerate(plan):
        try:
            steps.append(PlanStepModel(**raw_step))
        except ValidationError as e:
            raise PlanValidationError(f"Plan step at index {index} failed schema validation: {e}") from e

    step_numbers = [s.step for s in steps]
    duplicates: set[int] = set(step_numbers)
    if len(step_numbers) > len(duplicates):
        raise PlanValidationError(f"Duplicate step numbers in plan: {sorted(duplicates)}")

    known_step_numbers = duplicates

    for s in steps:
        if s.agent not in known_agents:
            raise PlanValidationError(
                f"Step {s.step} references unknown agent '{s.agent}'. Known agents: {sorted(known_agents)}"
            )

        dropped_skills = [sk for sk in s.skills_needed if sk not in known_skills]
        if dropped_skills:
            log_event("plan_validation_unknown_skills_dropped", step=s.step, dropped_skills=dropped_skills)
            s.skills_needed = [sk for sk in s.skills_needed if sk in known_skills]

        if s.step in s.depends_on:
            raise PlanValidationError(f"Step {s.step} depends on itself.")

        dangling = [d for d in s.depends_on if d not in known_step_numbers]
        if dangling:
            raise PlanValidationError(f"Step {s.step} depends_on unknown step(s): {dangling}")

    # Kahn's algorithm: if not every step can be peeled off, a cycle remains.
    in_degree = {s.step: 0 for s in steps}
    dependents: dict[int, list[int]] = {s.step: [] for s in steps}
    for s in steps:
        for d in s.depends_on:
            dependents[d].append(s.step)
            in_degree[s.step] += 1

    queue = [n for n, deg in in_degree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if visited != len(steps):
        cyclic_steps = sorted(n for n, deg in in_degree.items() if deg > 0)
        raise PlanValidationError(f"Dependency cycle detected among steps: {cyclic_steps}")

    return sorted((s.model_dump() for s in steps), key=lambda d: d["step"])
