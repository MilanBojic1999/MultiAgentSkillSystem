from agents.agent_states import ExecutionStatus, PipelineResult


def derive_status(failed_steps: list[int], step_stats: list[dict]) -> ExecutionStatus:
    """Terminal execution status, derived once at assembly (Slice 4).

    ``completed`` when no step failed or was skipped; ``partial`` when
    containment failed or skipped steps while assembly still produced usable
    output. ``failed`` and ``running`` never describe an assembled result —
    they are transport/lifecycle states handled by the API layer.
    """
    if failed_steps or any(s.get("status") == "skipped" for s in step_stats):
        return "partial"
    return "completed"


def skipped_steps_from_stats(step_stats: list[dict]) -> list[int]:
    """Steps transitively skipped by a failed dependency (Slice 4).

    Derived from the final ``step_stats`` rows — the single source of truth —
    rather than a second reducer-backed state field.
    """
    return sorted(s["step"] for s in step_stats if s.get("status") == "skipped")


def pipeline_result(state: dict) -> PipelineResult:
    """Normalize a terminal graph state into the typed pipeline result (Slice 4).

    Presentation boundaries (CLI, API) read exactly this shape:
    ``status``, ``final_output``, ``failed_steps``, ``skipped_steps`` and
    step-ordered ``step_stats``. ``skipped_steps`` is derived from the final
    stats, and ``status`` falls back to ``derive_status`` for graph modules
    that do not write it themselves.
    """
    step_stats = sorted(state.get("step_stats", []), key=lambda s: s["step"])
    failed_steps = sorted(state.get("failed_steps", []))
    return {
        "status": state.get("status") or derive_status(failed_steps, step_stats),
        "final_output": state.get("final_output", "No final output produced."),
        "failed_steps": failed_steps,
        "skipped_steps": skipped_steps_from_stats(step_stats),
        "step_stats": step_stats,
    }


def assemble_node(state: dict) -> dict:
    plan = state.get("plan", [])
    results = state.get("results", {})
    failed_steps = state.get("failed_steps", [])

    parts: list[str] = []
    if failed_steps:
        steps_str = ", ".join(str(s) for s in sorted(failed_steps))
        parts.append(
            f"> ⚠️ {len(failed_steps)} of {len(plan)} steps failed "
            f"(steps {steps_str}). The output below is partial."
        )

    parts += [
        f"## Step {s['step']}: {s['subtask']}\n{results.get(s['step'], '')}"
        for s in plan
    ]
    return {
        "final_output": "\n\n".join(parts),
        "status": derive_status(failed_steps, state.get("step_stats", [])),
    }
