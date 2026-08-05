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
    return {"final_output": "\n\n".join(parts)}
