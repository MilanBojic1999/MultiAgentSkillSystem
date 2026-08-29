# Execution Effort Controls

## Context

The plan-and-execute pipeline currently has no user-facing workload policy: the orchestrator may produce any number of plan steps, a ReAct worker may make an unconstrained number of tool calls, parallel scheduling fans out every ready step, and only individual mechanisms have local limits. The user needs a per-run **effort slider** that makes those trade-offs deliberate.

The chosen product contract is a backend/API feature (a future frontend renders the slider), with named presets:

```text
Instant → Light → Standard → Thorough → Unlimited
```

- The selection is **per run**; it is not stored as a server/user preference.
- **Instant** bypasses ordinary planning and makes **exactly one configured writer-worker invocation**.
- Normal modes must include a real verification and correction loop.
- Worker retries and full replans are separate budgets.
- **Unlimited** means compatibility with the current high-effort behavior—not literal unbounded execution. Every run retains cancellation and hard safety ceilings.
- The effort-aware default topology is the existing **`yotta`** graph, because it already provides the required `planner → workers → verifier → writer` lifecycle. `parallel` and `sequential` remain explicitly selectable compatibility topologies.

The implementation must treat effort as an execution-policy contract, not as an LLM-provider `reasoning_effort` parameter: `llm_factory.create_llm()` targets arbitrary OpenAI-compatible endpoints, so a universal provider-specific reasoning control would be invalid or silently ignored.

## Recommended execution-policy contract

Create one central, serializable policy module—for example, `execution_policy.py`—rather than scattering preset checks through API, graph, and worker code.

### Public preset and normalization

- Define a canonical `EffortPreset` enum/string type: `instant`, `light`, `standard`, `thorough`, `unlimited`.
- Accept case-insensitive transport input (`"Instant"`, `"instant"`), normalize it once at the boundary, and use canonical lower-case values internally.
- Resolve an omitted effort field to `unlimited`.
- Expose a single `resolve_execution_policy(effort)` function that returns a plain serializable budget mapping suitable for `RunnableConfig["configurable"]` and checkpointing.

### Policy fields

The resolved policy must make every requested cost/quality lever explicit:

- `plan_enabled` and `instant_writer_only`
- `max_plan_steps`
- `max_worker_attempts` (total executions per step, including first execution)
- `max_tool_calls_per_attempt`
- `max_verification_attempts`
- `max_step_verification_retries` (retry a deficient worker result with verifier feedback)
- `max_replans` (return to the orchestrator with verifier feedback)
- `max_graph_dispatches` / scheduler iterations, to prevent pathological re-entry
- `react_recursion_limit`, to bound the ReAct model/tool loop even when no tool is called
- a wall-clock deadline or timeout budget propagated through the run context

Use these initial values as explicit, documented, test-pinned defaults; they are deliberately conservative and can later be tuned from observed `StepStats`:

| Preset | Ordinary planner | Plan steps | Worker attempts | Tool calls / attempt | Step-verification retries | Full replans | Verification attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Instant | no | one synthetic writer step | 1 | 0 | 0 | 0 | 0 |
| Light | yes | 3 | 1 | 2 | 0 | 0 | 1 |
| Standard | yes | 8 | 2 | 5 | 1 | 1 | 1 |
| Thorough | yes | 16 | 3 | 10 | 2 | 2 | 2 |
| Unlimited | yes | existing behavior, subject to hard ceiling | existing per-agent configuration | high hard ceiling | existing yotta behavior | existing yotta behavior | existing yotta behavior |

For **Unlimited**, define hard safety values separately (for example: 64 plan steps/total dispatches, the already-validated worker cap of 10 total attempts, a bounded ReAct recursion limit, and the existing yotta three-pass replan ceiling). These ceilings protect availability but do not alter ordinary current runs. Never implement an unbounded `while` loop or remove cancellation propagation.

### Interaction with agent configuration

Keep `agents/agent_config.json`'s static `execution.max_attempts` validation and `config_loader.get_max_attempts()` behavior. Compute effective attempts as:

```text
effective_worker_attempts = min(agent_configured_max_attempts, policy.max_worker_attempts)
```

Thus agent configuration remains an upper limit, Instant always remains one writer attempt, and Unlimited retains the current per-agent behavior (default two attempts; researcher’s configured three attempts). Do not change provider model, temperature, or `max_tokens` settings as a side effect of selecting effort.

## Implementation plan

1. **Add the policy domain model and its hermetic tests.**
   - Add `execution_policy.py` containing the preset enum/parser, immutable/serializable `ExecutionPolicy` representation, preset table, safety ceilings, a `resolve_execution_policy()` resolver, `effective_worker_attempts()`, and helpers to extract the policy from `RunnableConfig`.
   - Make the resolver the only source of defaults and numeric budgets. It must be independent of FastAPI so direct CLI and tests reuse exactly the same behavior.
   - Add `tests/test_execution_policy.py` for normalization, invalid values, omitted → Unlimited, every exact preset budget, separation of worker/verification/replan counters, static-agent-attempt intersection, and Unlimited’s finite safety bounds.

2. **Add effort to FastAPI, direct CLI, and HTTP-client CLI without persistence.**
   - Update `api_server.py`:
     - Add an optional validated `effort` field to `RunRequest`; clients receive FastAPI’s normal 422 validation for invalid names.
     - Resolve it in `_run_pipeline()` and include only serializable values under `config["configurable"]` beside `thread_id` and `task_id` (`effort`, `execution_policy`, and a deadline/start timestamp as needed).
     - Select `yotta` as the default graph for requests that do not explicitly choose a graph, per the chosen verification-first product behavior; retain explicit `parallel`/`sequential` selection.
     - Return selected `effort` plus backward-compatible execution/verification metadata (verification outcome, corrective retry/replan counts, and any safety-stop reason) on `RunResponse` and async `StatusResponse`, all with safe defaults so old clients remain compatible.
     - Preserve the choice in `_task_store` for async status reporting only; do not turn it into a persisted preference.
   - Update `run_pipeline.py` to accept `effort: str | None` in `run()`/`run_async()`, add `--effort` choices, resolve through the shared policy, and inject it into `_run_config()`.
   - Update `api_client.py` with `--effort`, extending `_run_body()` for both `/run` and `/run-async`; omission intentionally lets the server resolve Unlimited.
   - Extend `tests/test_api_server.py` and CLI/client tests to cover sync/async propagation, default resolution, invalid input, response compatibility, and `task_id`/artifact config preservation.

3. **Make the `yotta` graph policy-aware and make it the effort topology.**
   - Modify `graphs/yotta_graph.py`, which already owns the correct lifecycle: planner, dependency-aware parallel workers, `verify_node`, corrective step retry/replan routing, and `writer_node` synthesis.
   - Replace its hard-coded `_MAX_REPLANS`, `current_retries < 2`, verifier retry assumptions, and writer/worker attempt lookup with policy-derived caps. Continue to use `run_step_with_attempts()` as the single retry owner—do not add nested worker `RetryPolicy` instances.
   - Pass the policy through every `Send` payload and every worker/verifier/writer invocation so budget enforcement has the same meaning in normal execution, correction passes, and synthesis.
   - Add a graph-level dispatch/scheduler counter to `YottaState`; increment on each new execution wave and fail or finalize safely with a structured `safety_stop_reason` if the policy’s finite ceiling is reached.
   - Preserve `YottaState`’s reset-aware results reducer when a replan replaces the plan. The replan path must carry the verifier feedback in state for the planner, reset stale results correctly, and never rerun a previously accepted result merely because its numeric step ID was reused.
   - Keep yotta’s direct empty-plan → writer route for its search-sufficient case, but distinguish it from Instant’s guaranteed one-writer route in logs/metadata.

4. **Implement Instant as a graph route, not an API short circuit.**
   - Add an entry/router node in `graphs/yotta_graph.py` (or a small shared graph-local route helper) that reads the resolved policy before `orchestrator`.
   - For `instant`, construct one synthetic `PlanStep` assigned to the configured/validated `writer` agent with the original task, set a one-step plan, and dispatch it directly to the existing yotta writer-worker machinery.
   - Route that worker result to normal assembly/final-output handling while skipping the ordinary orchestrator, scheduler fan-out, verifier, step retry, and full replan nodes.
   - Ensure the path preserves `RunnableConfig` so file/artifact context, thread ID, task ID, logging, output validation, failure containment, `PipelineResult`, and cancellation semantics match normal runs.
   - Add graph tests that prove: zero orchestrator invocations; exactly one writer invocation; no tool calls; no verifier/replan attempt; one normal stats/result row; and correct contained-failure/cancellation behavior.

5. **Enforce plan, worker, tool, and ReAct-loop budgets at the execution point.**
   - Update `agents/orchestrator_node.py` to resolve the current policy from graph state/config and enforce `max_plan_steps` *after* `validate_plan()` normalizes the JSON plan. A plan exceeding the selected cap produces precise feedback suitable for a bounded replan, not silent truncation that breaks dependencies.
   - Add policy-aware planning feedback to the yotta replan prompt/state. The existing planner system instructions already describe `verifier_report` and stable step IDs; extend that contract to name policy violations and remaining correction budget.
   - Update `agents/sub_agents_nodes.py` to calculate effective attempts using the policy and static per-agent config. Thread the policy to verifier and writer synthetic steps as well as ordinary workers.
   - Enforce tool-call count **during** ReAct execution rather than only counting it after `agent.ainvoke()` returns. Implement a per-attempt tool-budget guard/callback (or schema-preserving wrapped tools if the installed LangGraph version requires it) that increments a local counter on tool start and raises a dedicated controlled budget exception once the cap is exceeded. On exhaustion, resume the agent once on the same checkpointer thread with a strict finalize instruction (no more tools) so the step succeeds with the information already retrieved; the bounded attempt/containment policy decides retry-vs-contain only if that finalize pass itself requests tools. Record the count in existing `StepStats`.
   - Set the policy’s `react_recursion_limit` in the runnable invocation config, rather than relying on LangGraph defaults. This bounds model/tool-turn recursion even for unusual tool patterns.
   - Do not pass speculative provider-specific `reasoning_effort`/`reasoning` parameters through `ChatOpenAI`; effort is enforced by pipeline orchestration and budgets.
   - Extend `tests/test_execution_attempts.py`, `tests/test_orchestrator_node.py`, and worker tests for plan-cap failure/replan feedback, capped attempts, real-time tool-budget blocking, ReAct recursion propagation, no nested retry multiplication, cancellation escaping, and Unlimited safety behavior.

6. **Complete the verification/correction state contract and presentation semantics.**
   - Extend `agents/agent_states.py` / `YottaState` with typed, reducer-safe fields for the selected effort/policy, verification status/report, verifier attempts, per-step verifier retry counts, full replan count, dispatch count, and `safety_stop_reason`.
   - Keep the two correction budgets independent:
     - **Per-step verifier retry**: re-dispatch only the failed result with feedback; bounded by `max_step_verification_retries`.
     - **Full replan**: invoke the planner with verifier report and revised task plan; bounded by `max_replans`.
   - In `verify_node`, replace literal `2` values with policy values, use the policy’s verifier-attempt cap, and make verification exhaustion deterministic. If quality verification cannot be resolved after its allowed correction budget, route to the writer with an explicit partial/verification-exhausted result—not an infinite loop or a transport-level 500 for an otherwise usable result.
   - Update `writer_node` to include verifier notes and safety/partial warnings in the final artifact without leaking internal prompts, credentials, or raw hidden state.
   - Update `assemble_node.py` / `pipeline_result()` to derive `partial` when verification exhausts, retain existing failed/skipped semantics, and normalize the newly exposed response metadata for both API and CLI.
   - Add yotta-focused integration tests for verification pass, step retry, replan, mixed verdicts, replan exhaustion, no duplicate verifier invocation, result reset behavior, and current Unlimited-equivalent behavior.

7. **Keep compatibility topology behavior explicit.**
   - Preserve `graphs/parallel_pipeline_graph.py` and `graphs/sequential_pipeline_graph.py` as opt-in compatibility graphs; do not silently graft yotta verification into them in this feature.
   - Document that effort-controlled default runs use yotta because verification/writer synthesis is part of the contract. A caller choosing `parallel` or `sequential` opts into their existing non-verifying assemble behavior; either reject non-Instant effort for those graphs with a clear 422/400 validation error, or surface `verification_supported: false` metadata. **Recommended:** reject non-Instant non-Unlimited effort values for these graphs to avoid falsely claiming verification; permit `unlimited` as the backwards-compatible legacy mode and route Instant through yotta unless the caller explicitly chooses the `yotta` graph.
   - Test graph/effort compatibility at the API boundary so behavior is intentional and discoverable.

8. **Document the slider-ready API contract and roadmap.**
   - Update `README.md` with a preset table, guarantees and safety limits, API request examples, direct CLI/API-client `--effort` examples, yotta-default behavior, Instant’s one-writer/no-tool guarantee, Unlimited’s compatibility-but-finite meaning, correction-loop semantics, and compatibility guidance for explicit legacy graphs.
   - Update `.env.example` only if a hard safety ceiling must be deployment-configurable; keep the default product presets in code so behavior is predictable. Do not add a global effort preference.
   - Add the feature and its acceptance criteria to `IMPROVEMENT_PLAN.md`, noting that it uses yotta rather than creating a fourth independently overlapping verification graph.
   - Log `execution_policy_resolved`, `instant_route_selected`, `tool_budget_exhausted`, `tool_budget_finalized`, `tool_budget_finalize_failed`, `verification_started`, `verification_finished`, `step_retry_scheduled`, `replan_scheduled`, `replan_exhausted`, and `effort_safety_stop` with `utils.logger.log_event`.

## Critical files

**New**
- `execution_policy.py`
- `tests/test_execution_policy.py`

**Core implementation**
- `api_server.py`
- `run_pipeline.py`
- `api_client.py`
- `graphs/yotta_graph.py`
- `agents/orchestrator_node.py`
- `agents/sub_agents_nodes.py`
- `agents/agent_states.py`
- `assemble_node.py`

**Compatibility and documentation**
- `graphs/parallel_pipeline_graph.py`
- `graphs/sequential_pipeline_graph.py`
- `README.md`
- `IMPROVEMENT_PLAN.md`
- `.env.example` (only if deployment-level hard ceilings become configurable)

**Representative test updates**
- `tests/test_api_server.py`
- `tests/test_orchestrator_node.py`
- `tests/test_execution_attempts.py`
- `tests/test_dispatch_dedup.py`
- yotta-specific graph/verification test module(s)

## Verification

1. Run the hermetic suite from the repository root:

   ```bash
   pytest
   ```

   It must cover every preset, parser/default behavior, API/CLI propagation, static-agent/policy attempt intersection, finite Unlimited guards, planning cap, tool budget, and cancellation.

2. Run focused graph tests with injected fake orchestrator, worker, verifier, and writer functions:
   - Instant: planner `0`, writer worker `1`, verifier `0`, tool invocations `0`.
   - Light/Standard/Thorough: assert exact caps for planned steps, attempts, tool calls, verifier calls, step retries, and full replans.
   - Unlimited: prove existing configured retry behavior remains while hard ceilings still terminate a deliberately pathological loop.
   - Replan: assert result reset and verifier feedback reach the planner; assert no duplicate dispatch/verification after a correction pass.
   - Containment: independent steps still complete; transitive dependents remain skipped; terminal outcome and metadata are `partial` where appropriate.

3. Exercise end-to-end entry points against a configured local model/MCP environment:

   ```bash
   python run_pipeline.py --graph yotta --effort instant "Summarize this task"
   python run_pipeline.py --graph yotta --effort standard "Research and explain X"
   python api_client.py --graph yotta --effort thorough "Research and explain X"
   ```

   Confirm API JSON includes selected effort and verification metadata, `/run-async` polling preserves the same information, and generated artifacts still resolve under the existing `task_id` path.

4. Check the documented compatibility matrix through `GET /graphs` plus API validation: `yotta` supports all effort presets; explicitly selected legacy `parallel`/`sequential` retain their documented behavior and do not silently claim verification guarantees.
