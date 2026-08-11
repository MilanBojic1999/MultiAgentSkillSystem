"""Minimal evaluation harness (plan 4.10): golden plan-shape tasks in
``evals/tasks/*.yaml`` run through the real orchestrator + a stub worker."""

from evals.runner import check_plan, default_tasks_dir, load_tasks, run_evals

__all__ = ["check_plan", "default_tasks_dir", "load_tasks", "run_evals"]
