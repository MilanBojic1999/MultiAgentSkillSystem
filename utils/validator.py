"""
Output validation for sub-agent results.
Called before writing results to shared state to prevent
malicious or malformed outputs from propagating.
"""

import re
from typing import Any

MAX_OUTPUT_LENGTH = 50_000     # characters
ALLOWED_PATTERNS = [
    re.compile(r".+"),         # At least one non-whitespace character
]
BLOCKED_PATTERNS = [
    re.compile(r"<script", re.IGNORECASE),         # XSS vectors
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"```system", re.IGNORECASE),       # Prompt leakage
]


def validate_step_output(step_num: int, agent_name: str, output: Any) -> str:
    """
    Validate and sanitize a sub-agent's output before it enters shared state.

    Returns the (possibly sanitized) output string.
    Raises ValueError if the output is irredeemably invalid.
    """
    if not isinstance(output, str):
        raise ValueError(
            f"Step {step_num} ({agent_name}): expected str output, got {type(output).__name__}"
        )

    if not output.strip():
        raise ValueError(
            f"Step {step_num} ({agent_name}): output is empty or whitespace-only"
        )

    if len(output) > MAX_OUTPUT_LENGTH:
        raise ValueError(
            f"Step {step_num} ({agent_name}): output is {len(output)} chars "
            f"(max {MAX_OUTPUT_LENGTH})"
        )

    for pattern in BLOCKED_PATTERNS:
        if pattern.search(output):
            raise ValueError(
                f"Step {step_num} ({agent_name}): output contains blocked pattern "
                f"'{pattern.pattern}'"
            )

    return output
