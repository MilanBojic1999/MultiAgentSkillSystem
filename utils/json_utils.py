import json
import re

def extract_json(text: str) -> dict | list:
    """
    Extract JSON from LLM response, handling common failure modes:
    - Markdown code fences (```json ... ```)
    - Leading/trailing prose

    Handles both JSON objects and JSON arrays (e.g. the verifier's
    per-subquery verdict list).
    """
    # Try direct parse first (best case)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    # Try to find the outermost JSON array, then object.  Arrays first:
    # the object regex would match `{...}, {...}` inside a multi-item
    # array, which is not valid JSON on its own.
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON from response: {text[:500]}...")
