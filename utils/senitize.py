import re
from typing import List

# Compile patterns once at startup for performance
# \s+ allows for variable whitespace, making it slightly harder to bypass
_INJECTION_PATTERNS: List[re.Pattern] = [
    # Classic jailbreaks
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"disregard\s+(previous|prior|all|your)\s+(instructions|rules|prompts|system)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(above|before|prior)", re.IGNORECASE),
    
    # Stricter "You are now" to avoid blocking normal text
    re.compile(r"you\s+are\s+now\s+(a|an|the|in)\s+(AI|assistant|bot|jailbreak|DAN)", re.IGNORECASE),
    
    # Data exfiltration via Markdown image tags
    re.compile(r"!\[.*\]\(https?://.*\)", re.IGNORECASE),
    
    # System prompt extraction
    re.compile(r"(print|output|repeat)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
]

def sanitize_content(text: str, source: str = "unknown") -> str:
    """
    Raises ValueError if the text looks like a prompt injection attempt.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"Potential prompt injection detected in content from '{source}': "
                f"matched pattern {pattern.pattern!r}"
            )
    return text