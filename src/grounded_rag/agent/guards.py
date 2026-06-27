"""Input guarding + the prompt-injection threat-model note.

We *flag*, not block, suspected injection attempts: the real defence is structural
— every tool is read-only by construction (the registry enforces it), so even a
successful injection cannot cause a side effect. Flagging surfaces the attempt in
the trace for auditing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

PROMPT_INJECTION_NOTE = (
    "Threat model: retrieved documents and user input are untrusted DATA, never "
    "instructions. The system prompt tells the model to ignore embedded commands, "
    "and — defence in depth — every registered tool is read-only, so a successful "
    "injection still cannot delete, write, or exfiltrate anything. Suspected "
    "injection attempts are flagged into the trace (injection_flagged) for audit."
)

_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(?:all\s+)?(?:the\s+)?previous\s+instructions",
        r"disregard\s+(?:the\s+)?(?:above|previous|prior)",
        r"forget\s+(?:everything|all|your\s+instructions)",
        r"you\s+are\s+now\s+",
        r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
        r"(?:print|repeat|show)\s+(?:your\s+)?(?:system\s+)?prompt",
        r"new\s+instructions\s*:",
    )
]


@dataclass(frozen=True)
class GuardResult:
    """The cleaned query plus what guarding observed."""

    text: str
    injection_flagged: bool
    truncated: bool


def _strip_control_chars(text: str) -> str:
    """Drop Unicode control characters, keeping newlines and tabs."""
    return "".join(
        ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )


def guard_input(query: str, *, max_chars: int) -> GuardResult:
    """Clean and length-cap ``query``, flagging suspected prompt-injection."""
    cleaned = _strip_control_chars(query)
    truncated = len(cleaned) > max_chars
    if truncated:
        cleaned = cleaned[:max_chars]
    flagged = any(pattern.search(cleaned) for pattern in _INJECTION_PATTERNS)
    return GuardResult(text=cleaned, injection_flagged=flagged, truncated=truncated)
