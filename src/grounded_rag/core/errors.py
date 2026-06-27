"""Typed exceptions for the whole package.

A single small hierarchy so callers can catch ``GroundedRagError`` broadly or a
specific subtype narrowly.
"""

from __future__ import annotations


class GroundedRagError(Exception):
    """Base class for every error this package raises deliberately."""


class GoldParseError(GroundedRagError):
    """A line in ``eval/gold.jsonl`` was malformed or violated an invariant."""

    def __init__(self, line_no: int, message: str) -> None:
        self.line_no = line_no
        self.message = message
        super().__init__(f"gold.jsonl line {line_no}: {message}")


class JudgeParseError(GroundedRagError):
    """The LLM judge returned output that could not be parsed into a ballot."""


class ToolError(GroundedRagError):
    """A tool rejected its arguments or failed while executing.

    Raised inside the agent loop and fed back to the model as a tool-result
    error payload rather than crashing the run.
    """


class RetrievalError(GroundedRagError):
    """Retrieval could not be performed (e.g. an empty or unbuilt index)."""
