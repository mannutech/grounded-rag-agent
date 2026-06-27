"""Agent-local Protocols, re-exporting the canonical agent types from core.

The agent calls retrieval through a thin :class:`AgentRetriever` adapter (built in
``retriever_adapter.py``) that returns already-flattened ``RetrievedChunk`` lists
and lets the model request a ``top_k``. Everything else (``AgentResult``,
``Citation``, ``ToolCallRecord`` …) lives in ``grounded_rag.core``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    Citation,
    RetrievedChunk,
    StageTiming,
    TokenUsage,
    ToolCallRecord,
)

__all__ = [
    "AgentRetriever",
    "AgentResult",
    "AgentRunTrace",
    "Citation",
    "RetrievedChunk",
    "StageTiming",
    "TokenUsage",
    "ToolCallRecord",
]


@runtime_checkable
class AgentRetriever(Protocol):
    """The agent's view of retrieval: a flat, optionally-sized result list.

    ``top_k=None`` uses the retriever's configured final context size; a value
    overrides it for that call.
    """

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]: ...
