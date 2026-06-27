"""Cohere client surfaces: the Protocol, the real wrapper (later), and the mock.

Everything that talks to Cohere goes through the :class:`CohereClient` Protocol,
so the real ``CohereClientWrapper`` and the offline ``MockCohereClient`` are
interchangeable everywhere downstream.
"""

from __future__ import annotations

from grounded_rag.core.clients.mock import (
    ChatResponse,
    Citation,
    EmbedResponse,
    MockCohereClient,
    RerankResponse,
    ScriptedCohereClient,
    ToolCall,
    chat_text,
    chat_tool_calls,
    deterministic_embedding,
    make_citation,
    make_source,
)
from grounded_rag.core.clients.protocol import CohereClient

__all__ = [
    "CohereClient",
    "MockCohereClient",
    "ScriptedCohereClient",
    # response shapes
    "ChatResponse",
    "EmbedResponse",
    "RerankResponse",
    "ToolCall",
    "Citation",
    # builders
    "chat_text",
    "chat_tool_calls",
    "make_citation",
    "make_source",
    "deterministic_embedding",
]
