"""Cohere client surfaces: the Protocol, the real wrapper (later), and the mock.

Everything that talks to Cohere goes through the :class:`CohereClient` Protocol,
so the real ``CohereClientWrapper`` and the offline ``MockCohereClient`` are
interchangeable everywhere downstream.
"""

from __future__ import annotations

from grounded_rag.core.clients.cohere_client import (
    CohereClientWrapper,
    build_client,
    default_is_retryable,
)
from grounded_rag.core.clients.embedder import CohereEmbedder
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
from grounded_rag.core.clients.reranker import CohereReranker

__all__ = [
    "CohereClient",
    "MockCohereClient",
    "ScriptedCohereClient",
    # real client + adapters
    "CohereClientWrapper",
    "CohereEmbedder",
    "CohereReranker",
    "build_client",
    "default_is_retryable",
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
