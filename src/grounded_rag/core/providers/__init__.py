"""Provider-agnostic chat adapters (Cohere · OpenAI · Anthropic).

One narrow ``ChatProvider`` Protocol, three vendor adapters that normalise their
native responses into a single ``ChatCompletion``, a mock, and a factory. This is
what lets the eval harness judge with a *different* model family than the system
under test. OpenAI/Anthropic SDKs are imported lazily (optional extras).
"""

from __future__ import annotations

from grounded_rag.core.providers.anthropic_provider import AnthropicChatProvider
from grounded_rag.core.providers.base import ChatCompletion, ChatProvider, safe_int
from grounded_rag.core.providers.cohere_provider import CohereChatProvider
from grounded_rag.core.providers.factory import DEFAULT_MODELS, build_chat_provider
from grounded_rag.core.providers.mock import MockChatProvider
from grounded_rag.core.providers.openai_provider import OpenAIChatProvider

__all__ = [
    "ChatCompletion",
    "ChatProvider",
    "safe_int",
    "CohereChatProvider",
    "OpenAIChatProvider",
    "AnthropicChatProvider",
    "MockChatProvider",
    "build_chat_provider",
    "DEFAULT_MODELS",
]
