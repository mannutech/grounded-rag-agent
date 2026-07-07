"""Build a :class:`ChatProvider` by name."""

from __future__ import annotations

from grounded_rag.core.errors import GroundedRagError
from grounded_rag.core.providers.anthropic_provider import AnthropicChatProvider
from grounded_rag.core.providers.base import ChatProvider
from grounded_rag.core.providers.cohere_provider import CohereChatProvider
from grounded_rag.core.providers.openai_provider import OpenAIChatProvider
from grounded_rag.core.types import CohereClient

#: Default judge model per provider (override via config).
DEFAULT_MODELS: dict[str, str] = {
    "cohere": "command-a-03-2025",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
}


def build_chat_provider(
    provider: str,
    model: str | None = None,
    *,
    cohere_client: CohereClient | None = None,
    api_key: str | None = None,
) -> ChatProvider:
    """Construct a chat provider.

    ``cohere`` reuses an existing :class:`CohereClient` (so the mock/wrapper and its
    rate limiting are shared); ``openai`` / ``anthropic`` build their own SDK client
    lazily from ``api_key`` or the standard env var.
    """
    key = provider.lower()
    if key not in DEFAULT_MODELS:
        raise GroundedRagError(f"unknown chat provider: {provider!r} (use cohere|openai|anthropic)")
    resolved_model = model or DEFAULT_MODELS[key]

    if key == "cohere":
        if cohere_client is None:
            raise GroundedRagError("cohere provider requires a cohere_client")
        return CohereChatProvider(cohere_client, resolved_model)
    if key == "openai":
        return OpenAIChatProvider(resolved_model, api_key=api_key)
    return AnthropicChatProvider(resolved_model, api_key=api_key)
