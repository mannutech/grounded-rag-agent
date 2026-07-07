"""Cohere chat adapter — wraps the existing ``CohereClient`` as a ``ChatProvider``."""

from __future__ import annotations

from typing import Any

from grounded_rag.core.providers.base import ChatCompletion, safe_int
from grounded_rag.core.types import CohereClient, TokenUsage


class CohereChatProvider:
    """Adapts a :class:`CohereClient` (real wrapper or mock) to ``ChatProvider``."""

    name = "cohere"

    def __init__(self, client: CohereClient, model: str) -> None:
        self._client = client
        self.model = model

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            response_format=response_format,
        )
        content = getattr(resp.message, "content", None)
        text = getattr(content[0], "text", "") if content else ""
        usage = TokenUsage(
            input_tokens=safe_int(getattr(resp.usage.tokens, "input_tokens", 0)),
            output_tokens=safe_int(getattr(resp.usage.tokens, "output_tokens", 0)),
        )
        return ChatCompletion(
            text=text if isinstance(text, str) else "",
            usage=usage,
            finish_reason=getattr(resp, "finish_reason", "stop") or "stop",
            provider=self.name,
            model=self.model,
        )
