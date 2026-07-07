"""Anthropic (Claude) chat adapter (requires the ``anthropic`` extra).

Normalises two Anthropic-specific differences from the OpenAI/Cohere shape:

* the **system prompt is a separate parameter**, not a message with ``role:"system"``
  — system messages are extracted and concatenated into ``system=``.
* Anthropic has **no ``seed`` and no ``json_object`` response mode** — both are
  ignored here; JSON discipline is enforced by the judge prompt + the harness's
  defensive parser, which already tolerates prose/fences.

The SDK is imported lazily; a ``client`` can be injected for offline testing.
"""

from __future__ import annotations

import os
from typing import Any

from grounded_rag.core.providers.base import ChatCompletion, safe_int
from grounded_rag.core.types import TokenUsage

_DEFAULT_MAX_TOKENS = 1024


def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Pull ``role:"system"`` messages into a single system string; keep the rest."""
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
        else:
            conversation.append(message)
    return "\n\n".join(system_parts), conversation


class AnthropicChatProvider:
    """Adapts the Anthropic Messages API to ``ChatProvider``."""

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is None:
            import anthropic  # lazy

            client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._client = client

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        system, conversation = _split_system(messages)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system,
            messages=conversation,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "text") == "text"
        )
        usage = TokenUsage(
            input_tokens=safe_int(getattr(resp.usage, "input_tokens", 0)),
            output_tokens=safe_int(getattr(resp.usage, "output_tokens", 0)),
        )
        return ChatCompletion(
            text=text,
            usage=usage,
            finish_reason=getattr(resp, "stop_reason", "stop") or "stop",
            provider=self.name,
            model=self.model,
        )
