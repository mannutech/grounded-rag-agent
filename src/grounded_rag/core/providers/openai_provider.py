"""OpenAI chat adapter (requires the ``openai`` extra).

The SDK is imported lazily so the package never hard-depends on it. A ``client``
can be injected for offline testing (a fake with the same ``chat.completions.create``
surface), so no network or key is needed to test the adapter.
"""

from __future__ import annotations

import os
from typing import Any

from grounded_rag.core.providers.base import ChatCompletion, safe_int
from grounded_rag.core.types import TokenUsage


class OpenAIChatProvider:
    """Adapts the OpenAI Chat Completions API to ``ChatProvider``."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        if client is None:
            import openai  # lazy: only when actually used

            client = openai.OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url
            )
        self._client = client

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if seed is not None:
            kwargs["seed"] = seed
        # OpenAI's json_object mode requires the token "json" somewhere in the prompt;
        # the judge rubric already contains "JSON", so this is safe for that use.
        if response_format is not None:
            kwargs["response_format"] = response_format
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = TokenUsage(
            input_tokens=safe_int(getattr(resp.usage, "prompt_tokens", 0)),
            output_tokens=safe_int(getattr(resp.usage, "completion_tokens", 0)),
        )
        return ChatCompletion(
            text=choice.message.content or "",
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            provider=self.name,
            model=self.model,
        )
