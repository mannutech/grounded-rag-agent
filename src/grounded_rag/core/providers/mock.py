"""A provider-native mock for offline judge/generation tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from grounded_rag.core.providers.base import ChatCompletion
from grounded_rag.core.types import TokenUsage

TextRouter = Callable[[list[dict[str, Any]]], str]


class MockChatProvider:
    """A ``ChatProvider`` that returns canned/routed text with fixed token usage."""

    name = "mock"
    is_mock = True

    def __init__(
        self,
        *,
        text: str = "{}",
        texts: list[str] | None = None,
        router: TextRouter | None = None,
        model: str = "mock-model",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        self.model = model
        self._text = text
        self._texts = texts  # served in order across calls (repeats the last when exhausted)
        self._router = router
        self._input = input_tokens
        self._output = output_tokens
        self.calls = 0

    def _next_text(self, messages: list[dict[str, Any]]) -> str:
        if self._router is not None:
            return self._router(messages)
        if self._texts is not None:
            return self._texts[min(self.calls - 1, len(self._texts) - 1)]
        return self._text

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        self.calls += 1
        text = self._next_text(messages)
        return ChatCompletion(
            text=text,
            usage=TokenUsage(input_tokens=self._input, output_tokens=self._output),
            provider=self.name,
            model=self.model,
        )
