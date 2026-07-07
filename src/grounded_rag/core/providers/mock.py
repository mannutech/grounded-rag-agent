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
        router: TextRouter | None = None,
        model: str = "mock-model",
        input_tokens: int = 10,
        output_tokens: int = 5,
    ) -> None:
        self.model = model
        self._text = text
        self._router = router
        self._input = input_tokens
        self._output = output_tokens
        self.calls = 0

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        self.calls += 1
        text = self._router(messages) if self._router is not None else self._text
        return ChatCompletion(
            text=text,
            usage=TokenUsage(input_tokens=self._input, output_tokens=self._output),
            provider=self.name,
            model=self.model,
        )
