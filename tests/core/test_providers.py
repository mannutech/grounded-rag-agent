"""Provider adapters normalise Cohere/OpenAI/Anthropic responses — all offline.

OpenAI/Anthropic SDKs are never imported: fake clients with the same surface are
injected, so these tests need no extra dependencies and no network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from grounded_rag.core.clients.mock import MockCohereClient, chat_text
from grounded_rag.core.errors import GroundedRagError
from grounded_rag.core.providers import (
    AnthropicChatProvider,
    ChatProvider,
    CohereChatProvider,
    MockChatProvider,
    OpenAIChatProvider,
    build_chat_provider,
)

# -- OpenAI -----------------------------------------------------------------


class _FakeOpenAI:
    def __init__(self) -> None:
        self.captured: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: object) -> object:
        self.captured = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hi from gpt"), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        )


def test_openai_adapter_normalises() -> None:
    fake = _FakeOpenAI()
    provider = OpenAIChatProvider("gpt-4o", client=fake)
    out = provider.complete(messages=[{"role": "user", "content": "hi"}], seed=5)
    assert out.text == "hi from gpt"
    assert out.usage.input_tokens == 7
    assert out.usage.output_tokens == 3
    assert out.provider == "openai" and out.model == "gpt-4o"
    assert fake.captured["seed"] == 5  # seed passed through


# -- Anthropic --------------------------------------------------------------


class _FakeAnthropic:
    def __init__(self) -> None:
        self.captured: dict = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: object) -> object:
        self.captured = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi from claude")],
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
            stop_reason="end_turn",
        )


def test_anthropic_adapter_splits_system_and_normalises() -> None:
    fake = _FakeAnthropic()
    provider = AnthropicChatProvider("claude-sonnet-4-5", client=fake)
    out = provider.complete(
        messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    )
    assert out.text == "hi from claude"
    assert out.usage.input_tokens == 8 and out.usage.output_tokens == 4
    # system extracted into its own param; conversation excludes it
    assert fake.captured["system"] == "be terse"
    assert fake.captured["messages"] == [{"role": "user", "content": "hi"}]


# -- Cohere -----------------------------------------------------------------


def test_cohere_adapter_normalises() -> None:
    client = MockCohereClient(chat_router=lambda m, t: chat_text("hi from cohere"))
    provider = CohereChatProvider(client, "command-a-03-2025")
    out = provider.complete(messages=[{"role": "user", "content": "hi"}])
    assert out.text == "hi from cohere"
    assert out.usage.total > 0
    assert out.provider == "cohere"


# -- mock + protocol + factory ---------------------------------------------


def test_all_adapters_satisfy_protocol() -> None:
    assert isinstance(MockChatProvider(), ChatProvider)
    assert isinstance(OpenAIChatProvider("gpt-4o", client=_FakeOpenAI()), ChatProvider)
    assert isinstance(CohereChatProvider(MockCohereClient(), "m"), ChatProvider)


def test_mock_provider_routes_text() -> None:
    provider = MockChatProvider(router=lambda m: "routed")
    assert provider.complete(messages=[]).text == "routed"


def test_factory_cohere_and_errors() -> None:
    client = MockCohereClient()
    provider = build_chat_provider("cohere", cohere_client=client)
    assert isinstance(provider, CohereChatProvider)
    assert provider.model == "command-a-03-2025"  # default resolved

    with pytest.raises(GroundedRagError, match="requires a cohere_client"):
        build_chat_provider("cohere")
    with pytest.raises(GroundedRagError, match="unknown chat provider"):
        build_chat_provider("llama", cohere_client=client)
