"""Provider-agnostic chat surface.

The judge (and, later, generation) only needs *chat*: given messages, return text
+ token usage. ``ChatProvider`` is that one narrow contract; per-vendor adapters
(Cohere, OpenAI, Anthropic) normalise their native response shapes into a single
``ChatCompletion``. This is what lets the eval harness use a **different model
family as the judge** than the system under test — the fix for same-family
self-preference bias.

Retrieval (embed + rerank) is deliberately *not* part of this Protocol: OpenAI has
no reranker and Anthropic has neither embeddings nor rerank, so bundling them would
make the contract un-implementable. Retrieval stays on its own seams.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from grounded_rag.core.types import TokenUsage


def safe_int(value: Any) -> int:
    """Coerce a possibly-missing token count to an int."""
    return int(value) if isinstance(value, (int, float)) else 0


class ChatCompletion(BaseModel, frozen=True):
    """A normalised chat result, identical across providers."""

    text: str
    usage: TokenUsage
    finish_reason: str = "stop"
    provider: str = "unknown"
    model: str = "unknown"


@runtime_checkable
class ChatProvider(Protocol):
    """One vendor-neutral chat call. The model id is bound at construction."""

    name: str
    model: str

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion: ...
