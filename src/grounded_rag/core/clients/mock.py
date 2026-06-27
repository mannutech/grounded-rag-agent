"""A deterministic, offline Cohere client double.

This is the lynchpin of the "mock-first" design: ``MockCohereClient`` satisfies
the :class:`grounded_rag.core.types.CohereClient` Protocol and returns objects
whose attribute paths mirror the **real** ``cohere.ClientV2`` response shapes
exactly — so the wrapper, adapter, agent, and eval code under test all run their
production paths with no network and no API key.

Attribute paths that must stay faithful to the SDK::

    embed:  resp.embeddings.float        -> list[list[float]]
            resp.meta.billed_units.input_tokens
    rerank: resp.results[i].index, .relevance_score   (sorted desc; no .document)
            resp.meta.billed_units.search_units
    chat:   resp.message.content[0].text
            resp.message.tool_calls[i].id / .function.name / .function.arguments  (JSON str)
            resp.message.tool_plan
            resp.message.citations[i].start / .end / .text / .sources[j].id
            resp.usage.tokens.input_tokens / .output_tokens
            resp.finish_reason

Determinism: embeddings are a stable function of the text (SHA-256 seeded), so the
same text always yields the same vector, distinct texts yield distinct vectors,
and prepending a file path (the retrieval A/B) yields a genuinely different
vector. Nothing here uses wall-clock time or process-salted ``hash()``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Response shapes (mirror cohere.ClientV2 pydantic models, attribute-for-attribute)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _BilledUnits:
    input_tokens: int = 0
    output_tokens: int = 0
    search_units: int = 0


@dataclass(frozen=True, slots=True)
class _Meta:
    billed_units: _BilledUnits


@dataclass(frozen=True, slots=True)
class _Embeddings:
    float: list[list[float]]  # mirrors the SDK attribute name exactly


@dataclass(frozen=True, slots=True)
class EmbedResponse:
    embeddings: _Embeddings
    meta: _Meta


@dataclass(frozen=True, slots=True)
class RerankResult:
    index: int
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RerankResponse:
    results: list[RerankResult]
    meta: _Meta


@dataclass(frozen=True, slots=True)
class _ContentBlock:
    text: str
    type: str = "text"


@dataclass(frozen=True, slots=True)
class _ToolCallFunction:
    name: str
    arguments: str  # JSON STRING, exactly like the SDK (callers must json.loads)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str  # mirrors the SDK attribute name
    function: _ToolCallFunction
    type: str = "function"


@dataclass(frozen=True, slots=True)
class _CitationSource:
    id: str  # mirrors the SDK attribute name
    type: str = "document"


@dataclass(frozen=True, slots=True)
class Citation:
    start: int
    end: int
    text: str
    sources: list[_CitationSource]


@dataclass(frozen=True, slots=True)
class _Message:
    content: list[_ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] | None = None
    tool_plan: str | None = None
    citations: list[Citation] | None = None
    role: str = "assistant"


@dataclass(frozen=True, slots=True)
class _Tokens:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class _Usage:
    tokens: _Tokens


@dataclass(frozen=True, slots=True)
class ChatResponse:
    message: _Message
    usage: _Usage
    finish_reason: str = "COMPLETE"


# ---------------------------------------------------------------------------
# Builder helpers — make scripting chat turns ergonomic in tests
# ---------------------------------------------------------------------------


def make_source(source_id: str) -> _CitationSource:
    """Build a citation source whose ``.id`` ties back to a document/tool result."""
    return _CitationSource(id=source_id)


def make_citation(start: int, end: int, text: str, source_ids: list[str]) -> Citation:
    """Build a citation spanning ``[start, end)`` of the answer."""
    return Citation(start=start, end=end, text=text, sources=[make_source(s) for s in source_ids])


def chat_text(
    text: str,
    *,
    citations: list[Citation] | None = None,
    input_tokens: int = 50,
    output_tokens: int = 20,
    tool_plan: str | None = None,
) -> ChatResponse:
    """A final-answer chat turn (no tool calls)."""
    return ChatResponse(
        message=_Message(
            content=[_ContentBlock(text=text)], tool_plan=tool_plan, citations=citations
        ),
        usage=_Usage(tokens=_Tokens(input_tokens=input_tokens, output_tokens=output_tokens)),
        finish_reason="COMPLETE",
    )


def chat_tool_calls(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    tool_plan: str | None = None,
    input_tokens: int = 40,
    output_tokens: int = 10,
    id_prefix: str = "call",
) -> ChatResponse:
    """A tool-calling chat turn. ``calls`` is a list of ``(tool_name, arguments)``.

    Arguments are JSON-serialised into ``.function.arguments`` exactly like the
    SDK, so the loop under test must ``json.loads`` them.
    """
    tool_calls = [
        ToolCall(
            id=f"{id_prefix}_{i}",
            function=_ToolCallFunction(name=name, arguments=json.dumps(args)),
        )
        for i, (name, args) in enumerate(calls)
    ]
    return ChatResponse(
        message=_Message(content=[], tool_calls=tool_calls, tool_plan=tool_plan),
        usage=_Usage(tokens=_Tokens(input_tokens=input_tokens, output_tokens=output_tokens)),
        finish_reason="TOOL_CALL",
    )


# ---------------------------------------------------------------------------
# Deterministic embedding
# ---------------------------------------------------------------------------


def _stable_seed(text: str) -> int:
    """A process-stable 64-bit seed for ``text`` (SHA-256, not salted ``hash()``)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def deterministic_embedding(text: str, dim: int) -> list[float]:
    """An L2-normalised, content-derived unit vector. Same text -> same vector."""
    rng = np.random.default_rng(_stable_seed(text))
    vec = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:  # vanishingly unlikely; keep it total
        return [0.0] * dim
    normed: list[float] = (vec / norm).tolist()
    return normed


def _estimate_tokens(text: str) -> int:
    """A deterministic ~4-chars-per-token estimate (>= 1)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# The mock client
# ---------------------------------------------------------------------------

ChatRouter = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], ChatResponse]
RerankReorder = Callable[[str, list[str]], list[tuple[int, float]]]


class MockCohereClient:
    """A deterministic stand-in for ``cohere.ClientV2`` (satisfies ``CohereClient``).

    Args:
        embed_dim: Dimensionality of the deterministic embeddings.
        rerank_reorder: Optional ``(query, documents) -> [(index, score), ...]`` to
            control rerank output (default: identity order, descending scores).
        chat_router: Optional ``(messages, tools) -> ChatResponse`` for full control
            over chat replies (used by the eval runner integration test).
        default_answer: The reply when no router is set and no script is queued.
    """

    is_mock: bool = True

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        rerank_reorder: RerankReorder | None = None,
        chat_router: ChatRouter | None = None,
        default_answer: str = "This is a grounded mock answer.",
    ) -> None:
        self.embed_dim = embed_dim
        self._rerank_reorder = rerank_reorder
        self._chat_router = chat_router
        self._default_answer = default_answer
        self.call_counts: dict[str, int] = {"embed": 0, "rerank": 0, "chat": 0}

    # -- embed --------------------------------------------------------------
    def embed(
        self,
        *,
        model: str,
        texts: list[str],
        input_type: str,
        embedding_types: list[str] | None = None,
    ) -> EmbedResponse:
        self.call_counts["embed"] += 1
        vectors = [deterministic_embedding(t, self.embed_dim) for t in texts]
        billed = _BilledUnits(input_tokens=sum(_estimate_tokens(t) for t in texts))
        return EmbedResponse(embeddings=_Embeddings(float=vectors), meta=_Meta(billed_units=billed))

    # -- rerank -------------------------------------------------------------
    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        max_tokens_per_doc: int = 4096,
    ) -> RerankResponse:
        self.call_counts["rerank"] += 1
        if self._rerank_reorder is not None:
            pairs = list(self._rerank_reorder(query, documents))
        else:
            # Identity order with strictly descending synthetic scores so ties
            # break by ascending index.
            n = len(documents)
            pairs = [(i, 1.0 - i / max(n, 1)) for i in range(n)]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        if top_n is not None:
            pairs = pairs[:top_n]
        results = [RerankResult(index=i, relevance_score=float(s)) for i, s in pairs]
        billed = _BilledUnits(search_units=1)
        return RerankResponse(results=results, meta=_Meta(billed_units=billed))

    # -- chat ---------------------------------------------------------------
    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        citation_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.call_counts["chat"] += 1
        if self._chat_router is not None:
            return self._chat_router(messages, tools)
        return chat_text(self._default_answer)


class ScriptedCohereClient(MockCohereClient):
    """A :class:`MockCohereClient` whose ``chat`` replays a fixed list of turns.

    Used to drive the agent's bounded tool-use loop deterministically (tool-call
    turn, then tool-error recovery, then a final cited answer). ``embed`` and
    ``rerank`` keep their deterministic behaviour from the base class.

    Args:
        turns: The chat responses to return, in order.
        repeat_last: When the script is exhausted, repeat the final turn instead
            of raising — useful for exercising the ``max_steps`` guard.
    """

    def __init__(
        self,
        turns: list[ChatResponse],
        *,
        repeat_last: bool = False,
        embed_dim: int = 1024,
        rerank_reorder: RerankReorder | None = None,
    ) -> None:
        super().__init__(embed_dim=embed_dim, rerank_reorder=rerank_reorder)
        if not turns:
            raise ValueError("ScriptedCohereClient requires at least one turn")
        self._turns = turns
        self._repeat_last = repeat_last
        self._idx = 0

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        citation_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        self.call_counts["chat"] += 1
        if self._idx >= len(self._turns):
            if self._repeat_last:
                return self._turns[-1]
            raise IndexError("ScriptedCohereClient exhausted its scripted turns")
        turn = self._turns[self._idx]
        self._idx += 1
        return turn
