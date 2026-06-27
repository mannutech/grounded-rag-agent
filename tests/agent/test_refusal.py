"""Refusal paths: weak retrieval, empty retrieval, sentinel, unsupported, max_steps."""

from __future__ import annotations

from grounded_rag.agent.loop import run_agent
from grounded_rag.agent.prompts import INSUFFICIENT_CONTEXT_SENTINEL, REFUSAL_TEXT
from grounded_rag.core.clients.mock import (
    ScriptedCohereClient,
    chat_text,
    chat_tool_calls,
    make_citation,
)
from grounded_rag.core.config import AgentConfig, CoherePricing

_PRICING = CoherePricing()
_MODEL = "command-a-03-2025"


def _run(client, retriever, config=None):  # type: ignore[no-untyped-def]
    return run_agent(
        "a question",
        retriever=retriever,
        client=client,
        config=config or AgentConfig(),
        pricing=_PRICING,
        model=_MODEL,
    )


def test_weak_retrieval_refuses(retriever_factory, chunk) -> None:  # type: ignore[no-untyped-def]
    # retrieval returns a chunk below the rerank threshold (0.30)
    client = ScriptedCohereClient(
        [
            chat_tool_calls([("search_docs", {"query": "q"})]),
            chat_text("an answer", citations=[make_citation(0, 2, "an", ["d1::0"])]),
        ]
    )
    result = _run(client, retriever_factory([chunk("d1::0", 0.10)]))
    assert result.refused is True
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert result.steps >= 1


def test_empty_retrieval_refuses(retriever_factory) -> None:  # type: ignore[no-untyped-def]
    client = ScriptedCohereClient(
        [chat_tool_calls([("search_docs", {"query": "q"})]), chat_text("an answer")]
    )
    result = _run(client, retriever_factory([]))
    assert result.refused is True


def test_sentinel_refuses_even_with_good_retrieval(retriever_factory, chunk) -> None:  # type: ignore[no-untyped-def]
    client = ScriptedCohereClient(
        [
            chat_tool_calls([("search_docs", {"query": "q"})]),
            chat_text(f"{INSUFFICIENT_CONTEXT_SENTINEL}"),
        ]
    )
    result = _run(client, retriever_factory([chunk("d1::0", 0.95)]))
    assert result.refused is True


def test_unsupported_answer_refuses(retriever_factory, chunk) -> None:  # type: ignore[no-untyped-def]
    # good retrieval, but the answer cites nothing -> unsupported -> refuse
    client = ScriptedCohereClient(
        [chat_tool_calls([("search_docs", {"query": "q"})]), chat_text("an uncited answer")]
    )
    result = _run(client, retriever_factory([chunk("d1::0", 0.95)]))
    assert result.refused is True


def test_max_steps_exhausted_refuses_without_error(retriever_factory, chunk) -> None:  # type: ignore[no-untyped-def]
    # the model always asks for a tool, never gives a final answer
    client = ScriptedCohereClient(
        [chat_tool_calls([("search_docs", {"query": "x"})])], repeat_last=True
    )
    result = _run(
        client, retriever_factory([chunk("d1::0", 0.95)]), config=AgentConfig(max_steps=3)
    )
    assert result.refused is True
    assert result.steps == 3
