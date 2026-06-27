"""MockCohereClient: faithful SDK attribute paths + deterministic behaviour.

If these assertions hold, every downstream suite that runs against the mock is
exercising the same attribute access it would against the real ``cohere.ClientV2``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from grounded_rag.core.clients import (
    MockCohereClient,
    ScriptedCohereClient,
    chat_text,
    chat_tool_calls,
    make_citation,
)
from grounded_rag.core.types import CohereClient


def test_satisfies_cohere_client_protocol() -> None:
    client = MockCohereClient()
    assert isinstance(client, CohereClient)
    assert client.is_mock is True


# -- embed ------------------------------------------------------------------


def test_embed_shape_and_billed_units() -> None:
    client = MockCohereClient(embed_dim=16)
    resp = client.embed(
        model="embed-english-v3.0",
        texts=["hello world", "goodbye world"],
        input_type="search_document",
        embedding_types=["float"],
    )
    vectors = resp.embeddings.float  # exact SDK path
    assert len(vectors) == 2
    assert all(len(v) == 16 for v in vectors)
    assert resp.meta.billed_units.input_tokens > 0


def test_embed_is_deterministic_and_normalised() -> None:
    client = MockCohereClient(embed_dim=32)
    a1 = client.embed(model="m", texts=["alpha"], input_type="search_query").embeddings.float[0]
    a2 = client.embed(model="m", texts=["alpha"], input_type="search_query").embeddings.float[0]
    b = client.embed(model="m", texts=["beta"], input_type="search_query").embeddings.float[0]
    assert a1 == a2  # same text -> identical vector
    assert a1 != b  # distinct text -> distinct vector
    assert np.linalg.norm(a1) == pytest.approx(1.0, abs=1e-6)


def test_embed_path_prepend_changes_vector() -> None:
    """The retrieval path-embedding A/B relies on prepending changing the vector."""
    client = MockCohereClient(embed_dim=32)
    plain = client.embed(model="m", texts=["the body"], input_type="search_document")
    pathed = client.embed(model="m", texts=["docs/a.md\n\nthe body"], input_type="search_document")
    assert plain.embeddings.float[0] != pathed.embeddings.float[0]


# -- rerank -----------------------------------------------------------------


def test_rerank_default_is_sorted_descending_with_indices() -> None:
    client = MockCohereClient()
    resp = client.rerank(model="rerank-v3.5", query="q", documents=["d0", "d1", "d2"], top_n=2)
    assert [r.index for r in resp.results] == [0, 1]  # identity order, top_n=2
    scores = [r.relevance_score for r in resp.results]
    assert scores == sorted(scores, reverse=True)
    assert not hasattr(resp.results[0], "document")  # v2 has no return_documents
    assert resp.meta.billed_units.search_units == 1


def test_rerank_custom_reorder() -> None:
    def reverse(query: str, documents: list[str]) -> list[tuple[int, float]]:
        n = len(documents)
        return [(n - 1 - i, 0.9 - 0.1 * i) for i in range(n)]

    client = MockCohereClient(rerank_reorder=reverse)
    resp = client.rerank(model="rerank-v3.5", query="q", documents=["d0", "d1", "d2"])
    assert [r.index for r in resp.results] == [2, 1, 0]


# -- chat -------------------------------------------------------------------


def test_chat_default_final_answer() -> None:
    client = MockCohereClient(default_answer="grounded reply")
    resp = client.chat(model="command-a-03-2025", messages=[{"role": "user", "content": "hi"}])
    assert resp.message.content[0].text == "grounded reply"
    assert resp.message.tool_calls is None
    assert resp.usage.tokens.input_tokens > 0


def test_chat_tool_calls_arguments_are_json_strings() -> None:
    resp = chat_tool_calls([("search_docs", {"query": "retries", "top_k": 5})])
    tc = resp.message.tool_calls[0]  # type: ignore[index]
    assert tc.id == "call_0"
    assert tc.function.name == "search_docs"
    assert isinstance(tc.function.arguments, str)  # JSON STRING, like the SDK
    assert json.loads(tc.function.arguments) == {"query": "retries", "top_k": 5}
    assert resp.finish_reason == "TOOL_CALL"


def test_chat_text_with_citations() -> None:
    resp = chat_text("answer", citations=[make_citation(0, 6, "answer", ["d1::0", "d2::1"])])
    cits = resp.message.citations
    assert cits is not None
    assert cits[0].start == 0
    assert cits[0].end == 6
    assert [s.id for s in cits[0].sources] == ["d1::0", "d2::1"]


# -- scripted ---------------------------------------------------------------


def test_scripted_serves_turns_in_order() -> None:
    client = ScriptedCohereClient(
        [chat_tool_calls([("calculator", {"expr": "2+2"})]), chat_text("the answer is 4")]
    )
    first = client.chat(model="m", messages=[])
    second = client.chat(model="m", messages=[])
    assert first.message.tool_calls is not None
    assert second.message.content[0].text == "the answer is 4"


def test_scripted_repeat_last_for_max_steps() -> None:
    client = ScriptedCohereClient(
        [chat_tool_calls([("search_docs", {"query": "x"})])], repeat_last=True
    )
    for _ in range(5):
        resp = client.chat(model="m", messages=[])
        assert resp.message.tool_calls is not None  # never runs out


def test_scripted_raises_when_exhausted() -> None:
    client = ScriptedCohereClient([chat_text("only one")])
    client.chat(model="m", messages=[])
    with pytest.raises(IndexError):
        client.chat(model="m", messages=[])


def test_scripted_requires_at_least_one_turn() -> None:
    with pytest.raises(ValueError, match="at least one turn"):
        ScriptedCohereClient([])
