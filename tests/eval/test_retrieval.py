"""Retrieval-only eval: metrics, sparse (BM25) retrieval, and no-embed guarantee."""

from __future__ import annotations

import pytest

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.types import (
    Chunk,
    GoldRecord,
    QueryType,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)
from grounded_rag.eval.metrics import reciprocal_rank
from grounded_rag.eval.retrieval import evaluate_retrieval
from grounded_rag.retrieval.retriever import build_index, build_retriever


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a"], ["a", "b"]) == 1.0
    assert reciprocal_rank(["b"], ["a", "b", "c"]) == pytest.approx(1 / 2)
    assert reciprocal_rank(["c"], ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["z"], ["a", "b"]) == 0.0


def _scored(doc_ids: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"{d}::0",
                doc_id=d,
                file_path="p",
                ordinal=0,
                text="t",
                start_char=0,
                end_char=1,
                n_tokens=1,
            ),
            score=1.0 - 0.1 * i,
            rank=i,
            stage="rerank",
        )
        for i, d in enumerate(doc_ids)
    ]


class _FakeRetriever:
    def __init__(self, per_query: dict[str, list[str]]) -> None:
        self._per_query = per_query

    def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            results=_scored(self._per_query[query]),
            mode=RetrievalMode.SPARSE,
            reranked=False,
            embed_file_path=False,
        )


def test_evaluate_retrieval_metrics_and_skips_must_refuse() -> None:
    gold = [
        GoldRecord(id="A", question="qa", type=QueryType.NORMAL, relevant_doc_ids=["dA"]),
        GoldRecord(id="B", question="qb", type=QueryType.NORMAL, relevant_doc_ids=["dB"]),
        GoldRecord(id="R", question="qr", type=QueryType.MUST_REFUSE, must_refuse=True),
    ]
    retriever = _FakeRetriever({"qa": ["dA", "dX"], "qb": ["dX", "dY", "dB"]})
    report = evaluate_retrieval(gold, retriever, k_values=[1, 5])

    assert report.n == 2  # must_refuse skipped
    assert report.recall_at_k[1] == pytest.approx(0.5)  # A hit, B miss at 1
    assert report.recall_at_k[5] == pytest.approx(1.0)  # both within 5
    assert report.mrr == pytest.approx((1.0 + 1 / 3) / 2)


def _chunk(doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::0",
        doc_id=doc_id,
        file_path=f"{doc_id}.md",
        ordinal=0,
        text=text,
        start_char=0,
        end_char=len(text),
        n_tokens=len(text.split()),
    )


class _ExplodingEmbedder:
    """Proves sparse retrieval never touches the embedder."""

    dim = 8

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("sparse mode must not embed")

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("sparse mode must not embed")


def test_sparse_retrieval_ranks_lexically_and_never_embeds() -> None:
    chunks = [
        _chunk("d0", "the cat sat on the mat"),
        _chunk("d1", "quantum entanglement in modern physics"),
        _chunk("d2", "the dog ran across the park"),
    ]
    config = RetrievalConfig(
        mode=RetrievalMode.SPARSE, use_reranker=False, top_k=10, rerank_top_n=5
    )
    index = build_index(chunks, _ExplodingEmbedder(), config)  # would raise if it embedded
    retriever = build_retriever(config, index=index, embedder=_ExplodingEmbedder())

    result = retriever.retrieve("quantum physics")
    assert result.mode is RetrievalMode.SPARSE
    assert result.results[0].chunk.doc_id == "d1"  # lexical match ranks first

    gold = [
        GoldRecord(
            id="g", question="quantum physics", type=QueryType.NORMAL, relevant_doc_ids=["d1"]
        )
    ]
    report = evaluate_retrieval(gold, retriever, k_values=[1, 3])
    assert report.recall_at_k[1] == 1.0
