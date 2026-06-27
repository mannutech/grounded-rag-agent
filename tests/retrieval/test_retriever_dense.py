"""DenseRetriever wiring over a (mock) Cohere embedder."""

from __future__ import annotations

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.types import Chunk, RetrievalMode
from grounded_rag.retrieval.retriever import build_index, build_retriever


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d::{i}",
            doc_id="d",
            file_path="docs/d.md",
            ordinal=i,
            text=f"document chunk number {i} about retries and backoff",
            start_char=0,
            end_char=10,
            n_tokens=7,
        )
        for i in range(n)
    ]


def test_dense_retrieve_shape_and_ordering(client_embedder) -> None:  # type: ignore[no-untyped-def]
    config = RetrievalConfig(mode=RetrievalMode.DENSE, use_reranker=False, top_k=10, rerank_top_n=3)
    index = build_index(_chunks(6), client_embedder, config)
    retriever = build_retriever(config, index=index, embedder=client_embedder)

    result = retriever.retrieve("how do retries work")
    assert result.mode is RetrievalMode.DENSE
    assert result.reranked is False
    assert len(result.results) == 3  # cut to rerank_top_n
    assert all(sc.stage == "dense" for sc in result.results)
    scores = [sc.score for sc in result.results]
    assert scores == sorted(scores, reverse=True)
    assert [sc.rank for sc in result.results] == [0, 1, 2]
