"""The rerank stage is genuinely wired: toggling it changes the output and flags."""

from __future__ import annotations

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.types import Chunk, RetrievalMode
from grounded_rag.retrieval.retriever import build_index, build_retriever


def _reverse(query: str, documents: list[str]) -> list[tuple[int, float]]:
    """A reranker that reverses candidate order (descending synthetic scores)."""
    n = len(documents)
    return [(n - 1 - i, 1.0 - 0.01 * i) for i in range(n)]


def _chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d::{i}",
            doc_id="d",
            file_path="docs/d.md",
            ordinal=i,
            text=f"chunk {i} text",
            start_char=0,
            end_char=12,
            n_tokens=3,
        )
        for i in range(n)
    ]


def test_rerank_toggle_changes_order_and_flags(  # type: ignore[no-untyped-def]
    make_mock_client, make_client_embedder, make_client_reranker
) -> None:
    client = make_mock_client(embed_dim=48, rerank_reorder=_reverse)
    embedder = make_client_embedder(client)
    reranker = make_client_reranker(client)
    chunks = _chunks(4)

    off_cfg = RetrievalConfig(
        mode=RetrievalMode.DENSE, use_reranker=False, top_k=10, rerank_top_n=10
    )
    index = build_index(chunks, embedder, off_cfg)

    off = build_retriever(off_cfg, index=index, embedder=embedder).retrieve("q")  # no reranker
    off_ids = [sc.chunk.chunk_id for sc in off.results]
    assert off.reranked is False
    assert all(sc.stage == "dense" for sc in off.results)

    on_cfg = off_cfg.model_copy(update={"use_reranker": True})
    on = build_retriever(on_cfg, index=index, embedder=embedder, reranker=reranker).retrieve("q")
    on_ids = [sc.chunk.chunk_id for sc in on.results]
    assert on.reranked is True
    assert all(sc.stage == "rerank" for sc in on.results)
    assert on_ids == list(reversed(off_ids))  # the reranker reversed the order
