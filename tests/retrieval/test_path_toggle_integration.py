"""The path-in-embedding A/B is real end-to-end: it changes the document index."""

from __future__ import annotations

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.types import Chunk, RetrievalMode


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d::{i}",
            doc_id="d",
            file_path=f"docs/topic_{i}.md",
            ordinal=i,
            text=f"body text for chunk {i}",
            start_char=0,
            end_char=20,
            n_tokens=5,
        )
        for i in range(4)
    ]


def test_path_toggle_produces_a_different_index(client_embedder) -> None:  # type: ignore[no-untyped-def]
    from grounded_rag.retrieval.retriever import build_index, build_retriever

    chunks = _chunks()
    on_cfg = RetrievalConfig(
        mode=RetrievalMode.DENSE,
        use_reranker=False,
        embed_file_path=True,
        top_k=10,
        rerank_top_n=10,
    )
    off_cfg = on_cfg.model_copy(update={"embed_file_path": False})

    on = build_retriever(
        on_cfg, index=build_index(chunks, client_embedder, on_cfg), embedder=client_embedder
    ).retrieve("which topic covers chunk 2")
    off = build_retriever(
        off_cfg, index=build_index(chunks, client_embedder, off_cfg), embedder=client_embedder
    ).retrieve("which topic covers chunk 2")

    assert on.embed_file_path is True
    assert off.embed_file_path is False
    # The query embedding is identical; only the document vectors differ, so the
    # cosine scores must differ — proving the toggle reaches the index.
    assert [sc.score for sc in on.results] != [sc.score for sc in off.results]
