"""The default ``memory`` backend never imports the optional Chroma dependency."""

from __future__ import annotations

import sys

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.types import Chunk, RetrievalMode
from grounded_rag.retrieval.retriever import build_index, build_retriever


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"d::{i}",
            doc_id="d",
            file_path="docs/d.md",
            ordinal=i,
            text=f"chunk {i}",
            start_char=0,
            end_char=7,
            n_tokens=2,
        )
        for i in range(3)
    ]


def test_memory_backend_does_not_import_chromadb(client_embedder) -> None:  # type: ignore[no-untyped-def]
    assert "chromadb" not in sys.modules  # importing the package must not pull it in

    config = RetrievalConfig(
        mode=RetrievalMode.DENSE, use_reranker=False, vector_store="memory", top_k=5, rerank_top_n=5
    )
    index = build_index(_chunks(), client_embedder, config)
    build_retriever(config, index=index, embedder=client_embedder).retrieve("chunk 1")

    assert "chromadb" not in sys.modules  # a full memory-backed retrieve stays Chroma-free
