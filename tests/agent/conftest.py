"""Shared agent test doubles — a fake retriever and a chunk builder."""

from __future__ import annotations

import pytest

from grounded_rag.core.types import RetrievedChunk


class FakeRetriever:
    """Returns a fixed list of RetrievedChunk, honouring an optional top_k."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._chunks[:top_k] if top_k is not None else self._chunks


def rchunk(chunk_id: str, score: float, *, stage: str = "rerank") -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=chunk_id.split("::")[0],
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source="docs/a.md",
        score=score,
        rank=0,
        stage=stage,
    )


@pytest.fixture
def retriever_factory():  # type: ignore[no-untyped-def]
    return FakeRetriever


@pytest.fixture
def chunk():  # type: ignore[no-untyped-def]
    return rchunk
