"""InMemoryVectorStore: insert/query/get round-trips and validation."""

from __future__ import annotations

import pytest

from grounded_rag.core.types import Chunk
from grounded_rag.retrieval.store import InMemoryVectorStore


def _chunk(idx: int) -> Chunk:
    return Chunk(
        chunk_id=f"d::{idx}",
        doc_id="d",
        file_path="docs/d.md",
        ordinal=idx,
        text=f"chunk {idx}",
        start_char=0,
        end_char=7,
        n_tokens=2,
    )


def test_add_then_query_returns_dense_scored_chunks_descending() -> None:
    store = InMemoryVectorStore(dim=2)
    store.add(
        [_chunk(0), _chunk(1), _chunk(2)],
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
    )
    results = store.query([1.0, 0.0], top_k=3)
    assert [r.chunk.chunk_id for r in results] == ["d::0", "d::2", "d::1"]
    assert all(r.stage == "dense" for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == [0, 1, 2]


def test_len_and_get_round_trip() -> None:
    store = InMemoryVectorStore(dim=2)
    store.add([_chunk(0), _chunk(1)], [[1.0, 0.0], [0.0, 1.0]])
    assert len(store) == 2
    assert store.get("d::1") == _chunk(1)
    assert store.get("missing") is None


def test_query_empty_store_returns_empty() -> None:
    assert InMemoryVectorStore(dim=3).query([1.0, 0.0, 0.0], top_k=5) == []


def test_dim_mismatch_raises() -> None:
    store = InMemoryVectorStore(dim=3)
    with pytest.raises(ValueError, match="dim"):
        store.add([_chunk(0)], [[1.0, 0.0]])


def test_length_mismatch_raises() -> None:
    store = InMemoryVectorStore(dim=2)
    with pytest.raises(ValueError, match="same length"):
        store.add([_chunk(0), _chunk(1)], [[1.0, 0.0]])
