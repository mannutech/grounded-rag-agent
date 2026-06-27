"""Shared retrieval test doubles — all offline, no Cohere client construction.

``ClientEmbedder`` / ``ClientReranker`` delegate to a ``MockCohereClient`` so they
exercise the same translation the real adapters will (deterministic, hash-seeded
vectors). ``StaticEmbedder`` returns caller-specified vectors so hybrid/rerank
tests can assert exact rankings.
"""

from __future__ import annotations

import pytest

from grounded_rag.core.clients import MockCohereClient


class ClientEmbedder:
    """An ``Embedder`` backed by a (mock) Cohere client."""

    def __init__(self, client: MockCohereClient) -> None:
        self._client = client
        self.dim = client.embed_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(
            model="embed-english-v3.0", texts=texts, input_type="search_document"
        ).embeddings.float

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed(
            model="embed-english-v3.0", texts=[text], input_type="search_query"
        ).embeddings.float[0]


class ClientReranker:
    """A ``Reranker`` backed by a (mock) Cohere client."""

    def __init__(self, client: MockCohereClient) -> None:
        self._client = client

    def rerank(self, *, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        res = self._client.rerank(
            model="rerank-v3.5", query=query, documents=documents, top_n=top_n
        )
        return [(r.index, r.relevance_score) for r in res.results]


class StaticEmbedder:
    """An ``Embedder`` returning exactly the vectors it is given (for crafted tests)."""

    def __init__(self, dim: int, vectors: dict[str, list[float]]) -> None:
        self.dim = dim
        self._vectors = vectors

    def _lookup(self, text: str) -> list[float]:
        if text not in self._vectors:
            raise KeyError(f"StaticEmbedder has no vector for {text!r}")
        return self._vectors[text]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._lookup(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._lookup(text)


@pytest.fixture
def mock_client() -> MockCohereClient:
    return MockCohereClient(embed_dim=64)


@pytest.fixture
def client_embedder(mock_client: MockCohereClient) -> ClientEmbedder:
    return ClientEmbedder(mock_client)


# Factory fixtures so tests can build doubles without importing classes across files.


@pytest.fixture
def make_mock_client():  # type: ignore[no-untyped-def]
    return MockCohereClient


@pytest.fixture
def make_client_embedder():  # type: ignore[no-untyped-def]
    return ClientEmbedder


@pytest.fixture
def make_client_reranker():  # type: ignore[no-untyped-def]
    return ClientReranker


@pytest.fixture
def make_static_embedder():  # type: ignore[no-untyped-def]
    return StaticEmbedder
