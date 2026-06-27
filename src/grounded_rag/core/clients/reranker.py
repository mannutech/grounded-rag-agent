"""Cohere rerank adapter implementing the ``Reranker`` Protocol.

Translates the v2 rerank response (results sorted by relevance, each with an
``index`` into the original documents and a ``relevance_score``) into the plain
``(original_index, score)`` pairs the retriever expects. The v2 API does not echo
document text, so callers map back to their own list via the index.
"""

from __future__ import annotations

from grounded_rag.core.types import CohereClient


class CohereReranker:
    """Adapts a :class:`CohereClient` to the ``Reranker`` Protocol."""

    def __init__(self, client: CohereClient, *, model: str = "rerank-v3.5") -> None:
        self._client = client
        self.model = model

    def rerank(self, *, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        resp = self._client.rerank(model=self.model, query=query, documents=documents, top_n=top_n)
        return [(int(r.index), float(r.relevance_score)) for r in resp.results]
