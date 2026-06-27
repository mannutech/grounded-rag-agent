"""Adapt a core ``Retriever`` to the agent's ``AgentRetriever`` view.

The core retriever returns a rich ``RetrievalResult`` (``ScoredChunk`` carrying the
internal ``Chunk`` + stage). The agent wants a flat ``RetrievedChunk`` list it can
cite and gate on. This adapter does that mapping and surfaces the rerank score
(``ScoredChunk.score``) so the agent's weak-retrieval refusal check can use it.
"""

from __future__ import annotations

from grounded_rag.core.types import RetrievedChunk, Retriever, ScoredChunk


class RetrieverAdapter:
    """Wraps a core :class:`Retriever`, exposing ``retrieve(query, *, top_k)``."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    @staticmethod
    def _to_retrieved(scored: ScoredChunk) -> RetrievedChunk:
        chunk = scored.chunk
        return RetrievedChunk(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.file_path,
            score=scored.score,
            rank=scored.rank,
            stage=scored.stage,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        results = self._retriever.retrieve(query).results
        if top_k is not None:
            results = results[:top_k]  # narrows the configured result set
        return [self._to_retrieved(sc) for sc in results]
