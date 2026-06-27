"""Optional Chroma-backed vector store (requires the ``chroma`` extra).

``chromadb`` is imported lazily inside ``__init__`` so importing this module — or
the whole ``grounded_rag.retrieval`` package — never pulls in Chroma. The default
``memory`` backend keeps CI dependency-free; this adapter exists to show the seam
is real and the ``VectorStore`` Protocol is genuinely pluggable.
"""

from __future__ import annotations

from typing import Any

from grounded_rag.core.types import Chunk, ScoredChunk


class ChromaVectorStore:
    """A :class:`~grounded_rag.retrieval.store.VectorStore` backed by Chroma (cosine)."""

    def __init__(self, *, dim: int, collection_name: str = "grounded_rag") -> None:
        import chromadb  # lazy: only when this backend is actually selected

        self.dim = dim
        self._client: Any = chromadb.Client()
        self._collection: Any = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "file_path": c.file_path,
                    "ordinal": c.ordinal,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "n_tokens": c.n_tokens,
                }
                for c in chunks
            ],
        )
        for c in chunks:
            self._chunks[c.chunk_id] = c

    def query(self, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        res = self._collection.query(
            query_embeddings=[embedding], n_results=min(top_k, len(self._chunks))
        )
        ids: list[str] = res["ids"][0]
        distances: list[float] = res["distances"][0]
        return [
            ScoredChunk(
                chunk=self._chunks[cid],
                score=1.0 - float(dist),  # cosine distance -> similarity
                rank=rank,
                stage="dense",
            )
            for rank, (cid, dist) in enumerate(zip(ids, distances, strict=True))
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)
