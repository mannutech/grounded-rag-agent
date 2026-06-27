"""Vector store: a Protocol, a numpy in-memory default, and the cosine math.

The in-memory store is the zero-dependency default. Heavier backends (Chroma,
pgvector) implement the same ``VectorStore`` Protocol behind lazy imports so the
package neither hard-depends on them nor changes any calling code.

Cosine ties always break by ascending index, which is what makes the math tests
assert exact orderings.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from grounded_rag.core.types import Chunk, ScoredChunk

Vector = NDArray[np.float64]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity of two vectors; 0.0 if either is a zero vector."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_top_k(query: Vector, matrix: Vector, k: int) -> list[tuple[int, float]]:
    """Top-``k`` rows of ``matrix`` by cosine to ``query``.

    Returns ``(row_index, score)`` pairs sorted by score descending, ties broken
    by ascending index. ``k`` larger than the corpus returns the whole corpus.
    """
    n_rows = int(matrix.shape[0])
    if n_rows == 0 or k <= 0:
        return []
    q_norm = float(np.linalg.norm(query))
    if q_norm == 0.0:
        return []
    row_norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(row_norms == 0.0, 1.0, row_norms)
    sims = (matrix @ query) / (safe * q_norm)
    sims = np.where(row_norms == 0.0, 0.0, sims)
    # Primary key: sims descending (sort -sims ascending). Secondary: index ascending.
    order = np.lexsort((np.arange(n_rows), -sims))
    cut = min(k, n_rows)
    return [(int(i), float(sims[i])) for i in order[:cut]]


@runtime_checkable
class VectorStore(Protocol):
    """Minimal dense index surface the retrievers depend on."""

    dim: int

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(self, embedding: list[float], top_k: int) -> list[ScoredChunk]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    """A numpy-backed cosine index. Rows are L2-normalised once at insert time."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        self._matrix: Vector = np.zeros((0, dim), dtype=np.float64)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        rows: list[Vector] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            if len(emb) != self.dim:
                raise ValueError(f"embedding dim {len(emb)} != store dim {self.dim}")
            vec = np.asarray(emb, dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            rows.append(vec / norm if norm > 0.0 else vec)
            self._index[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
        if rows:
            self._matrix = np.vstack([self._matrix, np.array(rows, dtype=np.float64)])

    def query(self, embedding: list[float], top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        query_vec = np.asarray(embedding, dtype=np.float64)
        pairs = cosine_top_k(query_vec, self._matrix, top_k)
        return [
            ScoredChunk(chunk=self._chunks[idx], score=score, rank=rank, stage="dense")
            for rank, (idx, score) in enumerate(pairs)
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        idx = self._index.get(chunk_id)
        return self._chunks[idx] if idx is not None else None

    def __len__(self) -> int:
        return len(self._chunks)
