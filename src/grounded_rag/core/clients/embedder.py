"""Cohere embedding adapter implementing the ``Embedder`` Protocol.

Splits document vs query embedding so the correct ``input_type`` is always used —
``embed-english-v3.0`` requires ``search_document`` for the corpus and
``search_query`` for queries, and mixing them degrades retrieval. Works against
either the real wrapper or the mock, since both satisfy ``CohereClient``.
"""

from __future__ import annotations

from grounded_rag.core.types import CohereClient

# embed-english-v3.0 returns 1024-dimensional vectors.
_DEFAULT_EMBED_DIM = 1024
# Cohere's embed endpoint accepts at most 96 texts per request.
_MAX_TEXTS_PER_REQUEST = 96


class CohereEmbedder:
    """Adapts a :class:`CohereClient` to the dense ``Embedder`` Protocol."""

    def __init__(
        self,
        client: CohereClient,
        *,
        model: str = "embed-english-v3.0",
        dim: int = _DEFAULT_EMBED_DIM,
    ) -> None:
        self._client = client
        self.model = model
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch to respect the 96-texts-per-request limit (a real corpus exceeds it).
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_TEXTS_PER_REQUEST):
            batch = texts[start : start + _MAX_TEXTS_PER_REQUEST]
            resp = self._client.embed(
                model=self.model,
                texts=batch,
                input_type="search_document",
                embedding_types=["float"],
            )
            vectors.extend(resp.embeddings.float)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embed(
            model=self.model, texts=[text], input_type="search_query", embedding_types=["float"]
        )
        vector: list[float] = resp.embeddings.float[0]
        return vector
