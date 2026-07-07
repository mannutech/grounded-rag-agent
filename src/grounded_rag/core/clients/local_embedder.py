"""A local, dependency-light embedder (requires the ``local`` extra).

Uses ``model2vec`` static (distilled) embeddings — a lookup-table model with no
torch/ONNX and no GPU — so dense and hybrid retrieval can be **evaluated fully
offline** at zero API cost. It implements the same ``Embedder`` Protocol as the
Cohere embedder, so the retrieval stack is unchanged.

Quality is lower than a live API embedder, but it is real semantic retrieval —
enough to compare sparse vs dense vs hybrid on a large corpus without a key.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_MODEL = "minishlab/potion-base-8M"


class LocalEmbedder:
    """Implements ``Embedder`` via a local model2vec static model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, *, model: Any | None = None) -> None:
        if model is None:
            from model2vec import StaticModel  # lazy: only when selected

            model = StaticModel.from_pretrained(model_name)
        self._model = model
        self.dim = len(self._encode(["dimension probe"])[0])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in row] for row in self._model.encode(list(texts))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
