"""Sparse lexical retrieval via BM25.

Wraps ``rank_bm25.BM25Okapi`` (lazy-imported so it isn't loaded until used). We
deliberately don't hand-roll the BM25 scoring loop — the IDF smoothing and the
k1/b length-normalisation term are easy to get subtly wrong and hard to notice.
The tokenizer *is* ours and deterministic so rankings are reproducible.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from grounded_rag.core.errors import RetrievalError
from grounded_rag.core.types import Chunk

_WORD_RE = re.compile(r"\w+")


class BM25Index:
    """A fitted BM25 index over a fixed, ordered corpus of chunks.

    Scores are returned aligned to ``fit`` order, so the same chunk list must
    feed both this index and the dense store for hybrid fusion indices to line up.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._bm25: object | None = None
        self._chunks: list[Chunk] = []

    def fit(self, chunks: list[Chunk]) -> None:
        """Build the index. An empty corpus is allowed (``scores`` returns empty)."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi  # lazy: keep import cost off the hot path

        corpus = [self.tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def scores(self, query: str) -> NDArray[np.float64]:
        """BM25 score for every chunk, aligned to ``fit`` order."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float64)
        if self._bm25 is None:
            raise RetrievalError("BM25Index.fit must be called before scores")
        raw = self._bm25.get_scores(self.tokenize(query))  # type: ignore[attr-defined]
        return np.asarray(raw, dtype=np.float64)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Deterministic tokenizer: lowercase, word characters only."""
        return _WORD_RE.findall(text.lower())

    def __len__(self) -> int:
        return len(self._chunks)
