"""Deterministic, token-aware document chunking.

Splits a document into fixed-size windows with overlap. "Tokens" default to
whitespace-delimited runs — a dependency-free approximation that is fully
deterministic; a real BPE tokenizer can be injected via ``token_estimator``
without changing any downstream code.

Determinism guarantees (relied on by the no-network tests):

* ``chunk_id == f"{doc_id}::{ordinal}"`` and the same input always yields the
  same ids and char offsets.
* ``chunk.text == document_text[chunk.start_char:chunk.end_char]``.
* ``Chunk.text`` never contains the file path.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from grounded_rag.core.types import Chunk

_TOKEN_RE = re.compile(r"\S+")

# A token estimator maps text -> a list of (start_char, end_char) spans.
TokenEstimator = Callable[[str], list[tuple[int, int]]]


def _whitespace_token_spans(text: str) -> list[tuple[int, int]]:
    """Token spans = maximal runs of non-whitespace characters."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


class Chunker:
    """Splits documents into overlapping, deterministic chunks.

    Args:
        chunk_tokens: Maximum tokens per chunk.
        overlap: Tokens shared between consecutive chunks (must be < chunk_tokens).
        token_estimator: Optional override mapping text to token spans.
    """

    def __init__(
        self,
        *,
        chunk_tokens: int = 512,
        overlap: int = 64,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if overlap < 0 or overlap >= chunk_tokens:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap = overlap
        self._token_spans = token_estimator or _whitespace_token_spans

    def split(self, *, doc_id: str, file_path: str, text: str) -> list[Chunk]:
        """Split a single document into chunks (empty/whitespace -> no chunks)."""
        spans = self._token_spans(text)
        if not spans:
            return []

        step = self.chunk_tokens - self.overlap
        chunks: list[Chunk] = []
        n = len(spans)
        start_idx = 0
        ordinal = 0
        while start_idx < n:
            window = spans[start_idx : start_idx + self.chunk_tokens]
            start_char = window[0][0]
            end_char = window[-1][1]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{ordinal}",
                    doc_id=doc_id,
                    file_path=file_path,
                    ordinal=ordinal,
                    text=text[start_char:end_char],
                    start_char=start_char,
                    end_char=end_char,
                    n_tokens=len(window),
                )
            )
            ordinal += 1
            if start_idx + self.chunk_tokens >= n:  # this window reached the end
                break
            start_idx += step
        return chunks


def chunk_document(
    doc_id: str,
    file_path: str,
    text: str,
    *,
    chunk_tokens: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Convenience wrapper building a :class:`Chunker` for one document."""
    return Chunker(chunk_tokens=chunk_tokens, overlap=overlap).split(
        doc_id=doc_id, file_path=file_path, text=text
    )
