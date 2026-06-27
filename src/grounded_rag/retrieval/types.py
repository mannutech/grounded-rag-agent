"""Retrieval's public data types — re-exported from :mod:`grounded_rag.core`.

The definitions live in ``core`` (so the ``Retriever`` Protocol's return type is
reachable without an upward import). This module gives the retrieval subsystem a
self-contained import surface for those types.
"""

from __future__ import annotations

from grounded_rag.core.types import (
    Chunk,
    FusionMethod,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)

__all__ = ["Chunk", "ScoredChunk", "RetrievalResult", "RetrievalMode", "FusionMethod"]
