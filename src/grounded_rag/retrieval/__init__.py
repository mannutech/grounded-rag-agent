"""Retrieval: chunking, dense + BM25 + hybrid retrieval, pluggable vector store.

Depends only on ``grounded_rag.core``. Public surface: the pure building blocks
(chunker, cosine/BM25/fusion, in-memory store) plus the orchestrating retrievers
and the ``build_index`` / ``build_retriever`` ingestion seam.
"""

from __future__ import annotations

from grounded_rag.retrieval.bm25 import BM25Index
from grounded_rag.retrieval.chunker import Chunker, chunk_document
from grounded_rag.retrieval.embedder_iface import embed_text_for_chunk
from grounded_rag.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from grounded_rag.retrieval.retriever import (
    DenseRetriever,
    HybridRetriever,
    RetrievalIndex,
    build_index,
    build_retriever,
    make_vector_store,
)
from grounded_rag.retrieval.store import (
    InMemoryVectorStore,
    VectorStore,
    cosine_similarity,
    cosine_top_k,
)
from grounded_rag.retrieval.types import (
    Chunk,
    FusionMethod,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)

__all__ = [
    # types
    "Chunk",
    "ScoredChunk",
    "RetrievalResult",
    "RetrievalMode",
    "FusionMethod",
    # chunking
    "Chunker",
    "chunk_document",
    # embedding seam
    "embed_text_for_chunk",
    # dense store + cosine math
    "VectorStore",
    "InMemoryVectorStore",
    "cosine_similarity",
    "cosine_top_k",
    # sparse
    "BM25Index",
    # fusion
    "reciprocal_rank_fusion",
    "weighted_score_fusion",
    # orchestration + ingestion
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalIndex",
    "build_index",
    "build_retriever",
    "make_vector_store",
]
