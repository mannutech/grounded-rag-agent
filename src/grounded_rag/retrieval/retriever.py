"""Orchestrating retrievers + ingestion.

Two retrievers behind one ``Retriever`` Protocol:

* :class:`DenseRetriever` — embed the query, cosine top-k from the vector store.
* :class:`HybridRetriever` — run dense and BM25 independently, fuse their
  rankings (RRF by default), then optionally rerank.

Rerank is a *post-stage toggle decoupled from mode*: when enabled, the first
stage produces ``top_k`` candidates and the reranker (Cohere ``rerank-v3.5``)
cuts to ``rerank_top_n`` over the chunks' **raw** text; when disabled, the final
cut is taken straight off the dense/fused list. This keeps the three evaluation
axes — {dense, hybrid} x {rerank on/off} x {path on/off} — fully independent.

Ingestion (``build_index``) is where the path-in-embedding toggle is applied, so
it is an index-time decision: the evaluation builds two indexes (path-on/off).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.errors import RetrievalError
from grounded_rag.core.types import (
    Chunk,
    Embedder,
    FusionMethod,
    Reranker,
    RetrievalMode,
    RetrievalResult,
    Retriever,
    ScoredChunk,
)
from grounded_rag.retrieval.bm25 import BM25Index
from grounded_rag.retrieval.embedder_iface import embed_text_for_chunk
from grounded_rag.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from grounded_rag.retrieval.store import InMemoryVectorStore, VectorStore


def make_vector_store(kind: str, *, dim: int) -> VectorStore:
    """Build a vector store backend by name.

    ``"memory"`` is the zero-dependency default. ``"chroma"`` lazily imports the
    optional adapter (requires the ``chroma`` extra). ``"pgvector"`` is reserved.
    """
    if kind == "memory":
        return InMemoryVectorStore(dim=dim)
    if kind == "chroma":
        from grounded_rag.retrieval.store_chroma import ChromaVectorStore  # lazy

        return ChromaVectorStore(dim=dim)
    if kind == "pgvector":
        raise RetrievalError("pgvector backend is not implemented yet")
    raise RetrievalError(f"unknown vector_store backend: {kind!r}")


@dataclass(frozen=True)
class RetrievalIndex:
    """A built index: dense store + BM25 + the ordered corpus they share."""

    store: VectorStore
    bm25: BM25Index
    corpus: list[Chunk]
    embed_file_path: bool


def build_index(chunks: list[Chunk], embedder: Embedder, config: RetrievalConfig) -> RetrievalIndex:
    """Embed and index ``chunks`` under ``config`` (applies the path toggle once).

    The same ordered ``chunks`` feed both the dense store and the BM25 index so
    their indices stay aligned for hybrid fusion. In ``sparse`` (BM25-only) mode the
    dense store is left empty and no embeddings are computed — so retrieval can be
    evaluated with no embedding model or API at all.
    """
    store = make_vector_store(config.vector_store, dim=embedder.dim)
    if config.mode is not RetrievalMode.SPARSE:
        texts = [embed_text_for_chunk(c, embed_file_path=config.embed_file_path) for c in chunks]
        store.add(chunks, embedder.embed_documents(texts))
    bm25 = BM25Index()
    bm25.fit(chunks)
    return RetrievalIndex(
        store=store, bm25=bm25, corpus=list(chunks), embed_file_path=config.embed_file_path
    )


def _rerank(
    reranker: Reranker | None,
    config: RetrievalConfig,
    query: str,
    candidates: list[ScoredChunk],
) -> tuple[list[ScoredChunk], bool]:
    """Apply rerank if enabled, else take the top ``rerank_top_n`` candidates."""
    if not (config.use_reranker and reranker is not None):
        return candidates[: config.rerank_top_n], False
    # Rerank over the RAW chunk text (never the path-prepended embedding text).
    pairs = reranker.rerank(
        query=query, documents=[c.chunk.text for c in candidates], top_n=config.rerank_top_n
    )
    reranked = [
        ScoredChunk(chunk=candidates[idx].chunk, score=score, rank=rank, stage="rerank")
        for rank, (idx, score) in enumerate(pairs)
    ]
    return reranked, True


class DenseRetriever:
    """Dense-only retrieval: cosine top-k, optionally reranked."""

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        store: VectorStore,
        embedder: Embedder,
        embed_file_path: bool,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.embed_file_path = embed_file_path
        self.reranker = reranker

    def retrieve(self, query: str) -> RetrievalResult:
        query_vec = self.embedder.embed_query(query)
        candidates = self.store.query(query_vec, self.config.top_k)
        results, reranked = _rerank(self.reranker, self.config, query, candidates)
        return RetrievalResult(
            query=query,
            results=results,
            mode=RetrievalMode.DENSE,
            reranked=reranked,
            embed_file_path=self.embed_file_path,
        )


class HybridRetriever:
    """Dense + BM25 fused (RRF/weighted), optionally reranked."""

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        store: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        corpus: list[Chunk],
        embed_file_path: bool,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.corpus = corpus
        self.embed_file_path = embed_file_path
        self.reranker = reranker
        self._corpus_index = {chunk.chunk_id: i for i, chunk in enumerate(corpus)}

    def _bm25_ranking(self, query: str) -> tuple[list[int], np.ndarray]:
        scores = self.bm25.scores(query)
        if scores.shape[0] == 0:
            return [], scores
        order = np.lexsort((np.arange(scores.shape[0]), -scores))
        return [int(i) for i in order[: self.config.top_k]], scores

    def retrieve(self, query: str) -> RetrievalResult:
        query_vec = self.embedder.embed_query(query)
        dense_results = self.store.query(query_vec, self.config.top_k)
        dense_ranking = [self._corpus_index[r.chunk.chunk_id] for r in dense_results]
        bm25_ranking, bm25_scores = self._bm25_ranking(query)

        if self.config.fusion is FusionMethod.RRF:
            fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=self.config.rrf_k)
        else:
            dense_pairs = [(self._corpus_index[r.chunk.chunk_id], r.score) for r in dense_results]
            sparse_pairs = [(i, float(bm25_scores[i])) for i in bm25_ranking]
            fused = weighted_score_fusion(
                dense_pairs, sparse_pairs, dense_weight=self.config.dense_weight
            )

        candidates = [
            ScoredChunk(chunk=self.corpus[idx], score=score, rank=rank, stage="fused")
            for rank, (idx, score) in enumerate(fused[: self.config.top_k])
        ]
        results, reranked = _rerank(self.reranker, self.config, query, candidates)
        return RetrievalResult(
            query=query,
            results=results,
            mode=RetrievalMode.HYBRID,
            reranked=reranked,
            embed_file_path=self.embed_file_path,
        )


class SparseRetriever:
    """BM25-only retrieval (no embeddings), optionally reranked."""

    def __init__(
        self,
        *,
        config: RetrievalConfig,
        bm25: BM25Index,
        corpus: list[Chunk],
        embed_file_path: bool,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.bm25 = bm25
        self.corpus = corpus
        self.embed_file_path = embed_file_path
        self.reranker = reranker

    def retrieve(self, query: str) -> RetrievalResult:
        scores = self.bm25.scores(query)
        candidates: list[ScoredChunk] = []
        if scores.shape[0] > 0:
            order = np.lexsort((np.arange(scores.shape[0]), -scores))
            candidates = [
                ScoredChunk(
                    chunk=self.corpus[int(i)], score=float(scores[int(i)]), rank=rank, stage="bm25"
                )
                for rank, i in enumerate(order[: self.config.top_k])
            ]
        results, reranked = _rerank(self.reranker, self.config, query, candidates)
        return RetrievalResult(
            query=query,
            results=results,
            mode=RetrievalMode.SPARSE,
            reranked=reranked,
            embed_file_path=self.embed_file_path,
        )


def build_retriever(
    config: RetrievalConfig,
    *,
    index: RetrievalIndex,
    embedder: Embedder,
    reranker: Reranker | None = None,
) -> Retriever:
    """Construct the retriever ``config.mode`` selects, wired to a built ``index``."""
    if config.use_reranker and reranker is None:
        raise RetrievalError("config.use_reranker is True but no reranker was provided")
    if config.mode is RetrievalMode.SPARSE:
        return SparseRetriever(
            config=config,
            bm25=index.bm25,
            corpus=index.corpus,
            embed_file_path=index.embed_file_path,
            reranker=reranker,
        )
    if config.mode is RetrievalMode.DENSE:
        return DenseRetriever(
            config=config,
            store=index.store,
            embedder=embedder,
            embed_file_path=index.embed_file_path,
            reranker=reranker,
        )
    return HybridRetriever(
        config=config,
        store=index.store,
        bm25=index.bm25,
        embedder=embedder,
        corpus=index.corpus,
        embed_file_path=index.embed_file_path,
        reranker=reranker,
    )
