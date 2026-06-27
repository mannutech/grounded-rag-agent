"""HybridRetriever: BM25 demonstrably lifts a lexically-matched chunk that dense buries."""

from __future__ import annotations

from grounded_rag.core.config import RetrievalConfig
from grounded_rag.core.types import Chunk, FusionMethod, RetrievalMode
from grounded_rag.retrieval.retriever import build_index, build_retriever

# Three chunks. c1 is the only one containing the rare query term "zorptron", but
# its dense vector is orthogonal to the query, so pure dense ranks it LAST.
_C0 = "alpha alpha distractor content here"
_C1 = "the special zorptron mechanism explained"
_C2 = "beta gamma neutral filler text"

_VECTORS = {
    "zorptron": [1.0, 0.0, 0.0],  # the query vector
    _C0: [1.0, 0.0, 0.0],  # dense-aligned distractor -> dense rank 0
    _C2: [0.5, 0.5, 0.0],  # partial -> dense rank 1
    _C1: [0.0, 0.0, 1.0],  # orthogonal -> dense rank 2 (last)
}


def _chunks() -> list[Chunk]:
    texts = [_C0, _C1, _C2]
    return [
        Chunk(
            chunk_id=f"d::{i}",
            doc_id="d",
            file_path="docs/d.md",
            ordinal=i,
            text=t,
            start_char=0,
            end_char=len(t),
            n_tokens=len(t.split()),
        )
        for i, t in enumerate(texts)
    ]


def _config(mode: RetrievalMode) -> RetrievalConfig:
    return RetrievalConfig(
        mode=mode,
        use_reranker=False,
        embed_file_path=False,  # StaticEmbedder keys on raw chunk text
        top_k=10,
        rerank_top_n=10,
        fusion=FusionMethod.RRF,
        rrf_k=60,
    )


def test_hybrid_lifts_lexical_match_that_dense_buries(make_static_embedder) -> None:  # type: ignore[no-untyped-def]
    embedder = make_static_embedder(3, _VECTORS)
    chunks = _chunks()

    dense_cfg = _config(RetrievalMode.DENSE)
    index = build_index(chunks, embedder, dense_cfg)
    dense_ids = [
        sc.chunk.chunk_id
        for sc in build_retriever(dense_cfg, index=index, embedder=embedder)
        .retrieve("zorptron")
        .results
    ]

    hybrid_cfg = _config(RetrievalMode.HYBRID)
    hybrid_result = build_retriever(hybrid_cfg, index=index, embedder=embedder).retrieve("zorptron")
    hybrid_ids = [sc.chunk.chunk_id for sc in hybrid_result.results]

    assert dense_ids[-1] == "d::1"  # pure dense buries the lexical match
    assert hybrid_ids.index("d::1") < dense_ids.index("d::1")  # hybrid lifts it
    assert hybrid_result.mode is RetrievalMode.HYBRID
    assert all(sc.stage == "fused" for sc in hybrid_result.results)
