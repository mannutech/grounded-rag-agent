"""RetrieverAdapter flattens ScoredChunk -> RetrievedChunk and surfaces rerank score."""

from __future__ import annotations

from grounded_rag.agent.retriever_adapter import RetrieverAdapter
from grounded_rag.core.types import Chunk, RetrievalMode, RetrievalResult, ScoredChunk


def _scored(idx: int, score: float, stage: str = "rerank") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"d::{idx}",
        doc_id="d",
        file_path="docs/d.md",
        ordinal=idx,
        text=f"chunk {idx}",
        start_char=0,
        end_char=7,
        n_tokens=2,
    )
    return ScoredChunk(chunk=chunk, score=score, rank=idx, stage=stage)


class _FakeCoreRetriever:
    def __init__(self, scored: list[ScoredChunk]) -> None:
        self._scored = scored

    def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            results=self._scored,
            mode=RetrievalMode.HYBRID,
            reranked=True,
            embed_file_path=True,
        )


def test_adapter_maps_fields_and_surfaces_score() -> None:
    core = _FakeCoreRetriever([_scored(0, 0.91), _scored(1, 0.42)])
    adapter = RetrieverAdapter(core)
    out = adapter.retrieve("q")
    assert [c.chunk_id for c in out] == ["d::0", "d::1"]
    assert [c.score for c in out] == [0.91, 0.42]  # rerank score surfaced for the refusal gate
    assert out[0].source == "docs/d.md"
    assert out[0].doc_id == "d"
    assert out[0].stage == "rerank"


def test_adapter_top_k_narrows() -> None:
    core = _FakeCoreRetriever([_scored(i, 1.0 - i / 10) for i in range(5)])
    adapter = RetrieverAdapter(core)
    assert len(adapter.retrieve("q", top_k=2)) == 2
    assert len(adapter.retrieve("q")) == 5
