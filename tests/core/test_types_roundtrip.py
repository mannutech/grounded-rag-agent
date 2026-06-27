"""Core types: serialisation round-trips, invariants, and helpers.

These run with no network and no client. They lock the canonical shapes so that
later subsystems (and report.json) cannot silently drift.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grounded_rag.core import git_sha
from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    Chunk,
    Citation,
    GoldRecord,
    JudgeBallot,
    QueryResult,
    QueryType,
    Report,
    ReportSummary,
    RetrievalMode,
    RetrievedChunk,
    ScoredChunk,
    StageTiming,
    TokenUsage,
    ToolCallRecord,
    Verdict,
    cited_doc_ids,
)


def _retrieved(doc_id: str = "d1", chunk_id: str = "d1::0") -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id, chunk_id=chunk_id, text="hello", source="docs/a.md", score=0.9, rank=0
    )


def _citation(doc_ids: list[str], chunk_ids: list[str]) -> Citation:
    return Citation(
        text="hello", start=0, end=5, chunk_ids=chunk_ids, doc_ids=doc_ids, sources=["docs/a.md"]
    )


def _agent_result(citations: list[Citation] | None = None) -> AgentResult:
    trace = AgentRunTrace(
        query="q",
        model="command-a-03-2025",
        steps=2,
        timings=[StageTiming(stage="retrieval", duration_ms=12.0)],
        tool_calls=[ToolCallRecord(step=1, name="search_docs", arguments={"query": "q"}, ok=True)],
        injection_flagged=False,
        seed=42,
        temperature=0.0,
        total_duration_ms=120.0,
    )
    return AgentResult(
        answer="hello",
        refused=False,
        citations=citations if citations is not None else [_citation(["d1"], ["d1::0"])],
        retrieved=[_retrieved()],
        tool_calls=trace.tool_calls,
        usage=TokenUsage(input_tokens=100, output_tokens=20, embed_tokens=8, rerank_docs=5),
        cost_usd=0.0012,
        steps=2,
        trace=trace,
    )


def test_token_usage_total() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=23)
    assert usage.total == 123
    assert TokenUsage().total == 0


@pytest.mark.parametrize(
    "model",
    [
        _retrieved(),
        _citation(["d1", "d2"], ["d1::0", "d2::1"]),
        ScoredChunk(
            chunk=Chunk(
                chunk_id="d1::0",
                doc_id="d1",
                file_path="docs/a.md",
                ordinal=0,
                text="hello",
                start_char=0,
                end_char=5,
                n_tokens=1,
            ),
            score=0.5,
            rank=0,
            stage="dense",
        ),
        _agent_result(),
    ],
)
def test_model_round_trips(model: object) -> None:
    """model_validate(model_dump()) reconstructs an equal object."""
    cls = type(model)
    dumped = model.model_dump()  # type: ignore[attr-defined]
    rebuilt = cls.model_validate(dumped)  # type: ignore[attr-defined]
    assert rebuilt == model


def test_nested_report_round_trips() -> None:
    verdict = Verdict(
        correct=True,
        grounded=True,
        refusal_appropriate=None,
        score=0.9,
        keypoint_recall=1.0,
        n_votes=3,
        agreement=1.0,
        ballots=[
            JudgeBallot(
                correct=True,
                grounded=True,
                refusal_appropriate=None,
                keypoints_hit=["a"],
                score=0.9,
                rationale="ok",
            )
        ],
    )
    qr = QueryResult(
        record_id="g1",
        type=QueryType.NORMAL,
        verdict=verdict,
        recall_at_k={1: 1.0, 5: 1.0, 10: None},  # None == N/A is representable
        result=_agent_result(),
        must_refuse=False,
        refusal_correct=None,
        judge_usage=TokenUsage(input_tokens=50, output_tokens=10),
    )
    report = Report(
        schema_version="1",
        git_sha="abc1234",
        timestamp="2026-06-28T00:00:00Z",
        seed=7,
        mock_mode=True,
        config={"mock_mode": True},
        summary=ReportSummary(
            n=1,
            correctness=1.0,
            groundedness=1.0,
            recall_at_k={1: 1.0, 5: 1.0, 10: None},
            refusal_accuracy=1.0,
            p50_latency_ms=120.0,
            p95_latency_ms=120.0,
            total_cost_usd=0.0012,
            cost_per_query_usd=0.0012,
        ),
        per_query=[qr],
    )
    rebuilt = Report.model_validate(report.model_dump())
    assert rebuilt == report
    assert rebuilt.per_query[0].recall_at_k[10] is None


def test_frozen_models_are_immutable() -> None:
    chunk = _retrieved()
    with pytest.raises(ValidationError):
        chunk.score = 0.1  # type: ignore[misc]


def test_cited_doc_ids_dedups_and_preserves_order() -> None:
    result = _agent_result(
        citations=[
            _citation(["d2"], ["d2::0"]),
            _citation(["d1", "d2"], ["d1::0", "d2::1"]),
            _citation(["d1"], ["d1::3"]),
        ]
    )
    assert cited_doc_ids(result) == ["d2", "d1"]


def test_gold_record_defaults() -> None:
    rec = GoldRecord(id="g1", question="q?", type=QueryType.MUST_REFUSE, must_refuse=True)
    assert rec.keypoints == []
    assert rec.relevant_doc_ids == []
    assert rec.expected_answer is None


def test_enums_have_expected_values() -> None:
    assert QueryType.MUST_REFUSE.value == "must_refuse"
    assert RetrievalMode.HYBRID.value == "hybrid"
    assert {t.value for t in QueryType} == {"normal", "adversarial", "multi_hop", "must_refuse"}


def test_git_sha_is_a_string() -> None:
    sha = git_sha()
    assert isinstance(sha, str)
    assert sha  # non-empty (either a real sha or "unknown")
