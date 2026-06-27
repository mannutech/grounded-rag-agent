"""Report rendering + JSON round-trip."""

from __future__ import annotations

from pathlib import Path

from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    QueryResult,
    QueryType,
    Report,
    ReportSummary,
    TokenUsage,
    Verdict,
)
from grounded_rag.eval.report import REPORT_SCHEMA_VERSION, report_to_markdown, write_report


def _report() -> Report:
    verdict = Verdict(
        correct=True,
        grounded=True,
        refusal_appropriate=None,
        score=0.9,
        keypoint_recall=1.0,
        n_votes=3,
        agreement=1.0,
        ballots=[],
    )
    trace = AgentRunTrace(
        query="q",
        model="m",
        steps=1,
        timings=[],
        tool_calls=[],
        injection_flagged=False,
        seed=42,
        temperature=0.0,
        total_duration_ms=20.0,
    )
    result = AgentResult(
        answer="a",
        refused=False,
        citations=[],
        retrieved=[],
        tool_calls=[],
        usage=TokenUsage(input_tokens=10, output_tokens=2),
        cost_usd=0.0,
        steps=1,
        trace=trace,
    )
    qr = QueryResult(
        record_id="n1",
        type=QueryType.NORMAL,
        verdict=verdict,
        recall_at_k={1: 1.0, 5: None},
        result=result,
        must_refuse=False,
        refusal_correct=True,
        judge_usage=TokenUsage(),
    )
    return Report(
        schema_version=REPORT_SCHEMA_VERSION,
        git_sha="abc1234",
        timestamp="2026-06-28T00:00:00Z",
        seed=42,
        mock_mode=True,
        config={"mock_mode": True},
        summary=ReportSummary(
            n=1,
            correctness=1.0,
            groundedness=1.0,
            recall_at_k={1: 1.0, 5: None},
            refusal_accuracy=1.0,
            p50_latency_ms=20.0,
            p95_latency_ms=20.0,
            total_cost_usd=0.0,
            cost_per_query_usd=0.0,
        ),
        per_query=[qr],
    )


def test_write_report_round_trips(tmp_path: Path) -> None:
    report = _report()
    json_path = write_report(report, tmp_path)
    assert json_path.name == "report.json"
    assert (tmp_path / "summary.md").exists()
    reloaded = Report.model_validate_json(json_path.read_text())
    assert reloaded == report
    assert reloaded.summary.recall_at_k[5] is None


def test_report_to_markdown_contains_headline() -> None:
    md = report_to_markdown(_report())
    assert "# Evaluation report" in md
    assert "Headline" in md
    assert "n1" in md  # per-query row
    assert "recall@1=1.000" in md
    assert "recall@5=N/A" in md
