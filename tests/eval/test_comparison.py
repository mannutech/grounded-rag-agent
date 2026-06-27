"""Comparison matrix: variant wiring + table rendering."""

from __future__ import annotations

import json
from pathlib import Path

from grounded_rag.core.clients.mock import MockCohereClient, chat_text
from grounded_rag.core.config import EvalConfig, JudgeConfig, Settings
from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    ComparisonMatrix,
    ComparisonRow,
    RetrievedChunk,
    RunVariant,
    TokenUsage,
)
from grounded_rag.eval.comparison import comparison_to_markdown, run_comparison


def _result(doc_id: str) -> AgentResult:
    chunk = RetrievedChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}::0",
        text="t",
        source="s",
        score=0.9,
        rank=0,
        stage="rerank",
    )
    trace = AgentRunTrace(
        query="q",
        model="m",
        steps=1,
        timings=[],
        tool_calls=[],
        injection_flagged=False,
        seed=None,
        temperature=0.0,
        total_duration_ms=10.0,
    )
    return AgentResult(
        answer="a",
        refused=False,
        citations=[],
        retrieved=[chunk],
        tool_calls=[],
        usage=TokenUsage(),
        cost_usd=0.0,
        steps=1,
        trace=trace,
    )


class _FakeAgent:
    def __init__(self, result: AgentResult) -> None:
        self._result = result

    def answer(self, question: str) -> AgentResult:
        return self._result


def _correct_ballot_client() -> MockCohereClient:
    ballot = json.dumps(
        {
            "correct": True,
            "grounded": True,
            "refusal_appropriate": None,
            "keypoints_hit": ["k"],
            "score": 0.9,
            "rationale": "r",
        }
    )
    return MockCohereClient(chat_router=lambda messages, tools: chat_text(ballot))


def test_run_comparison_rerank_helps_recall(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        '{"id":"n1","question":"q about retries","type":"normal","expected_answer":"a",'
        '"keypoints":["k"],"relevant_doc_ids":["d1"]}\n',
        encoding="utf-8",
    )

    def make_agent(variant: RunVariant) -> _FakeAgent:
        # rerank-on surfaces the relevant doc; rerank-off surfaces a wrong one.
        return _FakeAgent(_result("d1" if variant.rerank else "d9"))

    settings = Settings(
        eval=EvalConfig(gold_path=str(gold), k_values=[1, 5]), judge=JudgeConfig(n_votes=1)
    )
    variants = [
        RunVariant(name="rerank-on", rerank=True),
        RunVariant(name="rerank-off", rerank=False),
    ]

    matrix = run_comparison(settings, variants, make_agent, _correct_ballot_client())
    rows = {r.variant: r for r in matrix.rows}
    assert rows["rerank-on"].recall_at_5 == 1.0
    assert rows["rerank-off"].recall_at_5 == 0.0
    assert rows["rerank-on"].recall_at_5 > rows["rerank-off"].recall_at_5


def test_comparison_to_markdown_one_row_per_variant() -> None:
    matrix = ComparisonMatrix(
        variants=[RunVariant(name="a"), RunVariant(name="b")],
        rows=[
            ComparisonRow(
                variant="a",
                correctness=1.0,
                groundedness=1.0,
                recall_at_5=0.8,
                p95_latency_ms=120.0,
                cost_per_query_usd=0.001,
            ),
            ComparisonRow(
                variant="b",
                correctness=0.5,
                groundedness=0.5,
                recall_at_5=None,
                p95_latency_ms=90.0,
                cost_per_query_usd=0.002,
            ),
        ],
        report_version="1",
    )
    md = comparison_to_markdown(matrix)
    assert "| variant |" in md
    assert "| a |" in md
    assert "| b |" in md
    assert "N/A" in md  # b's recall@5 is None
