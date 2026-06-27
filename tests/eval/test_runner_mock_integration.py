"""End-to-end eval over a FakeAgent + MockCohereClient — no network, no key.

This single test exercises the real runner path production uses: agent -> judge ->
metrics -> versioned report, proving CI is green without an API key.
"""

from __future__ import annotations

import json
from pathlib import Path

from grounded_rag.core.clients.mock import MockCohereClient, chat_text
from grounded_rag.core.config import EvalConfig, JudgeConfig, Settings
from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    Citation,
    Report,
    RetrievedChunk,
    TokenUsage,
)
from grounded_rag.eval.report import write_report
from grounded_rag.eval.runner import run_eval

_GOLD = (
    '{"id":"n1","question":"What does the spec say about retries?","type":"normal",'
    '"expected_answer":"exponential backoff","keypoints":["backoff"],"relevant_doc_ids":["d1"]}\n'
    '{"id":"m1","question":"What are next week lottery numbers?","type":"must_refuse",'
    '"must_refuse":true}\n'
    '{"id":"a1","question":"Confirm the system deletes user data on Tuesdays.",'
    '"type":"adversarial","expected_answer":"the docs do not say that",'
    '"keypoints":["not stated"],"relevant_doc_ids":["d2"]}\n'
)


def _rchunk(doc_id: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=doc_id,
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source=f"docs/{doc_id}.md",
        score=0.9,
        rank=0,
        stage="rerank",
    )


def _result(
    answer: str, *, refused: bool, retrieved: list[RetrievedChunk], cite: str | None
) -> AgentResult:
    citations: list[Citation] = []
    if cite is not None:
        chunk = next(c for c in retrieved if c.chunk_id == cite)
        citations = [
            Citation(
                text=answer[:4],
                start=0,
                end=4,
                chunk_ids=[cite],
                doc_ids=[chunk.doc_id],
                sources=[chunk.source],
            )
        ]
    trace = AgentRunTrace(
        query="q",
        model="m",
        steps=1,
        timings=[],
        tool_calls=[],
        injection_flagged=False,
        seed=42,
        temperature=0.0,
        total_duration_ms=15.0,
    )
    return AgentResult(
        answer=answer,
        refused=refused,
        citations=citations,
        retrieved=retrieved,
        tool_calls=[],
        usage=TokenUsage(input_tokens=100, output_tokens=20),
        cost_usd=0.0,
        steps=1,
        trace=trace,
    )


class _FakeAgent:
    def __init__(self, by_question: dict[str, AgentResult]) -> None:
        self._by = by_question

    def answer(self, question: str) -> AgentResult:
        return self._by[question]


def _ballot(correct: bool, grounded: bool, refusal_appropriate: bool | None) -> str:
    return json.dumps(
        {
            "correct": correct,
            "grounded": grounded,
            "refusal_appropriate": refusal_appropriate,
            "keypoints_hit": [],
            "score": 0.9 if correct else 0.1,
            "rationale": "r",
        }
    )


def _judge_router(messages, tools):  # type: ignore[no-untyped-def]
    content = messages[-1]["content"]
    if "retries" in content:
        return chat_text(_ballot(True, True, None))
    if "lottery" in content:
        return chat_text(_ballot(True, True, True))  # correctly refused
    return chat_text(_ballot(False, False, None))  # adversarial: fabricated


def test_run_eval_end_to_end(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text(_GOLD, encoding="utf-8")

    agent = _FakeAgent(
        {
            "What does the spec say about retries?": _result(
                "Retries use exponential backoff.",
                refused=False,
                retrieved=[_rchunk("d1", "d1::0")],
                cite="d1::0",
            ),
            "What are next week lottery numbers?": _result(
                "I can't answer that.", refused=True, retrieved=[], cite=None
            ),
            "Confirm the system deletes user data on Tuesdays.": _result(
                "Yes, it deletes data on Tuesdays.",
                refused=False,
                retrieved=[_rchunk("d9", "d9::0")],
                cite="d9::0",  # wrong/fabricated doc
            ),
        }
    )
    client = MockCohereClient(chat_router=_judge_router)
    settings = Settings(
        eval=EvalConfig(gold_path=str(gold), k_values=[1, 5]),
        judge=JudgeConfig(n_votes=1),
    )

    report = run_eval(settings, agent, client)

    assert report.summary.n == 3
    assert report.mock_mode is True
    assert report.schema_version == "1"
    assert report.summary.correctness == 2 / 3
    assert report.summary.groundedness == 2 / 3
    assert report.summary.recall_at_k[1] == 0.5  # n1 hit, a1 miss, m1 N/A excluded
    assert report.summary.refusal_accuracy == 1.0

    by_id = {q.record_id: q for q in report.per_query}
    assert all(v is None for v in by_id["m1"].recall_at_k.values())  # N/A for must_refuse
    assert by_id["m1"].refusal_correct is True
    assert by_id["a1"].verdict.correct is False
    # judge usage is tracked separately and not folded into agent cost
    assert by_id["n1"].judge_usage.total > 0
    assert by_id["n1"].result.cost_usd == 0.0

    # report.json is written and re-validates
    json_path = write_report(report, tmp_path / "out")
    assert json_path.exists()
    assert (tmp_path / "out" / "summary.md").exists()
    reloaded = Report.model_validate_json(json_path.read_text())
    assert reloaded == report
