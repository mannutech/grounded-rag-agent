"""Evaluation metrics: exact values and the N/A recall convention."""

from __future__ import annotations

import pytest

from grounded_rag.core.config import CoherePricing
from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    TokenUsage,
    ToolCallRecord,
    Verdict,
)
from grounded_rag.eval.metrics import (
    aggregate_cost,
    aggregate_recall,
    correctness_from_verdicts,
    groundedness_score,
    latency_percentiles,
    recall_at_k,
    tool_call_efficiency,
)


def _verdict(*, correct: bool, grounded: bool) -> Verdict:
    return Verdict(
        correct=correct,
        grounded=grounded,
        refusal_appropriate=None,
        score=1.0 if correct else 0.0,
        keypoint_recall=1.0 if correct else 0.0,
        n_votes=1,
        agreement=1.0,
        ballots=[],
    )


def _tc(name: str, *, ok: bool = True, chunk_ids: tuple[str, ...] = ()) -> ToolCallRecord:
    return ToolCallRecord(step=0, name=name, arguments={}, ok=ok, result_chunk_ids=list(chunk_ids))


def _result(tool_calls: list[ToolCallRecord]) -> AgentResult:
    trace = AgentRunTrace(
        query="q",
        model="m",
        steps=1,
        timings=[],
        tool_calls=tool_calls,
        injection_flagged=False,
        seed=None,
        temperature=0.0,
        total_duration_ms=0.0,
    )
    return AgentResult(
        answer="a",
        refused=False,
        citations=[],
        retrieved=[],
        tool_calls=tool_calls,
        usage=TokenUsage(),
        cost_usd=0.0,
        steps=1,
        trace=trace,
    )


# -- recall -----------------------------------------------------------------


def test_recall_at_k_cases() -> None:
    assert recall_at_k(["d1", "d2"], ["d1", "d2", "d3"], k=3) == pytest.approx(1.0)
    assert recall_at_k(["d1", "d2"], ["d1", "d9"], k=2) == pytest.approx(0.5)
    assert recall_at_k(["d1"], ["d9", "d8"], k=2) == pytest.approx(0.0)
    assert recall_at_k(["d1"], ["d9", "d1"], k=1) == pytest.approx(0.0)  # d1 outside top-1
    assert recall_at_k([], ["d1"], k=5) is None  # N/A


def test_aggregate_recall_excludes_na() -> None:
    assert aggregate_recall([1.0, 0.0, None]) == pytest.approx(0.5)
    assert aggregate_recall([None, None]) is None


# -- verdict aggregates -----------------------------------------------------


def test_correctness_and_groundedness() -> None:
    verdicts = [
        _verdict(correct=True, grounded=True),
        _verdict(correct=False, grounded=True),
        _verdict(correct=True, grounded=False),
    ]
    assert correctness_from_verdicts(verdicts) == pytest.approx(2 / 3)
    assert groundedness_score(verdicts) == pytest.approx(2 / 3)
    assert correctness_from_verdicts([]) == 0.0


# -- latency ----------------------------------------------------------------


def test_latency_percentiles_known_array() -> None:
    stats = latency_percentiles([10, 20, 30, 40, 50])
    assert stats.p50_ms == pytest.approx(30.0)
    assert stats.p95_ms == pytest.approx(48.0)  # linear interpolation
    assert stats.mean_ms == pytest.approx(30.0)
    assert stats.max_ms == pytest.approx(50.0)


def test_latency_single_element() -> None:
    stats = latency_percentiles([42.0])
    assert stats.p50_ms == stats.p95_ms == stats.max_ms == pytest.approx(42.0)


# -- tool efficiency --------------------------------------------------------


def test_tool_call_efficiency() -> None:
    results = [
        _result([_tc("search_docs", chunk_ids=("c1", "c2")), _tc("calculator")]),
        _result([_tc("search_docs", chunk_ids=("c1",)), _tc("search_docs", chunk_ids=("c1",))]),
        _result([_tc("calculator", ok=False)]),
    ]
    eff = tool_call_efficiency(results)
    assert eff.total_calls == 5
    assert eff.avg_calls == pytest.approx(5 / 3)
    assert eff.failure_rate == pytest.approx(1 / 5)
    # 3 search_docs calls eligible; the 2nd in result-2 repeats an already-seen id.
    assert eff.redundant_rate == pytest.approx(1 / 3)


# -- cost -------------------------------------------------------------------


def test_aggregate_cost() -> None:
    pricing = CoherePricing(command_a_input_usd_per_1m=2.0, command_a_output_usd_per_1m=10.0)
    usages = [
        TokenUsage(input_tokens=1_000_000, output_tokens=0),
        TokenUsage(input_tokens=0, output_tokens=100_000),
    ]
    stats = aggregate_cost(usages, pricing)
    assert stats.total_usd == pytest.approx(3.0)
    assert stats.per_query_usd == pytest.approx(1.5)
    assert stats.total_tokens == 1_100_000
