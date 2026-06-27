"""Evaluation metrics — pure functions over per-query results.

Every metric is computed from raw per-query facts so ``report.json`` stays
debuggable (you can see exactly which gold id moved a number). Conventions that
must hold identically everywhere:

* ``recall_at_k`` returns ``None`` (N/A) when there is nothing relevant to find
  (open-ended or must_refuse cases); aggregates exclude ``None`` from the mean.
* cost uses the single :func:`grounded_rag.core.pricing.estimate_cost`.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from grounded_rag.core.config import CoherePricing
from grounded_rag.core.pricing import estimate_cost
from grounded_rag.core.types import AgentResult, TokenUsage, Verdict

__all__ = [
    "LatencyStats",
    "ToolEfficiency",
    "CostStats",
    "recall_at_k",
    "aggregate_recall",
    "correctness_from_verdicts",
    "groundedness_score",
    "latency_percentiles",
    "tool_call_efficiency",
    "estimate_cost",
    "aggregate_cost",
]


class LatencyStats(BaseModel, frozen=True):
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float


class ToolEfficiency(BaseModel, frozen=True):
    avg_calls: float
    failure_rate: float
    redundant_rate: float
    total_calls: int


class CostStats(BaseModel, frozen=True):
    total_usd: float
    per_query_usd: float
    total_tokens: int


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float | None:
    """Fraction of relevant doc ids present in the top-``k`` retrieved doc ids.

    Returns ``None`` (N/A) when ``relevant`` is empty.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    top_k = set(retrieved[:k])
    return len(relevant_set & top_k) / len(relevant_set)


def aggregate_recall(values: list[float | None]) -> float | None:
    """Mean recall over queries, excluding N/A (``None``) entries."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def correctness_from_verdicts(verdicts: list[Verdict]) -> float:
    """Fraction of verdicts judged correct."""
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.correct) / len(verdicts)


def groundedness_score(verdicts: list[Verdict]) -> float:
    """Fraction of verdicts judged grounded."""
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.grounded) / len(verdicts)


def latency_percentiles(latencies_ms: list[float]) -> LatencyStats:
    """p50 / p95 (linear interpolation) plus mean and max."""
    if not latencies_ms:
        return LatencyStats(p50_ms=0.0, p95_ms=0.0, mean_ms=0.0, max_ms=0.0)
    arr = np.asarray(latencies_ms, dtype=float)
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        mean_ms=float(arr.mean()),
        max_ms=float(arr.max()),
    )


def tool_call_efficiency(results: list[AgentResult]) -> ToolEfficiency:
    """Average tool calls per query, failure rate, and redundant-retrieval rate.

    A ``search_docs`` call is "redundant" when every chunk it returns was already
    surfaced by an earlier call in the same query.
    """
    n = len(results)
    total_calls = failed = redundant = redundant_eligible = 0
    for result in results:
        seen_ids: set[str] = set()
        for call in result.tool_calls:
            total_calls += 1
            if not call.ok:
                failed += 1
            if call.name == "search_docs" and call.result_chunk_ids:
                redundant_eligible += 1
                ids = set(call.result_chunk_ids)
                if ids <= seen_ids:
                    redundant += 1
                seen_ids |= ids
    return ToolEfficiency(
        avg_calls=total_calls / n if n else 0.0,
        failure_rate=failed / total_calls if total_calls else 0.0,
        redundant_rate=redundant / redundant_eligible if redundant_eligible else 0.0,
        total_calls=total_calls,
    )


def aggregate_cost(usages: list[TokenUsage], pricing: CoherePricing) -> CostStats:
    """Total / per-query USD and total chat tokens over a run."""
    n = len(usages)
    total_usd = sum(estimate_cost(u, pricing) for u in usages)
    total_tokens = sum(u.total for u in usages)
    return CostStats(
        total_usd=total_usd,
        per_query_usd=total_usd / n if n else 0.0,
        total_tokens=total_tokens,
    )
