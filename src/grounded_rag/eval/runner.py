"""The evaluation runner: gold set -> per-query results -> a versioned Report.

Each query runs the agent, then the judge (via a usage-counting proxy so judge
tokens are tracked *separately* and never inflate the agent's cost), then computes
per-query recall and refusal correctness. The aggregate is a self-describing
:class:`Report`. ``run_eval`` runs entirely offline against the mock client, which
is what makes CI free, fast, and deterministic.
"""

from __future__ import annotations

import datetime
from typing import Any

from grounded_rag.core.config import Settings
from grounded_rag.core.gitmeta import git_sha
from grounded_rag.core.types import (
    Agent,
    CohereClient,
    GoldRecord,
    QueryResult,
    Report,
    ReportSummary,
    TokenUsage,
)
from grounded_rag.eval.judge import judge
from grounded_rag.eval.metrics import (
    aggregate_cost,
    aggregate_recall,
    correctness_from_verdicts,
    groundedness_score,
    latency_percentiles,
    recall_at_k,
)
from grounded_rag.eval.report import REPORT_SCHEMA_VERSION
from grounded_rag.eval.schema import load_gold


def _safe_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


class _UsageCountingClient:
    """Wraps a client, summing chat token usage (used to isolate judge cost)."""

    def __init__(self, inner: CohereClient) -> None:
        self._inner = inner
        self.is_mock = getattr(inner, "is_mock", False)
        self._input = 0
        self._output = 0

    def chat(self, **kwargs: Any) -> Any:
        resp = self._inner.chat(**kwargs)
        self._input += _safe_int(getattr(resp.usage.tokens, "input_tokens", 0))
        self._output += _safe_int(getattr(resp.usage.tokens, "output_tokens", 0))
        return resp

    def embed(self, **kwargs: Any) -> Any:
        return self._inner.embed(**kwargs)

    def rerank(self, **kwargs: Any) -> Any:
        return self._inner.rerank(**kwargs)

    def usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=self._input, output_tokens=self._output)


def run_single_query(
    record: GoldRecord, agent: Agent, client: CohereClient, settings: Settings
) -> QueryResult:
    """Run the agent and judge on one gold record."""
    result = agent.answer(record.question)

    counting = _UsageCountingClient(client)
    judge_model = settings.judge.model_override or settings.cohere.generation_model
    verdict = judge(
        record,
        result,
        client=counting,
        model=judge_model,
        n_votes=settings.judge.n_votes,
        temperature=settings.judge.temperature,
        seed=settings.judge.seed,
        response_format_json=settings.judge.response_format_json,
    )

    retrieved_doc_ids = [chunk.doc_id for chunk in result.retrieved]
    recall = {
        k: recall_at_k(record.relevant_doc_ids, retrieved_doc_ids, k)
        for k in settings.eval.k_values
    }
    # Refusal correctness is defined for every query: a must_refuse case should
    # refuse, every other case should not. This captures both under- and over-refusal.
    refusal_correct = result.refused == record.must_refuse

    return QueryResult(
        record_id=record.id,
        type=record.type,
        verdict=verdict,
        recall_at_k=recall,
        result=result,
        must_refuse=record.must_refuse,
        refusal_correct=refusal_correct,
        judge_usage=counting.usage(),
    )


def _summarise(per_query: list[QueryResult], settings: Settings) -> ReportSummary:
    verdicts = [q.verdict for q in per_query]
    n = len(per_query)
    recall = {
        k: aggregate_recall([q.recall_at_k.get(k) for q in per_query])
        for k in settings.eval.k_values
    }
    refusal_accuracy = sum(1.0 for q in per_query if q.refusal_correct) / n if n else 1.0
    latency = latency_percentiles([q.result.trace.total_duration_ms for q in per_query])
    cost = aggregate_cost([q.result.usage for q in per_query], settings.cohere.pricing)
    return ReportSummary(
        n=n,
        correctness=correctness_from_verdicts(verdicts),
        groundedness=groundedness_score(verdicts),
        recall_at_k=recall,
        refusal_accuracy=refusal_accuracy,
        p50_latency_ms=latency.p50_ms,
        p95_latency_ms=latency.p95_ms,
        total_cost_usd=cost.total_usd,
        cost_per_query_usd=cost.per_query_usd,
    )


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def run_eval(settings: Settings, agent: Agent, client: CohereClient) -> Report:
    """Run the whole gold set and assemble a versioned :class:`Report`."""
    gold = load_gold(settings.eval.gold_path)
    per_query = [run_single_query(record, agent, client, settings) for record in gold]
    return Report(
        schema_version=REPORT_SCHEMA_VERSION,
        git_sha=git_sha(),
        timestamp=_now_iso(),
        seed=settings.agent.seed,
        mock_mode=getattr(client, "is_mock", False),
        config=settings.model_dump(mode="json"),
        summary=_summarise(per_query, settings),
        per_query=per_query,
    )
