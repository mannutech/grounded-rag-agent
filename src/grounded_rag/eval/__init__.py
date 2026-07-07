"""Evaluation: gold set, LLM-as-judge, metrics, runner, comparison, report.

The centerpiece. Depends on ``core``, ``retrieval``, and ``agent``. This commit
lands the gold-set loader and the metric functions; the judge, runner, report,
and comparison matrix follow.
"""

from __future__ import annotations

from grounded_rag.eval.comparison import (
    DEFAULT_VARIANTS,
    comparison_to_markdown,
    run_comparison,
)
from grounded_rag.eval.judge import (
    JUDGE_RUBRIC,
    build_judge_prompt,
    judge,
    parse_judge_response,
)
from grounded_rag.eval.metrics import (
    CostStats,
    LatencyStats,
    ToolEfficiency,
    aggregate_cost,
    aggregate_recall,
    correctness_from_verdicts,
    estimate_cost,
    groundedness_score,
    latency_percentiles,
    recall_at_k,
    reciprocal_rank,
    tool_call_efficiency,
)
from grounded_rag.eval.report import (
    REPORT_SCHEMA_VERSION,
    report_to_markdown,
    write_report,
)
from grounded_rag.eval.retrieval import (
    RetrievalReport,
    evaluate_retrieval,
    retrieval_report_to_markdown,
)
from grounded_rag.eval.runner import run_eval, run_single_query
from grounded_rag.eval.schema import load_gold

__all__ = [
    "load_gold",
    # judge
    "judge",
    "build_judge_prompt",
    "parse_judge_response",
    "JUDGE_RUBRIC",
    # runner + report
    "run_eval",
    "run_single_query",
    "write_report",
    "report_to_markdown",
    "REPORT_SCHEMA_VERSION",
    # comparison
    "run_comparison",
    "comparison_to_markdown",
    "DEFAULT_VARIANTS",
    # retrieval-only eval
    "evaluate_retrieval",
    "retrieval_report_to_markdown",
    "RetrievalReport",
    "reciprocal_rank",
    "recall_at_k",
    "aggregate_recall",
    "correctness_from_verdicts",
    "groundedness_score",
    "latency_percentiles",
    "tool_call_efficiency",
    "aggregate_cost",
    "estimate_cost",
    "LatencyStats",
    "ToolEfficiency",
    "CostStats",
]
