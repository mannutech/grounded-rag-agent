"""Comparison matrix — the whole point: decisions settled by measurement.

Runs the same gold set under several retrieval variants (rerank on/off,
path-in-embedding on/off, hybrid vs dense) and tabulates the headline metrics so
the README can quote a real A/B table rather than asserting "rerank helps".

Each variant gets an isolated ``Settings`` (via ``variant_to_overrides``) and an
agent built for that variant by ``agent_factory`` — the same gold set, different
retrieval configuration.
"""

from __future__ import annotations

from collections.abc import Callable

from grounded_rag.core.config import Settings, variant_to_overrides
from grounded_rag.core.types import (
    Agent,
    CohereClient,
    ComparisonMatrix,
    ComparisonRow,
    RetrievalMode,
    RunVariant,
)
from grounded_rag.eval.report import REPORT_SCHEMA_VERSION
from grounded_rag.eval.runner import run_eval

#: The four headline comparisons the brief calls for.
DEFAULT_VARIANTS: list[RunVariant] = [
    RunVariant(
        name="baseline (hybrid+rerank+path)",
        rerank=True,
        path_embedding=True,
        retrieval_mode=RetrievalMode.HYBRID,
    ),
    RunVariant(
        name="no-rerank", rerank=False, path_embedding=True, retrieval_mode=RetrievalMode.HYBRID
    ),
    RunVariant(
        name="no-path-embedding",
        rerank=True,
        path_embedding=False,
        retrieval_mode=RetrievalMode.HYBRID,
    ),
    RunVariant(
        name="dense-only", rerank=True, path_embedding=True, retrieval_mode=RetrievalMode.DENSE
    ),
]

AgentFactory = Callable[[RunVariant], Agent]


def run_comparison(
    settings: Settings,
    variants: list[RunVariant],
    agent_factory: AgentFactory,
    client: CohereClient,
) -> ComparisonMatrix:
    """Evaluate every variant and collect their headline metrics into a matrix."""
    rows: list[ComparisonRow] = []
    for variant in variants:
        variant_settings = variant_to_overrides(variant, settings)
        report = run_eval(variant_settings, agent_factory(variant), client)
        summary = report.summary
        rows.append(
            ComparisonRow(
                variant=variant.name,
                correctness=summary.correctness,
                groundedness=summary.groundedness,
                recall_at_5=summary.recall_at_k.get(5),
                p95_latency_ms=summary.p95_latency_ms,
                cost_per_query_usd=summary.cost_per_query_usd,
            )
        )
    return ComparisonMatrix(variants=variants, rows=rows, report_version=REPORT_SCHEMA_VERSION)


def comparison_to_markdown(matrix: ComparisonMatrix) -> str:
    """Render the comparison grid as the small Markdown table the README quotes."""
    lines = [
        "| variant | correctness | groundedness | recall@5 | p95 ms | cost/query |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for row in matrix.rows:
        recall = "N/A" if row.recall_at_5 is None else f"{row.recall_at_5:.3f}"
        lines.append(
            f"| {row.variant} | {row.correctness:.3f} | {row.groundedness:.3f} | "
            f"{recall} | {row.p95_latency_ms:.1f} | ${row.cost_per_query_usd:.4f} |"
        )
    return "\n".join(lines) + "\n"
