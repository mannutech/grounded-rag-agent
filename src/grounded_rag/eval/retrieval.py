"""Retrieval-only evaluation — deterministic IR metrics, no LLM in the loop.

This measures the *retrieval* stage directly (recall@k, MRR) against labeled
relevant documents, rather than inferring "context relevance" from an LLM judge.
It needs no generation and no judge, so it runs offline (BM25) or with a local
embedder (dense/hybrid) at zero API cost — and it's the classic-IR metric that
most eval tools replace with an LLM proxy.

Only records with ``relevant_doc_ids`` are scored (must-refuse cases have no
relevant document and are skipped).
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel

from grounded_rag.core.gitmeta import git_sha
from grounded_rag.core.types import GoldRecord, Retriever
from grounded_rag.eval.metrics import reciprocal_rank

RETRIEVAL_SCHEMA_VERSION = "1"


def _dedup(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top_k = set(retrieved[:k])
    return len(relevant & top_k) / len(relevant)


class RetrievalQueryResult(BaseModel):
    record_id: str
    relevant_doc_ids: list[str]
    retrieved_doc_ids: list[str]  # de-duplicated, rank-ordered
    recall_at_k: dict[int, float]
    reciprocal_rank: float


class RetrievalReport(BaseModel):
    schema_version: str
    git_sha: str
    timestamp: str
    mode: str
    reranked: bool
    n: int
    recall_at_k: dict[int, float]  # mean over scored queries
    mrr: float
    per_query: list[RetrievalQueryResult]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def evaluate_retrieval(
    gold: list[GoldRecord], retriever: Retriever, k_values: list[int]
) -> RetrievalReport:
    """Score a retriever against ``gold`` on recall@k + MRR (no generation/judge)."""
    per_query: list[RetrievalQueryResult] = []
    mode = "unknown"
    reranked = False

    for record in gold:
        if not record.relevant_doc_ids:
            continue  # must-refuse / open-ended: nothing to retrieve
        result = retriever.retrieve(record.question)
        mode, reranked = result.mode.value, result.reranked
        retrieved_ids = _dedup([sc.chunk.doc_id for sc in result.results])
        relevant = set(record.relevant_doc_ids)
        per_query.append(
            RetrievalQueryResult(
                record_id=record.id,
                relevant_doc_ids=record.relevant_doc_ids,
                retrieved_doc_ids=retrieved_ids,
                recall_at_k={k: _recall_at_k(relevant, retrieved_ids, k) for k in k_values},
                reciprocal_rank=reciprocal_rank(record.relevant_doc_ids, retrieved_ids),
            )
        )

    n = len(per_query)
    mean_recall = {
        k: (sum(q.recall_at_k[k] for q in per_query) / n if n else 0.0) for k in k_values
    }
    mrr = sum(q.reciprocal_rank for q in per_query) / n if n else 0.0
    return RetrievalReport(
        schema_version=RETRIEVAL_SCHEMA_VERSION,
        git_sha=git_sha(),
        timestamp=_now_iso(),
        mode=mode,
        reranked=reranked,
        n=n,
        recall_at_k=mean_recall,
        mrr=mrr,
        per_query=per_query,
    )


def retrieval_report_to_markdown(report: RetrievalReport, *, label: str = "") -> str:
    """One-row summary line (used standalone or as a comparison-table row)."""
    cols = " · ".join(f"recall@{k}={report.recall_at_k[k]:.3f}" for k in sorted(report.recall_at_k))
    name = label or f"{report.mode}{'+rerank' if report.reranked else ''}"
    return f"| {name} | {report.n} | {cols} | MRR={report.mrr:.3f} |"
