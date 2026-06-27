"""Shared types, Protocols, errors, and (later) config for grounded-rag-agent.

This subpackage has zero upward imports — every sibling depends on it, never the
other way around. Importing from ``grounded_rag.core`` gives you the canonical
definitions; do not redefine these types elsewhere.
"""

from __future__ import annotations

from grounded_rag.core.config import (
    AgentConfig,
    CoherePricing,
    CohereSettings,
    EvalConfig,
    JudgeConfig,
    RetrievalConfig,
    Settings,
    load_settings,
    variant_to_overrides,
)
from grounded_rag.core.errors import (
    GoldParseError,
    GroundedRagError,
    JudgeParseError,
    RetrievalError,
    ToolError,
)
from grounded_rag.core.gitmeta import git_sha
from grounded_rag.core.logging import configure_logging, get_logger
from grounded_rag.core.pricing import estimate_cost, rerank_search_units
from grounded_rag.core.types import (
    Agent,
    AgentResult,
    AgentRunTrace,
    Chunk,
    Citation,
    CohereClient,
    ComparisonMatrix,
    ComparisonRow,
    Embedder,
    FusionMethod,
    GoldRecord,
    JudgeBallot,
    QueryResult,
    QueryType,
    Report,
    ReportSummary,
    Reranker,
    RetrievalMode,
    RetrievalResult,
    RetrievedChunk,
    Retriever,
    RunVariant,
    ScoredChunk,
    StageTiming,
    TokenUsage,
    ToolCallRecord,
    Verdict,
    cited_doc_ids,
)

__all__ = [
    # errors
    "GroundedRagError",
    "GoldParseError",
    "JudgeParseError",
    "RetrievalError",
    "ToolError",
    # provenance
    "git_sha",
    # config
    "Settings",
    "CohereSettings",
    "CoherePricing",
    "RetrievalConfig",
    "AgentConfig",
    "JudgeConfig",
    "EvalConfig",
    "load_settings",
    "variant_to_overrides",
    # logging
    "configure_logging",
    "get_logger",
    # pricing
    "estimate_cost",
    "rerank_search_units",
    # enums
    "QueryType",
    "RetrievalMode",
    "FusionMethod",
    # retrieval data
    "Chunk",
    "ScoredChunk",
    "RetrievedChunk",
    "RetrievalResult",
    # agent output
    "Citation",
    "ToolCallRecord",
    "TokenUsage",
    "StageTiming",
    "AgentRunTrace",
    "AgentResult",
    "cited_doc_ids",
    # eval artifacts
    "GoldRecord",
    "RunVariant",
    "JudgeBallot",
    "Verdict",
    "QueryResult",
    "ReportSummary",
    "Report",
    "ComparisonRow",
    "ComparisonMatrix",
    # protocols
    "Retriever",
    "Agent",
    "Embedder",
    "Reranker",
    "CohereClient",
]
