"""Canonical cross-subsystem types and Protocols — the single source of truth.

Every type that crosses a subsystem boundary lives here so that the dependency
DAG stays acyclic::

    core  <-  retrieval  <-  agent  <-  eval

``core`` imports nothing from its siblings. Retrieval *data* types (``Chunk``,
``ScoredChunk``, ``RetrievalResult``) are defined here even though retrieval owns
the *logic* that produces them: the ``Retriever`` Protocol returns a
``RetrievalResult``, so its shape must be reachable from ``core`` without an
upward import.

All models are frozen pydantic v2 models (not dataclasses) so any object can be
serialised into ``report.json`` and round-tripped back for inspection.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums (shared vocabulary)
# ---------------------------------------------------------------------------


class QueryType(StrEnum):
    """The kind of gold-set question, which selects how it is scored."""

    NORMAL = "normal"
    ADVERSARIAL = "adversarial"
    MULTI_HOP = "multi_hop"
    MUST_REFUSE = "must_refuse"


class RetrievalMode(StrEnum):
    """First-stage retrieval strategy."""

    DENSE = "dense"
    HYBRID = "hybrid"
    SPARSE = "sparse"  # BM25-only (no embeddings — runs with no API/model)


class FusionMethod(StrEnum):
    """How dense and sparse rankings are combined in hybrid mode."""

    RRF = "rrf"  # reciprocal rank fusion (rank-based, scale-free)
    WEIGHTED = "weighted"  # min-max normalised weighted sum


# ---------------------------------------------------------------------------
# Retrieval data types (logic lives in grounded_rag.retrieval)
# ---------------------------------------------------------------------------


class Chunk(BaseModel, frozen=True):
    """A unit of indexed text.

    ``text`` is always pristine — the source path is *never* baked into it. The
    optional path-in-embedding behaviour is applied only at embed time by
    ``grounded_rag.retrieval.embedder_iface.embed_text_for_chunk``.
    """

    chunk_id: str  # f"{doc_id}::{ordinal}", stable and deterministic
    doc_id: str
    file_path: str  # source path; used by the path-prepend toggle
    ordinal: int  # 0-based position within the document
    text: str  # raw chunk text (NEVER includes the prepended path)
    start_char: int
    end_char: int
    n_tokens: int


class ScoredChunk(BaseModel, frozen=True):
    """A chunk paired with the score/rank of the stage that produced it."""

    chunk: Chunk
    score: float
    rank: int  # 0-based rank within its result list
    stage: str  # "dense" | "bm25" | "fused" | "rerank"


class RetrievedChunk(BaseModel, frozen=True):
    """The cross-subsystem retrieval result the agent and eval both consume.

    Flattened from ``ScoredChunk`` at the agent boundary. It carries *both*
    ``doc_id`` (the recall@k key) and ``chunk_id`` (the only id a citation ever
    resolves to), plus a human-readable ``source``.
    """

    doc_id: str  # recall@k key
    chunk_id: str  # citation key
    text: str
    source: str  # human URI / file path
    score: float  # rerank score if reranked, else fused/dense score
    rank: int  # 0-based post-rerank order
    stage: str = "rerank"


# ---------------------------------------------------------------------------
# Agent output contract
# ---------------------------------------------------------------------------


class Citation(BaseModel, frozen=True):
    """A grounded span of the answer tied back to retrieved chunks.

    A citation whose referenced ids are absent from the run-scoped ledger is
    dropped before this object is built, so a populated ``chunk_ids`` is a real,
    audited link — not a model-fabricated one.
    """

    text: str  # quoted answer substring
    start: int  # char offset into the answer
    end: int  # exclusive
    chunk_ids: list[str]  # resolved via the ledger
    doc_ids: list[str]  # derived from chunk_ids (for recall / groundedness)
    sources: list[str]


class ToolCallRecord(BaseModel, frozen=True):
    """One tool invocation inside the agent loop, success or failure."""

    step: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    duration_ms: float = 0.0
    result_chunk_ids: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel, frozen=True):
    """Billing units across a single query.

    ``input_tokens`` / ``output_tokens`` are chat tokens; ``embed_tokens`` and
    ``rerank_docs`` are the separate units Cohere bills embed and rerank on, so
    cost can be attributed per stage without conflating them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    rerank_docs: int = 0  # rerank-v3.5 bills per doc / search-unit

    @property
    def total(self) -> int:
        """Total chat tokens (input + output)."""
        return self.input_tokens + self.output_tokens


class StageTiming(BaseModel, frozen=True):
    """Wall-clock duration of one pipeline stage."""

    stage: str  # "retrieval" | "rerank" | "tool:search_docs" | "generation"
    duration_ms: float


class AgentRunTrace(BaseModel, frozen=True):
    """Per-query telemetry (the structured trace), distinct from the answer."""

    query: str
    model: str
    steps: int
    timings: list[StageTiming]
    tool_calls: list[ToolCallRecord]
    injection_flagged: bool
    seed: int | None
    temperature: float
    total_duration_ms: float


class AgentResult(BaseModel, frozen=True):
    """The agent's top-level return: answer + every artifact eval needs.

    eval reads ``retrieved`` for recall@k, ``citations`` for groundedness,
    ``usage``/``cost_usd`` for cost, and ``trace.total_duration_ms`` for latency.
    """

    answer: str
    refused: bool
    citations: list[Citation]
    retrieved: list[RetrievedChunk]  # post-rerank context (recall@k source)
    tool_calls: list[ToolCallRecord]
    usage: TokenUsage
    cost_usd: float
    steps: int
    trace: AgentRunTrace


def cited_doc_ids(result: AgentResult) -> list[str]:
    """Flatten an :class:`AgentResult`'s citations into a de-duplicated doc-id list.

    Order is preserved (first occurrence wins) so the list is deterministic.
    """
    seen: dict[str, None] = {}
    for citation in result.citations:
        for doc_id in citation.doc_ids:
            seen.setdefault(doc_id, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Retrieval result (the Retriever Protocol return)
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel, frozen=True):
    """Everything a single ``retrieve`` call produced, with the toggles it used.

    ``mode`` / ``reranked`` / ``embed_file_path`` echo the configuration actually
    in effect so eval can log which variant produced which numbers.
    """

    query: str
    results: list[ScoredChunk]
    mode: RetrievalMode
    reranked: bool
    embed_file_path: bool


# ---------------------------------------------------------------------------
# Evaluation gold set + run artifacts
# ---------------------------------------------------------------------------


class GoldRecord(BaseModel):
    """One labelled evaluation case."""

    id: str
    question: str
    type: QueryType
    must_refuse: bool = False
    expected_answer: str | None = None  # reference answer (None for must_refuse)
    keypoints: list[str] = Field(default_factory=list)  # atomic facts the answer must contain
    relevant_doc_ids: list[str] = Field(default_factory=list)  # gold ids -> recall@k
    notes: str | None = None  # author rationale, not scored


class RunVariant(BaseModel):
    """A named point in the {rerank} x {path-embedding} x {mode} comparison grid."""

    name: str
    rerank: bool = True
    path_embedding: bool = True
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID


class JudgeBallot(BaseModel):
    """A single LLM-judge vote."""

    correct: bool
    grounded: bool
    refusal_appropriate: bool | None  # only meaningful for must_refuse cases
    keypoints_hit: list[str]
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class Verdict(BaseModel):
    """The majority decision over N judge ballots, with an agreement signal."""

    correct: bool
    grounded: bool
    refusal_appropriate: bool | None
    score: float  # mean ballot score
    keypoint_recall: float  # |hit| / |keypoints|
    n_votes: int
    agreement: float  # winning-margin fraction in [0, 1]
    ballots: list[JudgeBallot]


class QueryResult(BaseModel):
    """Per-query evaluation outcome — kept raw so report.json is debuggable."""

    record_id: str
    type: QueryType
    verdict: Verdict
    recall_at_k: dict[int, float | None]  # None == N/A (empty relevant / must_refuse)
    result: AgentResult
    must_refuse: bool
    refusal_correct: bool | None
    judge_usage: TokenUsage  # SEPARATE from agent usage (no double count)


class ReportSummary(BaseModel):
    """Headline aggregates over a whole run."""

    n: int
    correctness: float
    groundedness: float
    recall_at_k: dict[int, float | None]
    refusal_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    cost_per_query_usd: float


class Report(BaseModel):
    """A versioned, self-describing evaluation run."""

    schema_version: str
    git_sha: str
    timestamp: str
    seed: int | None
    mock_mode: bool
    config: dict[str, Any]  # Settings.model_dump() snapshot
    summary: ReportSummary
    per_query: list[QueryResult]


class ComparisonRow(BaseModel):
    """One variant's headline metrics in the comparison table."""

    variant: str
    correctness: float
    groundedness: float
    recall_at_5: float | None
    p95_latency_ms: float
    cost_per_query_usd: float


class ComparisonMatrix(BaseModel):
    """The full comparison grid the README quotes."""

    variants: list[RunVariant]
    rows: list[ComparisonRow]
    report_version: str


# ---------------------------------------------------------------------------
# Protocols — subsystems depend on these, never on each other's concretes
# ---------------------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """Produces a :class:`RetrievalResult` for a query (mode/rerank set at build)."""

    def retrieve(self, query: str) -> RetrievalResult: ...


@runtime_checkable
class Agent(Protocol):
    """Answers a question end-to-end; the variant is folded in at construction."""

    def answer(self, question: str) -> AgentResult: ...


@runtime_checkable
class Embedder(Protocol):
    """Dense embedding surface; document vs query input types are kept distinct."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder rerank surface returning ``(original_index, score)`` pairs."""

    def rerank(
        self, *, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]: ...


@runtime_checkable
class CohereClient(Protocol):
    """The one client surface a wrapper and the mock both satisfy.

    Return objects mirror the real ``cohere.ClientV2`` response shapes exactly
    (e.g. ``resp.embeddings.float``, ``resp.results[i].relevance_score``,
    ``resp.message.content[0].text``) so the code under test is the production
    path. ``is_mock`` lets a run be honestly tagged in its report.
    """

    is_mock: bool

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        citation_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any: ...

    def embed(
        self,
        *,
        model: str,
        texts: list[str],
        input_type: str,
        embedding_types: list[str] | None = None,
    ) -> Any: ...

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        max_tokens_per_doc: int = 4096,
    ) -> Any: ...
