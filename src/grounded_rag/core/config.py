"""Configuration — one root ``Settings`` object, env-driven, no scattered constants.

Every tunable lives here as a typed, defaulted field. A single ``Settings`` tree
composes per-subsystem sub-models so there is exactly one home for each value:

* model ids / timeouts / retries / pricing -> ``cohere``
* retrieval toggles (the A/B axes)          -> ``retrieval``
* agent params + refusal thresholds         -> ``agent``   (eval reads these too)
* judge params                              -> ``judge``
* eval run params                           -> ``eval``

Override anything from the environment with the ``GRA_`` prefix and ``__`` to
descend into a sub-model, e.g. ``GRA_AGENT__SEED=7`` or
``GRA_RETRIEVAL__USE_RERANKER=false``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from grounded_rag.core.types import FusionMethod, RetrievalMode, RunVariant


class CoherePricing(BaseModel):
    """Per-unit Cohere prices used by BOTH the agent's cost accounting and eval.

    Defaults are ``0.0`` on purpose. Cohere's public pricing page exposes
    enterprise / Model-Vault rates rather than the per-token API rates for these
    models, and guessing prices from memory would be worse than honest zeros.
    Set these to the rates on your account's dashboard (https://cohere.com/pricing)
    before quoting cost. A run with unset pricing reports ``pricing_configured =
    False`` so a $0.00 cost column is never mistaken for a real measurement.

    Billing units (verified against the pricing page, 2026-06-28): chat and embed
    bill per token; rerank bills per **search unit**, where one search unit is a
    single query over up to 100 documents.
    """

    command_a_input_usd_per_1m: float = 0.0
    command_a_output_usd_per_1m: float = 0.0
    embed_english_v3_usd_per_1m: float = 0.0
    rerank_v35_usd_per_1k_searches: float = 0.0

    @property
    def configured(self) -> bool:
        """True if any non-zero rate is set (cost numbers are then meaningful)."""
        return any(
            v > 0.0
            for v in (
                self.command_a_input_usd_per_1m,
                self.command_a_output_usd_per_1m,
                self.embed_english_v3_usd_per_1m,
                self.rerank_v35_usd_per_1k_searches,
            )
        )


class CohereSettings(BaseModel):
    """Cohere client + model selection + robustness knobs."""

    api_key: str | None = None  # if None the wrapper falls back to CO_API_KEY
    base_url: str | None = None
    embed_model: str = "embed-english-v3.0"
    rerank_model: str = "rerank-v3.5"
    generation_model: str = "command-a-03-2025"
    timeout_s: float = 30.0
    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_max_s: float = 8.0
    # Proactive client-side throttle: minimum seconds between API calls. 0 disables.
    # Set this to respect a key's rate limit instead of relying only on 429 retries
    # (e.g. a 20-calls/min trial key -> ~3.2).
    min_request_interval_s: float = 0.0
    pricing: CoherePricing = Field(default_factory=CoherePricing)


class RetrievalConfig(BaseModel):
    """Retrieval pipeline + the three evaluation A/B axes."""

    mode: RetrievalMode = RetrievalMode.HYBRID
    use_reranker: bool = True  # rerank on/off A/B (maps RunVariant.rerank)
    embed_file_path: bool = True  # path-embed on/off A/B (maps RunVariant.path_embedding)
    top_k: int = 20  # first-stage candidate count
    rerank_top_n: int = 5  # FINAL context size (the agent reads THIS, not top_k)
    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60  # RRF damping constant
    dense_weight: float = 0.5  # only used when fusion == WEIGHTED
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    vector_store: str = "memory"  # "memory" | "chroma" | "pgvector"


class AgentConfig(BaseModel):
    """Generation params + the refusal thresholds (the single source of truth).

    ``min_rerank_score`` and ``min_citation_support`` are read by eval too, so the
    harness and the runtime can never disagree about what "refused" means.
    """

    temperature: float = 0.0
    seed: int | None = 42
    max_steps: int = 6
    min_citation_support: int = 1  # min resolvable citations before we trust an answer
    min_rerank_score: float = 0.30  # weak-retrieval threshold -> refuse below it
    max_query_chars: int = 4000
    otel_enabled: bool = False


class JudgeConfig(BaseModel):
    """LLM-as-judge params; variance reduced via temperature 0 + fixed seed + votes."""

    n_votes: int = 3
    temperature: float = 0.0
    seed: int | None = 7
    model_override: str | None = None  # defaults to cohere.generation_model
    response_format_json: bool = True  # request response_format={"type": "json_object"}


class EvalConfig(BaseModel):
    """Evaluation run parameters."""

    gold_path: str = "eval/gold.jsonl"
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10])
    out_dir: str = "eval/reports"
    variants: list[str] = Field(default_factory=lambda: ["baseline"])


class Settings(BaseSettings):
    """The root configuration object. Construct via :func:`load_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="GRA_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mock_mode: bool = False  # force the MockCohereClient even if a key is present
    log_level: str = "INFO"
    log_json: bool = True
    cohere: CohereSettings = Field(default_factory=CohereSettings)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def load_settings(**overrides: Any) -> Settings:
    """Load settings from the environment / ``.env``, with optional explicit overrides."""
    return Settings(**overrides)


def variant_to_overrides(variant: RunVariant, base: Settings) -> Settings:
    """Return a deep copy of ``base`` with the variant's A/B axes applied.

    Each comparison row gets an isolated ``Settings`` so matrix rows never leak
    configuration into one another. ``base`` is left untouched.
    """
    new = base.model_copy(deep=True)
    new.retrieval = new.retrieval.model_copy(
        update={
            "use_reranker": variant.rerank,
            "embed_file_path": variant.path_embedding,
            "mode": variant.retrieval_mode,
        }
    )
    return new
