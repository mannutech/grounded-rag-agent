"""Config: defaults, env overrides, the A/B variant mapping, and ownership invariants."""

from __future__ import annotations

import pytest

from grounded_rag.core.config import (
    AgentConfig,
    JudgeConfig,
    RetrievalConfig,
    Settings,
    load_settings,
    variant_to_overrides,
)
from grounded_rag.core.types import RetrievalMode, RunVariant


def test_defaults_load() -> None:
    s = load_settings()
    assert s.cohere.embed_model == "embed-english-v3.0"
    assert s.cohere.rerank_model == "rerank-v3.5"
    assert s.cohere.generation_model == "command-a-03-2025"
    assert s.retrieval.mode is RetrievalMode.HYBRID
    assert s.retrieval.rerank_top_n == 5
    assert s.agent.seed == 42
    assert s.judge.n_votes == 3
    assert s.eval.k_values == [1, 5, 10]
    assert s.mock_mode is False


def test_pricing_unconfigured_by_default() -> None:
    s = load_settings()
    assert s.cohere.pricing.configured is False


def test_env_override_nested_int(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRA_AGENT__SEED", "7")
    s = load_settings()
    assert s.agent.seed == 7


def test_env_override_nested_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRA_RETRIEVAL__USE_RERANKER", "false")
    s = load_settings()
    assert s.retrieval.use_reranker is False


def test_env_override_top_level_bool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRA_MOCK_MODE", "true")
    s = load_settings()
    assert s.mock_mode is True


def test_env_override_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRA_COHERE__PRICING__EMBED_ENGLISH_V3_USD_PER_1M", "0.10")
    s = load_settings()
    assert s.cohere.pricing.embed_english_v3_usd_per_1m == pytest.approx(0.10)
    assert s.cohere.pricing.configured is True


def test_variant_to_overrides_maps_axes_and_isolates_base() -> None:
    base = load_settings()
    assert base.retrieval.use_reranker is True  # baseline default

    variant = RunVariant(
        name="dense-no-rerank-no-path",
        rerank=False,
        path_embedding=False,
        retrieval_mode=RetrievalMode.DENSE,
    )
    derived = variant_to_overrides(variant, base)

    assert derived.retrieval.use_reranker is False
    assert derived.retrieval.embed_file_path is False
    assert derived.retrieval.mode is RetrievalMode.DENSE
    # base must be untouched (deep copy isolation)
    assert base.retrieval.use_reranker is True
    assert base.retrieval.embed_file_path is True
    assert base.retrieval.mode is RetrievalMode.HYBRID


def test_refusal_thresholds_live_only_on_agent_config() -> None:
    """Single source of truth: thresholds belong to the agent, not retrieval/judge."""
    agent_fields = set(AgentConfig.model_fields)
    assert {"min_rerank_score", "min_citation_support"} <= agent_fields
    assert "min_rerank_score" not in RetrievalConfig.model_fields
    assert "min_rerank_score" not in JudgeConfig.model_fields
    assert "min_citation_support" not in Settings.model_fields
