"""Cost estimation: exact USD math and the rerank search-unit rule."""

from __future__ import annotations

import pytest

from grounded_rag.core.config import CoherePricing
from grounded_rag.core.pricing import estimate_cost, rerank_search_units
from grounded_rag.core.types import TokenUsage

# A fixed, made-up price table so the math is exact and independent of real prices.
_PRICING = CoherePricing(
    command_a_input_usd_per_1m=2.0,
    command_a_output_usd_per_1m=10.0,
    embed_english_v3_usd_per_1m=0.1,
    rerank_v35_usd_per_1k_searches=2.0,
)


def test_rerank_search_units_rounds_up_per_100_docs() -> None:
    assert rerank_search_units(0) == 0
    assert rerank_search_units(1) == 1
    assert rerank_search_units(100) == 1
    assert rerank_search_units(101) == 2
    assert rerank_search_units(250) == 3


def test_estimate_cost_exact() -> None:
    usage = TokenUsage(
        input_tokens=1_000_000, output_tokens=500_000, embed_tokens=2_000_000, rerank_docs=50
    )
    # 1M*2 + 0.5M*10 + 2M*0.1 + ceil(50/100)=1 search -> 1/1000*2
    expected = 2.0 + 5.0 + 0.2 + (1 / 1000 * 2.0)
    assert estimate_cost(usage, _PRICING) == pytest.approx(expected)


def test_estimate_cost_zero_when_pricing_unset() -> None:
    usage = TokenUsage(input_tokens=10_000, output_tokens=5_000, embed_tokens=1_000, rerank_docs=20)
    assert estimate_cost(usage, CoherePricing()) == 0.0
    assert CoherePricing().configured is False
