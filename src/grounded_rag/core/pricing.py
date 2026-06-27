"""Cost estimation — the single implementation, used by agent and eval alike.

Putting this in ``core`` (where both ``TokenUsage`` and ``CoherePricing`` live)
guarantees the agent's per-query cost and the eval harness's cost metric can
never compute cost two different ways.

Billing model (verified against cohere.com/pricing, 2026-06-28): chat and embed
bill per token; rerank bills per **search unit**, where one search unit is a
single query over up to 100 documents.
"""

from __future__ import annotations

import math

from grounded_rag.core.config import CoherePricing
from grounded_rag.core.types import TokenUsage

_DOCS_PER_SEARCH_UNIT = 100


def rerank_search_units(rerank_docs: int) -> int:
    """Number of billed rerank search units for ``rerank_docs`` documents."""
    if rerank_docs <= 0:
        return 0
    return math.ceil(rerank_docs / _DOCS_PER_SEARCH_UNIT)


def estimate_cost(usage: TokenUsage, pricing: CoherePricing) -> float:
    """USD cost of one query's usage under ``pricing`` (0.0 if pricing is unset)."""
    chat_input = usage.input_tokens / 1_000_000 * pricing.command_a_input_usd_per_1m
    chat_output = usage.output_tokens / 1_000_000 * pricing.command_a_output_usd_per_1m
    embed = usage.embed_tokens / 1_000_000 * pricing.embed_english_v3_usd_per_1m
    rerank = rerank_search_units(usage.rerank_docs) / 1_000 * pricing.rerank_v35_usd_per_1k_searches
    return chat_input + chat_output + embed + rerank
