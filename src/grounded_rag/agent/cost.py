"""Agent-facing cost helper.

Delegates to the single :func:`grounded_rag.core.pricing.estimate_cost` so the
agent's per-query cost and the eval harness's cost metric stay identical.
"""

from __future__ import annotations

from grounded_rag.core.config import CoherePricing
from grounded_rag.core.pricing import estimate_cost
from grounded_rag.core.types import TokenUsage


def price_query(usage: TokenUsage, pricing: CoherePricing) -> float:
    """USD cost of a single query's token/search usage."""
    return estimate_cost(usage, pricing)
