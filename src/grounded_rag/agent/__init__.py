"""Agent: bounded Cohere tool-use loop with grounded, cited, refusable answers.

Depends on ``core`` and ``retrieval``. This commit lands the read-only tool
registry, the safe calculator, the search_docs tool, input guarding, the system
prompt, and cost accounting; citation mapping, the retriever adapter, and the
loop itself follow in the next commits.
"""

from __future__ import annotations

from grounded_rag.agent.cost import price_query
from grounded_rag.agent.guards import PROMPT_INJECTION_NOTE, GuardResult, guard_input
from grounded_rag.agent.prompts import (
    INSUFFICIENT_CONTEXT_SENTINEL,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
)
from grounded_rag.agent.tool_calculator import make_calculator_tool, safe_calculate
from grounded_rag.agent.tool_search_docs import make_search_docs_tool
from grounded_rag.agent.tools import ToolContext, ToolRegistry, ToolSpec
from grounded_rag.agent.types import AgentRetriever

__all__ = [
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "AgentRetriever",
    "make_calculator_tool",
    "make_search_docs_tool",
    "safe_calculate",
    "guard_input",
    "GuardResult",
    "PROMPT_INJECTION_NOTE",
    "price_query",
    "SYSTEM_PROMPT",
    "REFUSAL_TEXT",
    "INSUFFICIENT_CONTEXT_SENTINEL",
]
