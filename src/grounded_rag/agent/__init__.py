"""Agent: bounded Cohere tool-use loop with grounded, cited, refusable answers.

Depends on ``core`` and ``retrieval``. ``run_agent`` is the free-function core;
``RagAgent`` wraps it to satisfy the core ``Agent`` Protocol that eval depends on.
"""

from __future__ import annotations

from grounded_rag.agent.agent import RagAgent
from grounded_rag.agent.citations import extract_citations
from grounded_rag.agent.cost import price_query
from grounded_rag.agent.guards import PROMPT_INJECTION_NOTE, GuardResult, guard_input
from grounded_rag.agent.loop import default_registry, run_agent
from grounded_rag.agent.prompts import (
    INSUFFICIENT_CONTEXT_SENTINEL,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
)
from grounded_rag.agent.retriever_adapter import RetrieverAdapter
from grounded_rag.agent.tool_calculator import make_calculator_tool, safe_calculate
from grounded_rag.agent.tool_search_docs import make_search_docs_tool
from grounded_rag.agent.tools import ToolContext, ToolRegistry, ToolSpec
from grounded_rag.agent.types import AgentRetriever

__all__ = [
    # loop + agent
    "run_agent",
    "RagAgent",
    "default_registry",
    # tools
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "make_calculator_tool",
    "make_search_docs_tool",
    "safe_calculate",
    # retrieval + citations
    "AgentRetriever",
    "RetrieverAdapter",
    "extract_citations",
    # guards + prompts
    "guard_input",
    "GuardResult",
    "PROMPT_INJECTION_NOTE",
    "SYSTEM_PROMPT",
    "REFUSAL_TEXT",
    "INSUFFICIENT_CONTEXT_SENTINEL",
    # cost
    "price_query",
]
