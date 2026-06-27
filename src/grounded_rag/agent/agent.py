"""``RagAgent`` — a configured agent satisfying the core ``Agent`` Protocol.

The evaluation harness builds one ``RagAgent`` per comparison variant (the A/B
axes fold into the ``Settings`` it is given), so ``answer(question)`` is the single
entry point eval depends on. ``run_agent`` remains the free-function core.
"""

from __future__ import annotations

from grounded_rag.agent.loop import run_agent
from grounded_rag.agent.tools import ToolRegistry
from grounded_rag.agent.types import AgentRetriever
from grounded_rag.core.config import Settings
from grounded_rag.core.types import AgentResult, CohereClient


class RagAgent:
    """Wraps ``run_agent`` with a fixed retriever, client, and settings."""

    def __init__(
        self,
        *,
        retriever: AgentRetriever,
        client: CohereClient,
        settings: Settings,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._retriever = retriever
        self._client = client
        self._settings = settings
        self._registry = registry

    def answer(self, question: str) -> AgentResult:
        return run_agent(
            question,
            retriever=self._retriever,
            client=self._client,
            config=self._settings.agent,
            pricing=self._settings.cohere.pricing,
            model=self._settings.cohere.generation_model,
            registry=self._registry,
        )
