"""The bounded, grounded tool-use loop.

``run_agent`` drives a Cohere ``chat`` tool-use loop with hard guarantees:

* **Bounded** — at most ``config.max_steps`` model turns; exhausting them refuses
  rather than running unbounded.
* **Recoverable tools** — a tool that errors (bad args, unsafe input, JSON) is
  reported back to the model as a tool-result error and the run continues.
* **Grounded + refusable** — the agent refuses (returns ``REFUSAL_TEXT``) when
  retrieval is weak, the answer has no resolvable citation, the model emits the
  ``INSUFFICIENT_CONTEXT`` sentinel, or it hits the step limit.
* **Traced** — every generation and tool call is timed into the trace, with token
  usage and a per-query cost.

Scope note: token usage captures *generation* (chat) tokens precisely. Retrieval-
stage tokens (embed/rerank) are encapsulated inside the retriever and left at 0
here; generation dominates per-query cost. This is called out so a cost number is
never over-claimed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from grounded_rag.agent.citations import extract_citations
from grounded_rag.agent.cost import price_query
from grounded_rag.agent.guards import guard_input
from grounded_rag.agent.otel import span
from grounded_rag.agent.prompts import INSUFFICIENT_CONTEXT_SENTINEL, REFUSAL_TEXT, SYSTEM_PROMPT
from grounded_rag.agent.tool_calculator import make_calculator_tool
from grounded_rag.agent.tool_search_docs import make_search_docs_tool
from grounded_rag.agent.tools import ToolContext, ToolRegistry
from grounded_rag.agent.types import AgentRetriever
from grounded_rag.core.config import AgentConfig, CoherePricing
from grounded_rag.core.errors import ToolError
from grounded_rag.core.types import (
    AgentResult,
    AgentRunTrace,
    Citation,
    CohereClient,
    RetrievedChunk,
    StageTiming,
    TokenUsage,
    ToolCallRecord,
)


def default_registry() -> ToolRegistry:
    """A registry with the two read-only tools: search_docs + calculator."""
    registry = ToolRegistry()
    registry.register(make_search_docs_tool())
    registry.register(make_calculator_tool())
    return registry


def _safe_int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _final_text(message: Any) -> str | None:
    content = getattr(message, "content", None)
    if not content:
        return None
    text = getattr(content[0], "text", None)
    return text if isinstance(text, str) else None


def _tool_result_message(
    tool_call_id: str, documents: list[dict[str, Any]], error: str | None
) -> dict[str, Any]:
    """Build the role:"tool" message Cohere expects (document blocks for citations)."""
    if error is not None:
        blocks = [{"type": "document", "document": {"data": json.dumps({"error": error})}}]
    elif documents:
        blocks = []
        for doc in documents:
            document: dict[str, Any] = {"data": json.dumps(doc)}
            if "chunk_id" in doc:
                document["id"] = doc["chunk_id"]  # lets citations resolve to chunk ids
            blocks.append({"type": "document", "document": document})
    else:
        blocks = [
            {"type": "document", "document": {"data": json.dumps({"result": "no documents"})}}
        ]
    return {"role": "tool", "tool_call_id": tool_call_id, "content": blocks}


def _is_weak_retrieval(ledger: dict[str, RetrievedChunk], config: AgentConfig) -> bool:
    """Weak retrieval = nothing retrieved, or every reranked chunk below threshold."""
    if not ledger:
        return True
    reranked = [c.score for c in ledger.values() if c.stage == "rerank"]
    if reranked:
        return max(reranked) < config.min_rerank_score
    return False  # non-reranked results present -> rely on citation support instead


def run_agent(
    query: str,
    *,
    retriever: AgentRetriever,
    client: CohereClient,
    config: AgentConfig,
    pricing: CoherePricing,
    model: str,
    registry: ToolRegistry | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> AgentResult:
    """Answer ``query`` with a grounded, cited, refusable tool-use loop."""
    registry = registry or default_registry()
    guard = guard_input(query, max_chars=config.max_query_chars)
    ctx = ToolContext(retriever=retriever, config=config)

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": guard.text},
    ]
    timings: list[StageTiming] = []
    tool_records: list[ToolCallRecord] = []
    input_tokens = output_tokens = 0
    steps = 0
    final_message: Any = None
    final_text: str | None = None

    run_start = clock()
    for step in range(config.max_steps):
        steps += 1
        gen_start = clock()
        with span("generation", enabled=config.otel_enabled):
            resp = client.chat(
                model=model,
                messages=messages,
                tools=registry.cohere_specs(),
                temperature=config.temperature,
                seed=config.seed,
            )
        timings.append(StageTiming(stage="generation", duration_ms=(clock() - gen_start) * 1000))
        input_tokens += _safe_int(getattr(resp.usage.tokens, "input_tokens", 0))
        output_tokens += _safe_int(getattr(resp.usage.tokens, "output_tokens", 0))

        tool_calls = getattr(resp.message, "tool_calls", None)
        if not tool_calls:
            final_message = resp.message
            final_text = _final_text(resp.message)
            break

        messages.append(resp.message)  # assistant turn carries tool_plan + tool_calls
        for call in tool_calls:
            name = call.function.name
            call_start = clock()
            raw_args: dict[str, Any] = {}
            ok = True
            error: str | None = None
            documents: list[dict[str, Any]] = []
            try:
                parsed = json.loads(call.function.arguments)
                raw_args = parsed if isinstance(parsed, dict) else {}
                with span(f"tool:{name}", enabled=config.otel_enabled):
                    documents = registry.dispatch(name, raw_args, ctx)
            except (json.JSONDecodeError, ToolError) as exc:
                ok, error = False, str(exc)
            duration_ms = (clock() - call_start) * 1000
            timings.append(StageTiming(stage=f"tool:{name}", duration_ms=duration_ms))
            tool_records.append(
                ToolCallRecord(
                    step=step,
                    name=name,
                    arguments=raw_args,
                    ok=ok,
                    error=error,
                    duration_ms=duration_ms,
                    result_chunk_ids=[d["chunk_id"] for d in documents if "chunk_id" in d],
                )
            )
            messages.append(_tool_result_message(call.id, documents, error))

    citations: list[Citation] = []
    refused = False
    if final_text is None:
        refused = True  # exhausted max_steps without a final answer
    elif INSUFFICIENT_CONTEXT_SENTINEL in final_text:
        refused = True  # model-driven refusal
    else:
        citations = extract_citations(final_message, ctx.ledger)
        if _is_weak_retrieval(ctx.ledger, config) or len(citations) < config.min_citation_support:
            refused = True  # weak retrieval or unsupported claim
    if refused:
        citations = []

    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    trace = AgentRunTrace(
        query=guard.text,
        model=model,
        steps=steps,
        timings=timings,
        tool_calls=tool_records,
        injection_flagged=guard.injection_flagged,
        seed=config.seed,
        temperature=config.temperature,
        total_duration_ms=(clock() - run_start) * 1000,
    )
    return AgentResult(
        answer=REFUSAL_TEXT if refused else (final_text or REFUSAL_TEXT),
        refused=refused,
        citations=citations,
        retrieved=list(ctx.ledger.values()),
        tool_calls=tool_records,
        usage=usage,
        cost_usd=price_query(usage, pricing),
        steps=steps,
        trace=trace,
    )
