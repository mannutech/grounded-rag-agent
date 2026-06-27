"""Full bounded tool-use loop against a scripted Cohere client (no network)."""

from __future__ import annotations

from grounded_rag.agent.loop import run_agent
from grounded_rag.core.clients.mock import (
    ScriptedCohereClient,
    chat_text,
    chat_tool_calls,
    make_citation,
)
from grounded_rag.core.config import AgentConfig, CoherePricing

# Non-zero prices so the per-query cost is positive.
_PRICING = CoherePricing(command_a_input_usd_per_1m=2.0, command_a_output_usd_per_1m=10.0)
_MODEL = "command-a-03-2025"


def test_happy_path_tool_loop_then_cited_answer(retriever_factory, chunk) -> None:  # type: ignore[no-untyped-def]
    client = ScriptedCohereClient(
        [
            chat_tool_calls([("search_docs", {"query": "retries"})]),
            chat_tool_calls([("calculator", {"expression": "2+2"})]),
            chat_text(
                "Retries use exponential backoff; 2+2=4.",
                citations=[make_citation(0, 7, "Retries", ["d1::0"])],
            ),
        ]
    )
    config = AgentConfig(seed=42, temperature=0.0)
    result = run_agent(
        "how do retries work and what is 2+2",
        retriever=retriever_factory([chunk("d1::0", 0.85)]),
        client=client,
        config=config,
        pricing=_PRICING,
        model=_MODEL,
    )

    assert result.refused is False
    assert result.answer == "Retries use exponential backoff; 2+2=4."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_ids == ["d1::0"]
    assert result.steps == 3

    names = [tc.name for tc in result.tool_calls]
    assert names == ["search_docs", "calculator"]
    assert all(tc.ok for tc in result.tool_calls)

    assert result.usage.total > 0  # summed across all three turns
    assert result.cost_usd > 0
    stages = {t.stage for t in result.trace.timings}
    assert {"generation", "tool:search_docs", "tool:calculator"} <= stages
    assert result.trace.seed == 42
    assert result.trace.temperature == 0.0


def test_tool_error_is_recorded_and_run_recovers(retriever_factory, chunk) -> None:  # type: ignore[no-untyped-def]
    client = ScriptedCohereClient(
        [
            chat_tool_calls([("search_docs", {"query": "q"})]),
            chat_tool_calls([("calculator", {"expression": "os.system('x')"})]),  # unsafe
            chat_text("Final answer.", citations=[make_citation(0, 5, "Final", ["d1::0"])]),
        ]
    )
    result = run_agent(
        "question",
        retriever=retriever_factory([chunk("d1::0", 0.85)]),
        client=client,
        config=AgentConfig(),
        pricing=_PRICING,
        model=_MODEL,
    )

    assert result.refused is False
    calc = next(tc for tc in result.tool_calls if tc.name == "calculator")
    assert calc.ok is False
    assert calc.error is not None
    search = next(tc for tc in result.tool_calls if tc.name == "search_docs")
    assert search.ok is True
    assert result.steps == 3  # recovered and produced a final answer
