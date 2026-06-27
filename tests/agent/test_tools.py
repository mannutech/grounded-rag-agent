"""Tool registry: validation, read-only enforcement, and search_docs ledger."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from grounded_rag.agent.tool_calculator import make_calculator_tool
from grounded_rag.agent.tool_search_docs import make_search_docs_tool
from grounded_rag.agent.tools import ToolContext, ToolRegistry, ToolSpec
from grounded_rag.core.config import AgentConfig
from grounded_rag.core.errors import ToolError
from grounded_rag.core.types import RetrievedChunk


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_top_k: int | None = "unset"  # type: ignore[assignment]

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        self.last_top_k = top_k
        return self._chunks


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(
        doc_id=cid.split("::")[0],
        chunk_id=cid,
        text=f"text {cid}",
        source="docs/a.md",
        score=0.8,
        rank=0,
    )


def _ctx(chunks: list[RetrievedChunk] | None = None) -> ToolContext:
    return ToolContext(retriever=_FakeRetriever(chunks or []), config=AgentConfig())


def test_registry_dispatch_calculator() -> None:
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    out = registry.dispatch("calculator", {"expression": "2+3"}, _ctx())
    assert out == [{"tool": "calculator", "expression": "2+3", "result": 5.0}]


def test_registry_rejects_unknown_tool() -> None:
    with pytest.raises(ToolError, match="unknown tool"):
        ToolRegistry().dispatch("nope", {}, _ctx())


def test_registry_invalid_args_raise_tool_error() -> None:
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    with pytest.raises(ToolError, match="invalid arguments"):
        registry.dispatch("calculator", {"wrong": "field"}, _ctx())


def test_tool_failure_becomes_tool_error() -> None:
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    with pytest.raises(ToolError, match="failed"):
        registry.dispatch("calculator", {"expression": "os.system('x')"}, _ctx())


def test_registry_refuses_non_read_only_tool() -> None:
    class _Args(BaseModel):
        pass

    bad = ToolSpec(
        name="danger", description="", arg_model=_Args, handler=lambda a, c: [], read_only=False
    )
    with pytest.raises(ToolError, match="read-only"):
        ToolRegistry().register(bad)


def test_registry_rejects_duplicate_registration() -> None:
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    with pytest.raises(ToolError, match="already registered"):
        registry.register(make_calculator_tool())


def test_cohere_schema_shape() -> None:
    registry = ToolRegistry()
    registry.register(make_calculator_tool())
    registry.register(make_search_docs_tool())
    specs = registry.cohere_specs()
    assert {s["function"]["name"] for s in specs} == {"calculator", "search_docs"}
    calc = next(s for s in specs if s["function"]["name"] == "calculator")
    assert calc["type"] == "function"
    assert "expression" in calc["function"]["parameters"]["properties"]


def test_search_docs_populates_ledger_and_returns_chunk_ids() -> None:
    chunks = [_chunk("d1::0"), _chunk("d2::1")]
    ctx = _ctx(chunks)
    registry = ToolRegistry()
    registry.register(make_search_docs_tool())
    out = registry.dispatch("search_docs", {"query": "retries", "top_k": 2}, ctx)

    assert [d["chunk_id"] for d in out] == ["d1::0", "d2::1"]
    assert all("text" in d and "source" in d for d in out)
    # the ledger is the citation source of truth
    assert set(ctx.ledger) == {"d1::0", "d2::1"}
    assert ctx.ledger["d1::0"].doc_id == "d1"
    assert json.dumps(out)  # results are JSON-serialisable for the tool-result block
