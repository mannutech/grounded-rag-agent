"""The ``search_docs`` tool — the agent's only window onto document text.

Its handler calls the retriever, records every returned chunk in the run-scoped
ledger (keyed by ``chunk_id``), and returns the chunks to Cohere each carrying its
``chunk_id`` so the model's span citations can be resolved back to a real source.
Because this is the sole path that surfaces document text, the ledger is a
complete and trustworthy basis for citation checking.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from grounded_rag.agent.tools import ToolContext, ToolSpec


class SearchDocsArgs(BaseModel):
    """Arguments for the search_docs tool."""

    query: str = Field(description="The search query to find relevant documentation.")
    top_k: int | None = Field(
        default=None, description="Optional maximum number of snippets to return."
    )


def _search_docs_handler(args: BaseModel, ctx: ToolContext) -> list[dict[str, Any]]:
    assert isinstance(args, SearchDocsArgs)
    chunks = ctx.retriever.retrieve(args.query, top_k=args.top_k)
    documents: list[dict[str, Any]] = []
    for chunk in chunks:
        ctx.ledger[chunk.chunk_id] = chunk  # citation source of truth
        documents.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "score": chunk.score,
                "text": chunk.text,
            }
        )
    return documents


def make_search_docs_tool() -> ToolSpec:
    """Build the read-only search_docs :class:`ToolSpec`."""
    return ToolSpec(
        name="search_docs",
        description=(
            "Search the knowledge base and return the most relevant documentation "
            "snippets. Use this to ground every factual claim; cite the snippets you use."
        ),
        arg_model=SearchDocsArgs,
        handler=_search_docs_handler,
    )
