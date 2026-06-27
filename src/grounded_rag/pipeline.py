"""Application wiring: corpus -> index -> retriever -> agent.

This is the one place the subsystems are assembled into a runnable agent. The CLI
and the evaluation harness both go through here, so there is a single definition
of "build the system from the docs on disk".
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from grounded_rag.agent.agent import RagAgent
from grounded_rag.agent.retriever_adapter import RetrieverAdapter
from grounded_rag.core.clients.embedder import CohereEmbedder
from grounded_rag.core.clients.reranker import CohereReranker
from grounded_rag.core.config import Settings, variant_to_overrides
from grounded_rag.core.types import Chunk, CohereClient, RunVariant
from grounded_rag.retrieval.chunker import chunk_document
from grounded_rag.retrieval.retriever import build_index, build_retriever

_DOC_SUFFIXES = {".md", ".txt"}


def load_corpus(docs_dir: str | Path, settings: Settings) -> list[Chunk]:
    """Read and chunk every ``.md`` / ``.txt`` document under ``docs_dir``.

    ``doc_id`` is the file stem (so gold ``relevant_doc_ids`` read naturally);
    ``file_path`` is the path as given (used by the path-in-embedding toggle).
    """
    directory = Path(docs_dir)
    chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in _DOC_SUFFIXES:
            continue
        chunks.extend(
            chunk_document(
                doc_id=path.stem,
                file_path=str(path),
                text=path.read_text(encoding="utf-8"),
                chunk_tokens=settings.retrieval.chunk_tokens,
                overlap=settings.retrieval.chunk_overlap,
            )
        )
    return chunks


def build_retriever_adapter(
    settings: Settings, client: CohereClient, chunks: list[Chunk]
) -> RetrieverAdapter:
    """Embed + index ``chunks`` per ``settings`` and return the agent's retriever."""
    embedder = CohereEmbedder(client, model=settings.cohere.embed_model)
    reranker = CohereReranker(client, model=settings.cohere.rerank_model)
    index = build_index(chunks, embedder, settings.retrieval)
    retriever = build_retriever(
        settings.retrieval, index=index, embedder=embedder, reranker=reranker
    )
    return RetrieverAdapter(retriever)


def build_agent(settings: Settings, client: CohereClient, chunks: list[Chunk]) -> RagAgent:
    """Build a ready-to-answer :class:`RagAgent` for ``settings``."""
    adapter = build_retriever_adapter(settings, client, chunks)
    return RagAgent(retriever=adapter, client=client, settings=settings)


def make_agent_factory(
    settings: Settings, client: CohereClient, chunks: list[Chunk]
) -> Callable[[RunVariant], RagAgent]:
    """Return a factory that builds a variant-configured agent (rebuilding the index).

    The path-in-embedding axis changes document vectors, so each variant gets a
    freshly built index — the honest way to A/B it.
    """

    def factory(variant: RunVariant) -> RagAgent:
        return build_agent(variant_to_overrides(variant, settings), client, chunks)

    return factory
