"""Application wiring: corpus -> index -> retriever -> agent.

This is the one place the subsystems are assembled into a runnable agent. The CLI
and the evaluation harness both go through here, so there is a single definition
of "build the system from the docs on disk".
"""

from __future__ import annotations

import json
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


def _chunk_doc(doc_id: str, file_path: str, text: str, settings: Settings) -> list[Chunk]:
    return chunk_document(
        doc_id=doc_id,
        file_path=file_path,
        text=text,
        chunk_tokens=settings.retrieval.chunk_tokens,
        overlap=settings.retrieval.chunk_overlap,
    )


def load_corpus(docs_dir: str | Path, settings: Settings) -> list[Chunk]:
    """Load and chunk a corpus.

    ``docs_dir`` may be a directory of ``.md`` / ``.txt`` files (``doc_id`` = file
    stem) or a single ``.jsonl`` file with one ``{"doc_id", "text"}`` per line
    (the format used for large corpora like the SQuAD dataset).
    """
    path = Path(docs_dir)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_corpus(path, settings)
    chunks: list[Chunk] = []
    for file in sorted(path.rglob("*")):
        if file.suffix.lower() not in _DOC_SUFFIXES:
            continue
        chunks.extend(_chunk_doc(file.stem, str(file), file.read_text(encoding="utf-8"), settings))
    return chunks


def _load_jsonl_corpus(path: Path, settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        chunks.extend(
            _chunk_doc(record["doc_id"], record.get("source", str(path)), record["text"], settings)
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
