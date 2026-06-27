"""Resolve Cohere span citations back to real retrieved chunks.

Cohere returns citations as ``(start, end, text, sources)`` where each source has
an ``id`` referencing a document it was given. We resolve those ids against the
run-scoped ledger (every chunk ``search_docs`` surfaced). A source id that does
not resolve is dropped; a citation left with no resolvable source is dropped
entirely — so the agent can never present a fabricated id as a citation.

The resolver tolerates the SDK wrapping our id (e.g. ``"doc:search_docs:0:d1::0"``)
by falling back to a substring match against known chunk ids.
"""

from __future__ import annotations

from typing import Any

from grounded_rag.core.types import Citation, RetrievedChunk


def _resolve_source_id(source_id: str, ledger: dict[str, RetrievedChunk]) -> str | None:
    """Map a citation source id to a ledger chunk id, or None if unresolvable."""
    if source_id in ledger:
        return source_id
    # Fallback: the SDK may wrap our id inside its own source id string.
    for chunk_id in ledger:
        if chunk_id in source_id:
            return chunk_id
    return None


def _dedup(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def extract_citations(message: Any, ledger: dict[str, RetrievedChunk]) -> list[Citation]:
    """Build resolved, non-fabricated :class:`Citation` objects, ordered by start.

    Args:
        message: A Cohere ``response.message`` (real or mock) exposing ``citations``.
        ledger: chunk_id -> :class:`RetrievedChunk` recorded by ``search_docs``.
    """
    raw_citations = getattr(message, "citations", None) or []
    citations: list[Citation] = []
    for raw in raw_citations:
        chunk_ids: list[str] = []
        for source in getattr(raw, "sources", None) or []:
            source_id = getattr(source, "id", None)
            if not isinstance(source_id, str):
                continue
            resolved = _resolve_source_id(source_id, ledger)
            if resolved is not None:
                chunk_ids.append(resolved)
        if not chunk_ids:
            continue  # fabricated / unresolvable -> drop
        chunk_ids = _dedup(chunk_ids)
        citations.append(
            Citation(
                text=raw.text,
                start=raw.start,
                end=raw.end,
                chunk_ids=chunk_ids,
                doc_ids=_dedup([ledger[cid].doc_id for cid in chunk_ids]),
                sources=_dedup([ledger[cid].source for cid in chunk_ids]),
            )
        )
    citations.sort(key=lambda c: c.start)
    return citations
