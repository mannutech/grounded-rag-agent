"""The path-in-embedding toggle is honoured in exactly one place."""

from __future__ import annotations

from grounded_rag.core.types import Chunk
from grounded_rag.retrieval.embedder_iface import embed_text_for_chunk

_CHUNK = Chunk(
    chunk_id="d::0",
    doc_id="d",
    file_path="docs/retries.md",
    ordinal=0,
    text="Retries use exponential backoff.",
    start_char=0,
    end_char=32,
    n_tokens=4,
)


def test_path_prepended_when_enabled() -> None:
    out = embed_text_for_chunk(_CHUNK, embed_file_path=True)
    assert out.startswith("docs/retries.md")
    assert _CHUNK.text in out
    assert out == "docs/retries.md\n\nRetries use exponential backoff."


def test_text_only_when_disabled() -> None:
    assert embed_text_for_chunk(_CHUNK, embed_file_path=False) == _CHUNK.text
