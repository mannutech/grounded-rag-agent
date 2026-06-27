"""Chunker: determinism, boundaries, overlap, and edge cases."""

from __future__ import annotations

import pytest

from grounded_rag.retrieval.chunker import Chunker, chunk_document

_DOC = " ".join(f"word{i}" for i in range(250))  # 250 whitespace tokens


def test_short_doc_is_one_chunk_spanning_whole_text() -> None:
    chunks = chunk_document("d", "docs/d.md", "hello world", chunk_tokens=512, overlap=64)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "d::0"
    assert c.text == "hello world"
    assert c.start_char == 0
    assert c.end_char == len("hello world")
    assert c.n_tokens == 2


def test_offsets_slice_back_to_text() -> None:
    chunks = chunk_document("d", "docs/d.md", _DOC, chunk_tokens=100, overlap=20)
    for c in chunks:
        assert _DOC[c.start_char : c.end_char] == c.text
        assert c.n_tokens <= 100


def test_overlap_is_exactly_overlap_tokens() -> None:
    chunks = chunk_document("d", "docs/d.md", _DOC, chunk_tokens=100, overlap=20)
    assert len(chunks) >= 2
    # consecutive windows share `overlap` tokens => second chunk starts 80 tokens in.
    first_tokens = chunks[0].text.split()
    second_tokens = chunks[1].text.split()
    assert first_tokens[-20:] == second_tokens[:20]


def test_determinism() -> None:
    a = chunk_document("d", "docs/d.md", _DOC, chunk_tokens=100, overlap=20)
    b = chunk_document("d", "docs/d.md", _DOC, chunk_tokens=100, overlap=20)
    assert [c.model_dump() for c in a] == [c.model_dump() for c in b]
    assert [c.chunk_id for c in a] == [f"d::{i}" for i in range(len(a))]


def test_empty_and_whitespace_docs_yield_no_chunks() -> None:
    assert chunk_document("d", "docs/d.md", "") == []
    assert chunk_document("d", "docs/d.md", "   \n\t  ") == []


def test_text_never_contains_file_path() -> None:
    chunks = chunk_document("d", "secret/path.md", _DOC, chunk_tokens=50, overlap=10)
    assert all("secret/path.md" not in c.text for c in chunks)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Chunker(chunk_tokens=10, overlap=10)
    with pytest.raises(ValueError, match="chunk_tokens"):
        Chunker(chunk_tokens=0, overlap=0)
