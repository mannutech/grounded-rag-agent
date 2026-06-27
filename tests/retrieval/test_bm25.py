"""BM25: rare-term ranking, absent-term zeros, deterministic tokenizer."""

from __future__ import annotations

from grounded_rag.core.types import Chunk
from grounded_rag.retrieval.bm25 import BM25Index


def _chunk(idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"d::{idx}",
        doc_id="d",
        file_path="docs/d.md",
        ordinal=idx,
        text=text,
        start_char=0,
        end_char=len(text),
        n_tokens=len(text.split()),
    )


def test_rare_term_ranks_its_document_first() -> None:
    chunks = [
        _chunk(0, "the cat sat on the mat"),
        _chunk(1, "the dog ran across the quantum field"),
        _chunk(2, "the bird flew over the house"),
    ]
    index = BM25Index()
    index.fit(chunks)
    scores = index.scores("quantum")
    assert int(scores.argmax()) == 1  # only doc 1 contains "quantum"
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_absent_term_yields_all_zero_scores() -> None:
    chunks = [_chunk(0, "alpha beta"), _chunk(1, "gamma delta")]
    index = BM25Index()
    index.fit(chunks)
    scores = index.scores("nonexistentterm")
    assert scores.shape == (2,)
    assert (scores == 0.0).all()


def test_tokenizer_is_lowercase_word_chars() -> None:
    assert BM25Index.tokenize("Hello, WORLD! it's 2026.") == ["hello", "world", "it", "s", "2026"]


def test_empty_corpus_scores_empty() -> None:
    index = BM25Index()
    index.fit([])
    assert index.scores("anything").shape == (0,)
    assert len(index) == 0
