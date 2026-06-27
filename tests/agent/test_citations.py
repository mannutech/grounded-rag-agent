"""Citation resolution: ledger mapping, anti-fabrication, ordering."""

from __future__ import annotations

from grounded_rag.agent.citations import extract_citations
from grounded_rag.core.clients.mock import chat_text, make_citation
from grounded_rag.core.types import RetrievedChunk


def _ledger() -> dict[str, RetrievedChunk]:
    return {
        "d1::0": RetrievedChunk(
            doc_id="d1", chunk_id="d1::0", text="t1", source="docs/a.md", score=0.9, rank=0
        ),
        "d2::1": RetrievedChunk(
            doc_id="d2", chunk_id="d2::1", text="t2", source="docs/b.md", score=0.8, rank=1
        ),
    }


def _message(citations):  # type: ignore[no-untyped-def]
    return chat_text("the answer text here", citations=citations).message


def test_resolves_source_id_to_chunk() -> None:
    msg = _message([make_citation(0, 3, "the", ["d1::0"])])
    out = extract_citations(msg, _ledger())
    assert len(out) == 1
    assert out[0].chunk_ids == ["d1::0"]
    assert out[0].doc_ids == ["d1"]
    assert out[0].sources == ["docs/a.md"]
    assert (out[0].start, out[0].end, out[0].text) == (0, 3, "the")


def test_fabricated_id_is_dropped() -> None:
    msg = _message([make_citation(0, 3, "the", ["ghost::99"])])
    assert extract_citations(msg, _ledger()) == []


def test_wrapped_source_id_resolves_via_substring() -> None:
    msg = _message([make_citation(0, 3, "the", ["doc:search_docs:0:d2::1"])])
    out = extract_citations(msg, _ledger())
    assert out[0].chunk_ids == ["d2::1"]


def test_multiple_citations_ordered_by_start() -> None:
    msg = _message(
        [
            make_citation(10, 16, "answer", ["d2::1"]),
            make_citation(0, 3, "the", ["d1::0"]),
        ]
    )
    out = extract_citations(msg, _ledger())
    assert [c.start for c in out] == [0, 10]


def test_none_citations_returns_empty() -> None:
    assert extract_citations(_message(None), _ledger()) == []
