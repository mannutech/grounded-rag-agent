"""Pipeline + CLI wiring against the real corpus and gold set (mock client)."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded_rag.cli import main
from grounded_rag.core.clients.mock import MockCohereClient
from grounded_rag.core.config import EvalConfig, RetrievalConfig, Settings
from grounded_rag.eval.runner import run_eval
from grounded_rag.eval.schema import load_gold
from grounded_rag.pipeline import build_agent, build_retriever_adapter, load_corpus

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "data" / "docs"
_GOLD = _ROOT / "eval" / "gold.jsonl"


def _settings() -> Settings:
    return Settings(
        eval=EvalConfig(gold_path=str(_GOLD), k_values=[1, 5]),
        retrieval=RetrievalConfig(top_k=10, rerank_top_n=5),
    )


def test_load_corpus_reads_docs() -> None:
    chunks = load_corpus(_DOCS, _settings())
    assert chunks
    doc_ids = {c.doc_id for c in chunks}
    assert {"retries", "rate-limits", "security"} <= doc_ids
    assert all(c.file_path.endswith(".md") for c in chunks)


def test_load_jsonl_corpus(tmp_path: Path) -> None:
    jsonl = tmp_path / "docs.jsonl"
    jsonl.write_text(
        '{"doc_id": "a", "text": "alpha content here"}\n'
        '{"doc_id": "b", "text": "beta content here"}\n',
        encoding="utf-8",
    )
    chunks = load_corpus(jsonl, _settings())
    assert {c.doc_id for c in chunks} == {"a", "b"}


def test_hybrid_retrieval_surfaces_lexical_match() -> None:
    settings = _settings()
    chunks = load_corpus(_DOCS, settings)
    adapter = build_retriever_adapter(settings, MockCohereClient(), chunks)
    results = adapter.retrieve("exponential backoff when retrying requests", top_k=5)
    assert results
    # BM25 should surface the retries doc for this lexically strong query.
    assert "retries" in {r.doc_id for r in results}


def test_gold_relevant_ids_exist_in_corpus() -> None:
    corpus_ids = {p.stem for p in _DOCS.glob("*.md")}
    for record in load_gold(_GOLD):
        for doc_id in record.relevant_doc_ids:
            assert doc_id in corpus_ids, f"{record.id} references unknown doc {doc_id!r}"


def test_run_eval_over_real_gold_offline() -> None:
    settings = _settings()
    client = MockCohereClient()
    chunks = load_corpus(_DOCS, settings)
    agent = build_agent(settings, client, chunks)
    report = run_eval(settings, agent, client)
    assert report.summary.n == len(load_gold(_GOLD))
    assert report.mock_mode is True


def test_cli_ingest_and_ask(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--docs", str(_DOCS), "ingest"]) == 0
    out = capsys.readouterr().out
    assert "documents" in out
    assert "retries" in out

    assert main(["--docs", str(_DOCS), "ask", "What backoff is used for retries?"]) == 0
    assert "A:" in capsys.readouterr().out
