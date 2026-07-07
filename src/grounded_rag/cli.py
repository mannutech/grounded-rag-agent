"""Command-line interface: ``ingest`` / ``ask`` / ``eval``.

Runs fully offline in mock mode (no key). With ``CO_API_KEY`` set it uses the real
Cohere stack. All user-facing output goes to stdout; structured logs go to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grounded_rag.core.clients.cohere_client import build_client
from grounded_rag.core.clients.embedder import CohereEmbedder
from grounded_rag.core.config import Settings, load_settings
from grounded_rag.core.logging import configure_logging
from grounded_rag.core.types import Embedder, RetrievalMode
from grounded_rag.eval.comparison import DEFAULT_VARIANTS, comparison_to_markdown, run_comparison
from grounded_rag.eval.report import report_to_markdown, write_report
from grounded_rag.eval.retrieval import evaluate_retrieval, retrieval_report_to_markdown
from grounded_rag.eval.runner import run_eval
from grounded_rag.eval.schema import load_gold
from grounded_rag.pipeline import build_agent, load_corpus, make_agent_factory
from grounded_rag.retrieval.retriever import build_index, build_retriever

_DOCS_DIR = "data/docs"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-rag", description=__doc__)
    parser.add_argument(
        "--docs", default=_DOCS_DIR, help="documents directory (default: data/docs)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="chunk the corpus and report index stats")
    ask = sub.add_parser("ask", help="ask the agent a question")
    ask.add_argument("question", help="the question to answer")
    ev = sub.add_parser("eval", help="run the evaluation harness")
    ev.add_argument("--compare", action="store_true", help="run the variant comparison matrix")
    rv = sub.add_parser(
        "retrieval-eval", help="score retrieval only (recall@k + MRR); no LLM/judge"
    )
    rv.add_argument("--gold", default=None, help="gold JSONL (default: eval config)")
    rv.add_argument(
        "--compare", action="store_true", help="compare sparse vs dense vs hybrid retrieval"
    )
    rv.add_argument(
        "--embedder",
        choices=["cohere", "local"],
        default="cohere",
        help="'local' uses model2vec (offline, no key) for dense/hybrid",
    )
    return parser


def _make_embedder(kind: str, settings: Settings) -> Embedder:
    if kind == "local":
        from grounded_rag.core.clients.local_embedder import LocalEmbedder

        return LocalEmbedder()
    return CohereEmbedder(build_client(settings), model=settings.cohere.embed_model)


def _cmd_retrieval_eval(
    settings: Settings, docs: str, gold_path: str | None, compare: bool, embedder_kind: str
) -> int:
    embedder = _make_embedder(embedder_kind, settings)
    gold = load_gold(gold_path or settings.eval.gold_path)
    chunks = load_corpus(docs, settings)
    modes = (
        [RetrievalMode.SPARSE, RetrievalMode.DENSE, RetrievalMode.HYBRID]
        if compare
        else [settings.retrieval.mode]
    )
    out_dir = Path(settings.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Return at least max(k) results so recall@k is well-defined for every k.
    max_k = max(settings.eval.k_values)
    rows = ["| method | n | recall@k | MRR |", "|---|--:|---|--:|"]
    for mode in modes:
        cfg = settings.retrieval.model_copy(
            update={
                "mode": mode,
                "use_reranker": False,
                "rerank_top_n": max_k,
                "top_k": max(settings.retrieval.top_k, max_k),
            }
        )
        index = build_index(chunks, embedder, cfg)
        retriever = build_retriever(cfg, index=index, embedder=embedder, reranker=None)
        report = evaluate_retrieval(gold, retriever, settings.eval.k_values)
        (out_dir / f"retrieval_{mode.value}.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        rows.append(retrieval_report_to_markdown(report, label=mode.value))
    print("\n".join(rows))
    print(f"\n(reports written to {out_dir}/retrieval_*.json)")
    return 0


def _cmd_ingest(settings: Settings, docs: str) -> int:
    chunks = load_corpus(docs, settings)
    docs_count = len({c.doc_id for c in chunks})
    print(f"corpus: {docs_count} documents, {len(chunks)} chunks from {docs}/")
    for doc_id in sorted({c.doc_id for c in chunks}):
        n = sum(1 for c in chunks if c.doc_id == doc_id)
        print(f"  {doc_id}: {n} chunk(s)")
    return 0


def _cmd_ask(settings: Settings, docs: str, question: str) -> int:
    client = build_client(settings)
    chunks = load_corpus(docs, settings)
    result = build_agent(settings, client, chunks).answer(question)
    print(f"\nQ: {question}\n")
    print(f"A: {result.answer}\n")
    print(
        f"refused={result.refused}  steps={result.steps}  cost=${result.cost_usd:.4f}"
        f"  mock_mode={getattr(client, 'is_mock', False)}"
    )
    if result.citations:
        print("citations:")
        for c in result.citations:
            print(f"  - {', '.join(c.chunk_ids)} ({', '.join(c.sources)})")
    return 0


def _cmd_eval(settings: Settings, docs: str, compare: bool) -> int:
    client = build_client(settings)
    chunks = load_corpus(docs, settings)
    if compare:
        factory = make_agent_factory(settings, client, chunks)
        matrix = run_comparison(settings, DEFAULT_VARIANTS, factory, client)
        print(comparison_to_markdown(matrix))
        return 0
    report = run_eval(settings, build_agent(settings, client, chunks), client)
    path = write_report(report, settings.eval.out_dir)
    print(report_to_markdown(report))
    print(f"\nwrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    settings = load_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    if args.command == "ingest":
        return _cmd_ingest(settings, args.docs)
    if args.command == "ask":
        return _cmd_ask(settings, args.docs, args.question)
    if args.command == "eval":
        return _cmd_eval(settings, args.docs, args.compare)
    if args.command == "retrieval-eval":
        return _cmd_retrieval_eval(settings, args.docs, args.gold, args.compare, args.embedder)
    return 1


if __name__ == "__main__":
    sys.exit(main())
