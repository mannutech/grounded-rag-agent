"""Command-line interface: ``ingest`` / ``ask`` / ``eval``.

Runs fully offline in mock mode (no key). With ``CO_API_KEY`` set it uses the real
Cohere stack. All user-facing output goes to stdout; structured logs go to stderr.
"""

from __future__ import annotations

import argparse
import sys

from grounded_rag.core.clients.cohere_client import build_client
from grounded_rag.core.config import Settings, load_settings
from grounded_rag.core.logging import configure_logging
from grounded_rag.eval.comparison import DEFAULT_VARIANTS, comparison_to_markdown, run_comparison
from grounded_rag.eval.report import report_to_markdown, write_report
from grounded_rag.eval.runner import run_eval
from grounded_rag.pipeline import build_agent, load_corpus, make_agent_factory

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
    return parser


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
    return 1


if __name__ == "__main__":
    sys.exit(main())
