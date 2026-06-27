# grounded-rag-agent

**An evaluation-first, grounded, agentic RAG reference implementation on Cohere's stack.**

Most RAG demos answer questions. The interesting question is *how well, how grounded, and at
what cost* — and how you'd **know**. This project is built the other way around: every quality
claim is backed by a number the evaluation harness produces, the agent cites its sources and
refuses when the context doesn't support an answer, and the retrieval design decisions are
settled by an **A/B comparison**, not by vibes.

Stack: Cohere [`embed-english-v3.0`](https://docs.cohere.com/docs/models) +
[`rerank-v3.5`](https://docs.cohere.com/docs/models) +
[`command-a-03-2025`](https://docs.cohere.com/docs/models) via `cohere.ClientV2`.

> **Status:** work in progress, built commit-by-commit. It runs fully offline for tests and CI
> via a mocked Cohere client; real evaluation numbers are produced with one API key. The
> **Results** section below is filled in from a real `make eval` run.

## Why this exists

It's a portfolio artifact. Its job is to demonstrate three things that matter for production RAG:

1. **Evaluation discipline.** A versioned harness with correctness, groundedness, retrieval
   recall@k, latency p50/p95, tool-call efficiency, and token/cost accounting — plus baked-in
   comparison runs (rerank on/off, path-in-embedding on/off, hybrid vs dense).
2. **Grounded, auditable answers.** The agent cites retrieved chunk ids, refuses when retrieval
   is weak, and uses only read-only tools.
3. **Production rigor.** Typed boundaries, env-driven config, structured logging, robust Cohere
   calls (timeout / retry / backoff), and a mocked client so CI is free, fast, and deterministic.

## Architecture

A small dependency DAG with no import cycles. All cross-subsystem types and Protocols live in
`grounded_rag.core`; siblings depend only on `core` (and `eval` additionally on `retrieval`).

```
core  <-  retrieval  <-  agent  <-  eval
 |          |              |          |
 types/     chunk/embed/   tool-use   gold set / LLM-judge /
 config/    BM25+dense/    loop /     metrics / comparison /
 clients/   rerank /       citations  versioned report
 (mock)     pluggable      / refusal
            vector store
```

## Quickstart

```bash
make setup          # create a venv and install the package + dev tools (uses uv)
make test           # full suite, offline, no API key required
make check          # lint (ruff) + types (mypy) + tests — what CI runs

# With a Cohere API key (export CO_API_KEY=...):
make ingest         # build the retrieval index from data/docs/
make ask Q="what does the spec say about retries?"
make eval           # run the evaluation harness -> versioned eval/reports/report.json
```

## Results

> Populated by a real `make eval` run. See [the evaluation section](#evaluation) for the harness.

_TBD — comparison table (rerank on/off · path-embedding on/off · hybrid vs dense) lands with c14._

## Evaluation

_TBD — gold-set design, judge rubric, and metric definitions land with the eval subsystem._

## License

[MIT](LICENSE) © 2026 Hitesh Goel
