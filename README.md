# grounded-rag-agent

[![CI](https://github.com/mannutech/grounded-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/mannutech/grounded-rag-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**An evaluation-first, grounded, agentic RAG reference implementation on Cohere's stack.**

Most RAG demos answer questions. The interesting questions are *how well, how
grounded, at what cost* — and how you'd **know**. This project is built the other
way around: every quality claim is backed by a number the evaluation harness
produces, the agent cites its sources and refuses when the context doesn't support
an answer, and the retrieval design decisions are settled by an **A/B comparison**,
not by vibes.

Stack: Cohere [`embed-english-v3.0`](https://docs.cohere.com/docs/models) +
[`rerank-v3.5`](https://docs.cohere.com/docs/models) +
[`command-a-03-2025`](https://docs.cohere.com/docs/models) via `cohere.ClientV2`
(model ids verified current against docs.cohere.com, 2026-06-28).

It runs **fully offline** for tests and CI via a mocked Cohere client; real
evaluation numbers are produced with one API key.

## Why this exists

A portfolio artifact demonstrating three things that matter for production RAG:

1. **Evaluation discipline.** A versioned harness measuring correctness,
   groundedness, retrieval recall@k, latency p50/p95, tool-call efficiency, and
   token/cost — plus baked-in comparison runs (rerank on/off, path-in-embedding
   on/off, hybrid vs dense). Decisions made by measurement.
2. **Grounded, auditable answers.** The agent cites retrieved chunk ids, refuses
   when retrieval is weak or unsupported, and uses only read-only tools.
3. **Production rigor.** Typed boundaries (full type hints + strict mypy),
   env-driven config, structured logging, robust Cohere calls (timeout / retry /
   backoff), and a mocked client so CI is free, fast, and deterministic.

See [DESIGN.md](DESIGN.md) for the decisions and the verified Cohere SDK facts.

## Architecture

A small dependency DAG with no import cycles. `grounded_rag.core` owns every
cross-subsystem type and Protocol; siblings depend only on `core` (and `eval` on
`retrieval`).

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

Uses [uv](https://docs.astral.sh/uv/) as the runner.

```bash
make setup          # create a venv and install the package + dev tools
make check          # lint (ruff) + types (mypy) + tests — exactly what CI runs
make test           # full suite, offline, no API key required

# With a Cohere API key:
export CO_API_KEY=...
make ingest                                  # chunk the corpus, report index stats
make ask Q="what backoff is used for retries?"
make eval                                    # run the harness -> eval/reports/report.json
uv run python -m grounded_rag.cli eval --compare   # the A/B comparison table
```

Without a key everything still runs in **mock mode** (offline, deterministic);
reports are honestly tagged `mock_mode=true`.

## Evaluation

The point of the project. The gold set ([`eval/gold.jsonl`](eval/gold.jsonl)) spans
four case types, each scored differently:

| type | what it tests |
|---|---|
| `normal` | factual recall from a single doc |
| `multi_hop` | synthesis across two docs |
| `adversarial` | resisting a false premise in the question |
| `must_refuse` | declining when the answer isn't in the corpus |

**Metrics.** correctness and groundedness (LLM-judge), retrieval **recall@k**
(`None`/N/A for must-refuse, excluded from the mean), latency **p50/p95**,
tool-call efficiency (avg calls, failure rate, redundant-retrieval rate), and
token/cost. Each is a pure function over per-query facts, so `report.json` stays
debuggable down to the individual gold id.

**Judge variance reduction.** An explicit numbered rubric (correctness vs
keypoints, groundedness = every claim traceable to retrieved context, refusal
appropriateness), low temperature, a seed varied per vote, and an N-vote majority
whose winning margin is reported as `agreement`. Judge output is parsed defensively
and abstains rather than crashing on a bad response.

**Reproducible reports.** Every run writes a versioned, self-describing
`report.json` (schema version, git sha, seed, mock flag, config snapshot) and a
human-readable `summary.md` from the same object.

## Results

The comparison harness runs the same gold set under each retrieval variant. Run
`make eval` with `CO_API_KEY` set to populate this table from a real run
(`uv run python -m grounded_rag.cli eval --compare`):

| variant | correctness | groundedness | recall@5 | p95 ms | cost/query |
|---|--:|--:|--:|--:|--:|
| baseline (hybrid + rerank + path) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| no-rerank | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| no-path-embedding | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| dense-only | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Each row isolates one decision: does Cohere rerank earn its latency? does putting
the file path into the embedding help? does hybrid (BM25 + dense) beat dense alone?
The harness answers with numbers. (Mock mode runs the table end-to-end but produces
degenerate values — the offline mock neither retrieves nor judges like a real
model; real numbers need a key.)

## Project layout

```
src/grounded_rag/
  core/        types, config, logging, errors, clients (wrapper + mock + adapters), pricing
  retrieval/   chunker, cosine/BM25/fusion, vector store (+ lazy Chroma), retrievers, ingestion
  agent/       tool registry, calculator, search_docs, citations, guards, bounded loop, RagAgent
  eval/        gold schema, metrics, judge, runner, report, comparison
  pipeline.py  corpus -> index -> retriever -> agent assembly
  cli.py       ingest / ask / eval
data/docs/     sample corpus            eval/gold.jsonl   labelled gold set
tests/         unit + mocked-integration (offline) + a live smoke test (needs a key)
```

## Development

```bash
make fmt        # auto-format + auto-fix (ruff)
make lint       # ruff
make typecheck  # mypy (strict)
make test       # pytest (offline)
make check      # all three, like CI
```

CI runs ruff + mypy + pytest on Python 3.11 and 3.12. Pre-commit hooks are
available: `uv run --extra dev pre-commit install`.

## Non-goals

No fine-tuning, no Kubernetes/cloud deploy, no web scraping, and no heavyweight
framework (LangChain et al.) — the code is meant to be thin and legible. The corpus
is deliberately small and honest rather than inflated to fake scale.

## License

[MIT](LICENSE) © 2026 Hitesh Goel
