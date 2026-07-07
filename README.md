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

**Cross-family judging.** The judge runs through a provider-agnostic `ChatProvider`
(Cohere · OpenAI · Anthropic behind one adapter Protocol), so you can grade the
Cohere agent with a *different* model family — `GRA_JUDGE__PROVIDER=openai` or
`anthropic` — to guard against same-model self-preference bias. (Retrieval stays on
Cohere; OpenAI/Anthropic SDKs are optional extras, lazy-imported.)

**Reproducible reports.** Every run writes a versioned, self-describing
`report.json` (schema version, git sha, seed, mock flag, config snapshot) and a
human-readable `summary.md` from the same object.

## Results

From a real run against Cohere (`command-a-03-2025`, 12-case gold set,
[`eval/reports/report.json`](eval/reports/report.json)):

| queries | correctness | groundedness | refusal accuracy | recall@1 | recall@5 | recall@10 |
|--:|--:|--:|--:|--:|--:|--:|
| 12 | **1.000** | **1.000** | **1.000** | 0.778 | **1.000** | 1.000 |

Every case — normal, multi-hop, **adversarial** (resisting a false premise), and
**must-refuse** (declining when the answer isn't in the corpus) — was answered
correctly and grounded in retrieved context. recall@1 = 0.778 means the top reranked
chunk was from the gold doc on 7 of the 9 retrieval cases (must-refuse cases have no
relevant doc and are excluded); recall@5 = 1.000 means the gold doc was always in the
top 5.

> **Caveats (honest):** latency and cost are not quoted above. This run used a
> 20-requests/min trial key with a 3.3s client-side throttle, so the measured
> p50/p95 (~13s/~19s) are dominated by rate-limit spacing, not intrinsic latency.
> Cost is `$0.0000` because pricing is left unset (see [DESIGN.md](DESIGN.md));
> set `GRA_COHERE__PRICING__*` to your account rates for real cost numbers.

### A/B comparison

The comparison harness runs the same gold set under each retrieval variant —
isolating one decision per row: does rerank earn its keep? does putting the file
path into the embedding help? does hybrid (BM25 + dense) beat dense alone?

```bash
uv run python -m grounded_rag.cli eval --compare
```

Real run (`command-a-03-2025`, `eval/reports/comparison.md`):

| variant | correctness | groundedness | recall@5 |
|---|--:|--:|--:|
| baseline (hybrid + rerank + path) | 0.917 | 1.000 | 1.000 |
| no-rerank | 1.000 | 1.000 | 1.000 |
| no-path-embedding | 1.000 | 1.000 | 1.000 |
| dense-only | 1.000 | 1.000 | 1.000 |

**What this honestly shows.** On a clean 6-document corpus, **recall@5 saturates at
1.000 for every variant** — retrieval is easy enough that all four configs always
surface the gold doc in the top 5, so the A/B is *indistinguishable on recall here*.
That's a real null result, reported as-is rather than dressed up. The correctness
spread (0.917 vs 1.000) is within **single-vote judge noise**: this comparison ran
with `n_votes=1` to fit the trial key's 20-req/min limit, and the same baseline
config scored 1.000 in the headline run above — which is precisely why the judge
defaults to `n_votes=3` majority voting. To make the variants actually separate,
point the harness at a larger, noisier corpus (where rerank and hybrid earn their
keep) and run with the default vote count. The plumbing is the same; only the corpus
changes. (Mock mode runs every table end-to-end offline but produces degenerate
values — the offline mock neither retrieves nor judges like a real model.)

### Retrieval quality on a large corpus (SQuAD, offline)

The toy corpus above saturates recall. To measure retrieval where it actually
varies, the harness includes a **retrieval-only** evaluator (deterministic
recall@k + MRR against labeled relevant docs — no LLM, no judge) and a builder that
turns **SQuAD 2.0** into a 150-document corpus with 180 real questions
(`scripts/build_squad.py`). It runs **fully offline**: BM25 is pure computation and
the dense side uses a local `model2vec` embedder (no torch, no key).

```bash
uv run --extra local python -m grounded_rag.cli \
  --docs data/squad/docs.jsonl retrieval-eval --gold data/squad/gold.jsonl --compare --embedder local
```

Real result (150 questions, offline):

| method | recall@1 | recall@5 | recall@10 | MRR |
|---|--:|--:|--:|--:|
| BM25 (sparse) | **0.900** | 0.980 | 0.987 | **0.934** |
| dense (local) | 0.793 | 0.967 | 0.993 | 0.873 |
| hybrid (RRF) | 0.833 | 0.993 | **1.000** | 0.908 |

Now the methods are **distinguishable**. BM25 wins rank-1 and MRR (SQuAD questions
overlap lexically with their source paragraph); hybrid wins coverage (recall@10 =
1.000). And the two are genuinely complementary — BM25 finds 20 rank-1 hits dense
misses, dense finds 4 BM25 misses, so their union is correct 139/150 times at rank-1
— which is exactly why fusion helps. A live Cohere `embed-v3` embedder (drop the
`--embedder local` flag with a key) would raise the dense/hybrid numbers further.

### What the evaluation caught

The harness isn't decoration — on one run it flagged a real **over-refusal** on the
hardest case, `mh-sign-vs-rest`: *"What algorithm signs webhooks, and is that the
same algorithm used to encrypt data at rest?"* The agent **refused** a question it
should have answered, dropping correctness and refusal-accuracy to 0.917.

Root cause (a configuration effect, not a model failure): that question needs facts
from **two** docs at once — `webhooks.md` (HMAC-SHA256) and `security.md` (AES-256) —
but `rerank_top_n` defaults to **5** while the demo corpus has **6** docs, so one doc
is always cut. When the dropped doc is one of the two the question needs, the agent
can't ground both halves and honestly refuses rather than half-answer. It's
borderline: the same question answers perfectly (both docs cited) whenever both
survive the cut.

Two honest takeaways:

- A single `ask` shows the *happy path* and looks flawless; only running the gold set
  repeatedly reveals the system is ~92% reliable here with a specific weak spot. That
  gap is the entire reason this project is evaluation-first.
- The fix is a deliberate lever, not a silent default change: on a realistic corpus
  top-5 is the *right* call, so the demo simply raises it
  (`GRA_RETRIEVAL__RERANK_TOP_N=6`). The refusal itself is the grounded-over-
  comprehensive tradeoff working as designed — now visible as a number you can decide
  about.

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
