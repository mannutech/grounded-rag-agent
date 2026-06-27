# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to adhere to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Core: canonical types + Protocols, env-driven `Settings`, structured logging,
  git provenance, and a deterministic offline `MockCohereClient`.
- Retrieval: deterministic chunker; cosine, BM25, and RRF/weighted fusion;
  pluggable vector store (numpy in-memory default, lazy Chroma adapter); dense and
  hybrid retrievers with rerank + path-in-embedding toggles.
- Cohere client wrapper with timeout / retry / backoff, plus embed and rerank
  adapters.
- Agent: typed read-only tool registry, AST-safe calculator, `search_docs`,
  citation resolution, input guarding, and a bounded tool-use loop with a four-way
  refusal gate, trace, and cost.
- Evaluation: strict gold-set loader, metric functions (correctness, groundedness,
  recall@k, latency p50/p95, tool-call efficiency, cost), LLM-as-judge with an
  explicit rubric and majority voting, a versioned runner/report, and a comparison
  matrix over retrieval variants.
- A small sample corpus + 12-case gold set, a CLI (`ingest` / `ask` / `eval`), CI
  (ruff + mypy + pytest on 3.11 and 3.12), and pre-commit hooks.
