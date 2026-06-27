# Design notes

This document records the decisions behind `grounded-rag-agent` and the facts they
rest on. It is the "why", complementing the docstrings' "what".

## Dependency DAG (no cycles)

```
core  <-  retrieval  <-  agent  <-  eval
```

`grounded_rag.core` owns **every cross-subsystem type and Protocol** and imports
nothing from its siblings. Each sibling depends only on `core` (and `eval`
additionally on `retrieval`). This is what keeps the boundaries honest: there is
exactly one definition of `RetrievedChunk`, `Citation`, `AgentResult`,
`CohereClient`, etc., so the agent, the retriever, the mock, and the report can
never drift apart. The `pipeline` module is the only place that assembles the
subsystems into a runnable agent.

## Cohere SDK facts (verified against docs.cohere.com, 2026-06-28)

All three models in the brief are current and not deprecated: `embed-english-v3.0`,
`rerank-v3.5`, `command-a-03-2025`. We use `cohere.ClientV2` (the v2,
OpenAI-style `messages=` API), pinned to `cohere==7.0.4`.

- **embed v3 requires** both `input_type` and `embedding_types`. The corpus is
  embedded with `input_type="search_document"` and queries with `"search_query"` —
  mixing them degrades retrieval, so the `Embedder` Protocol splits the two methods
  to make the wrong call impossible. Vectors are read from `resp.embeddings.float`.
- **rerank v2 has no `return_documents`** and does not echo text; results map back
  to the input list via `result.index` (`result.relevance_score` is the score).
- **Tool use**: `tool_call.function.arguments` is a JSON *string* (we `json.loads`
  it); the assistant turn is appended verbatim; tool results go back as
  `role:"tool"` with document blocks so citations can ground in them.
- **Citations** (`resp.message.citations`) may be `None` and carry
  `start/end/text/sources[].id`; we resolve each source id back to a chunk id.

The `MockCohereClient` mirrors these exact attribute paths, so every offline test
exercises the same access the real SDK would.

## Key decisions

**Path-in-embedding is an A/B, isolated in one function.** Prepending a chunk's
file path to the embedded text is a deliberate, *testable* choice, not a default.
`embed_text_for_chunk` is the only place it happens; `Chunk.text` stays pristine.
Because it changes document vectors it is an index-time decision, so the eval
builds a fresh index per variant.

**Hybrid retrieval, RRF by default.** Dense (cosine over Cohere embeddings) and
sparse (BM25 via `rank_bm25`) run independently and fuse with Reciprocal Rank
Fusion — rank-based, so it needs no calibration between cosine and BM25 scales.
Rerank is a post-stage toggle decoupled from mode, making the three eval axes
({dense, hybrid} x {rerank on/off} x {path on/off}) fully independent. The vector
store is a Protocol: numpy in-memory by default, Chroma behind a lazy import + an
extra, so the default install stays dependency-light.

**Grounded, refusable agent.** The tool registry refuses to register a
non-read-only tool, so even a successful prompt injection cannot cause a side
effect. The calculator evaluates an AST allowlist (never `eval`/`exec`, exponent
capped against `9**9**9`). Citations resolve only to ids in the run-scoped ledger,
so a fabricated id is dropped. The agent refuses on weak retrieval (below
`min_rerank_score`), an unsupported answer (no resolvable citation), an
`INSUFFICIENT_CONTEXT` sentinel, or step exhaustion — and those thresholds live in
exactly one place (`AgentConfig`) that eval reads too, so runtime and evaluation
agree on what "refused" means.

**Evaluation is the centerpiece.** `recall_at_k` returns `None` (N/A) for
open-ended / must-refuse cases and aggregates exclude it — one convention
everywhere. The LLM judge has an explicit rubric and reduces variance with low
temperature, a seed varied per vote, and an N-vote majority whose margin is
reported as `agreement`; judge JSON is parsed defensively and a bad response
abstains rather than crashing. Judge tokens are counted through a proxy so they
never inflate the system-under-test's cost.

## Honest scope

- **Cost** captures generation (chat) tokens precisely. Retrieval-stage tokens
  (embed/rerank) are encapsulated in the retriever and reported as 0 at the agent
  level — generation dominates per-query cost. This is documented, not hidden.
- **Pricing** defaults to 0.0. Cohere's public page exposes enterprise/Vault rates,
  not per-token API rates for these models, and guessing is worse than honest
  zeros; a report flags an uncosted run rather than passing $0.00 off as real.
- **Mock mode** proves the system runs end-to-end offline; it does not produce
  meaningful quality numbers (the mock chat neither retrieves nor judges like a
  real model). Real numbers require `CO_API_KEY`.
