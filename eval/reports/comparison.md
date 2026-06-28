# Comparison matrix (real run)

Model: `command-a-03-2025`. Gold set: 12 cases. Judge: `n_votes=1` (single vote,
chosen to fit a 20-req/min trial key — see the noise note below).

| variant | correctness | groundedness | recall@5 | p95 ms | cost/query |
|---|--:|--:|--:|--:|--:|
| baseline (hybrid+rerank+path) | 0.917 | 1.000 | 1.000 | 22075.2 | $0.0000 |
| no-rerank | 1.000 | 1.000 | 1.000 | 18905.1 | $0.0000 |
| no-path-embedding | 1.000 | 1.000 | 1.000 | 19880.5 | $0.0000 |
| dense-only | 1.000 | 1.000 | 1.000 | 20738.2 | $0.0000 |

## Reading this honestly

- **recall@5 = 1.000 for every variant.** On a clean 6-document corpus, retrieval
  is easy: all four configs always place the gold doc in the top 5. The A/B is
  *indistinguishable on recall here* — a real null result, not a manufactured win.
  Differentiation needs a larger / noisier corpus.
- **The 0.917 vs 1.000 correctness spread is single-vote judge noise.** This run
  used `n_votes=1`; the same baseline config scored 1.000 in the headline run
  (`report.json`). That variance is exactly what the `n_votes=3` majority-vote
  default exists to smooth.
- **p95 latency (~18-22s) is throttle-dominated** by the trial key's 20-req/min
  limit (3.3s client-side spacing), not intrinsic latency.
- **Cost is $0.0000** because pricing is left unset (honest default).
