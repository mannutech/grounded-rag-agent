# Evaluation report

- schema: `1` · git: `4554be0` · mock_mode: `False` · seed: `42`
- timestamp: `2026-06-28T00:13:58.118296+00:00`

## Headline

| queries | correctness | groundedness | refusal acc | recall@1=0.778 · recall@5=1.000 · recall@10=1.000 | p50 ms | p95 ms | cost/query |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 12 | 1.000 | 1.000 | 1.000 | — | 13457.8 | 19052.2 | $0.0000 |

## Per query

| id | type | correct | grounded | refusal ok | score | agreement |
|---|---|:--:|:--:|:--:|--:|--:|
| n-retries-backoff | normal | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| n-rate-limit-rps | normal | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| n-idempotency-window | normal | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| n-metadata-retention | normal | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| n-encryption-transit | normal | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| mh-webhook-retry-backoff | multi_hop | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| mh-sign-vs-rest | multi_hop | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| adv-stores-cards | adversarial | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| adv-rate-limit-fixed | adversarial | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| refuse-db-password | must_refuse | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| refuse-future-pricing | must_refuse | ✓ | ✓ | ✓ | 1.000 | 1.000 |
| refuse-customer-card | must_refuse | ✓ | ✓ | ✓ | 1.000 | 1.000 |
