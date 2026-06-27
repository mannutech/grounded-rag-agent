# Retries

The Acme Payments client automatically retries failed requests. A request is
retried up to **5 times**. Retries use **exponential backoff** with full jitter:
the base delay is **200ms** and doubles on each attempt (200ms, 400ms, 800ms, …),
capped at a maximum delay of **8 seconds**.

Only transient failures are retried: HTTP **429** (rate limited) and **5xx**
responses. Other **4xx** responses are treated as permanent client errors and are
**not** retried. Requests that are not idempotent are only retried when an
Idempotency-Key is supplied (see Idempotency).
