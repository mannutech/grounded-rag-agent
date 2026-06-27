# Idempotency

All mutating (POST) requests accept an **Idempotency-Key** header — a unique
client-generated string. If two requests are sent with the same key, the API
processes the first and returns the **original response** for the duplicate,
rather than performing the action twice.

Idempotency keys are **retained for 24 hours**. After that window a key may be
reused. Use a fresh UUID per logical operation.
