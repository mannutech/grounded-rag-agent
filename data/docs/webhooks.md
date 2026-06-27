# Webhooks

Acme delivers events to your endpoint via HTTP POST. If your endpoint does not
return a 2xx response, the delivery is considered failed and is **retried up to 8
times over a 24-hour period** using **exponential backoff** (the same backoff
strategy the API client uses for request retries).

Every webhook is **signed with HMAC-SHA256** using your endpoint's signing
secret. The signature is sent in the `Acme-Signature` header. You must verify the
signature before trusting the payload; reject any request whose signature does not
match.
