# Security

Acme is **PCI-DSS Level 1** compliant. Raw card numbers are **never stored**:
card data is **tokenized at the moment of ingestion**, and only the token is
persisted. The token cannot be reversed back into a card number.

All data is **encrypted at rest with AES-256** and **in transit with TLS 1.2 or
higher**. Access to production systems requires hardware-backed multi-factor
authentication.
