"""The real Cohere client wrapper + a client factory.

``CohereClientWrapper`` adapts ``cohere.ClientV2`` to the :class:`CohereClient`
Protocol and owns the robustness the brief asks for: a request timeout (set on
the SDK client) plus retries with exponential backoff on transient failures
(429 / 5xx / network timeouts). Nothing downstream retries — this is the one
place that does.

The wrapper takes an optional injected ``client`` so its retry logic can be
tested offline with a flaky stand-in (no ``cohere`` import, no network).
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from grounded_rag.core.clients.mock import MockCohereClient
from grounded_rag.core.config import CohereSettings, Settings
from grounded_rag.core.errors import GroundedRagError
from grounded_rag.core.logging import get_logger
from grounded_rag.core.types import CohereClient

if TYPE_CHECKING:
    from collections.abc import Callable

_log = get_logger("cohere")

# HTTP statuses worth retrying: rate limiting, request timeout, and transient 5xx.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# Class names of transient SDK / httpx errors (matched by name to stay robust
# across SDK versions without importing fragile internal paths).
_RETRYABLE_NAMES = {
    "TooManyRequestsError",
    "ServiceUnavailableError",
    "GatewayTimeoutError",
    "InternalServerError",
    "RequestTimeoutError",
    "TimeoutException",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
}


def default_is_retryable(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient error worth retrying."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRYABLE_STATUS:
        return True
    return type(exc).__name__ in _RETRYABLE_NAMES


class CohereClientWrapper:
    """A retrying, timeout-bounded adapter over ``cohere.ClientV2``."""

    is_mock: bool = False

    def __init__(
        self,
        *,
        settings: CohereSettings,
        client: Any | None = None,
        is_retryable: Callable[[BaseException], bool] = default_is_retryable,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._is_retryable = is_retryable
        self._sleep = sleep
        if client is None:
            client = self._build_sdk_client(settings)
        self._client = client

    @staticmethod
    def _build_sdk_client(settings: CohereSettings) -> Any:
        import cohere  # lazy: only needed for the real path

        api_key = settings.api_key or os.environ.get("CO_API_KEY")
        if not api_key:
            raise GroundedRagError(
                "No Cohere API key: set GRA_COHERE__API_KEY or CO_API_KEY, or use mock_mode."
            )
        return cohere.ClientV2(
            api_key=api_key, base_url=settings.base_url, timeout=settings.timeout_s
        )

    def _call_with_retry(self, name: str, fn: Callable[..., Any], **kwargs: Any) -> Any:
        delay = self._settings.backoff_base_s
        last_exc: BaseException | None = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                return fn(**kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._settings.max_retries or not self._is_retryable(exc):
                    raise
                wait = min(delay, self._settings.backoff_max_s)
                _log.warning(
                    "cohere.retry", op=name, attempt=attempt + 1, wait_s=wait, error=str(exc)
                )
                self._sleep(wait)
                delay *= 2
        # Unreachable (loop either returns or raises), but keeps types honest.
        raise last_exc if last_exc is not None else RuntimeError("retry loop exited")

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        citation_options: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        # Pass optional params only when set, so an endpoint that doesn't accept
        # one (e.g. seed) isn't sent a null it might reject.
        if tools is not None:
            kwargs["tools"] = tools
        if seed is not None:
            kwargs["seed"] = seed
        if response_format is not None:
            kwargs["response_format"] = response_format
        if citation_options is not None:
            kwargs["citation_options"] = citation_options
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self._call_with_retry("chat", self._client.chat, **kwargs)

    def embed(
        self,
        *,
        model: str,
        texts: list[str],
        input_type: str,
        embedding_types: list[str] | None = None,
    ) -> Any:
        return self._call_with_retry(
            "embed",
            self._client.embed,
            model=model,
            texts=texts,
            input_type=input_type,
            embedding_types=embedding_types or ["float"],
        )

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        max_tokens_per_doc: int = 4096,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "max_tokens_per_doc": max_tokens_per_doc,
        }
        if top_n is not None:
            kwargs["top_n"] = top_n
        return self._call_with_retry("rerank", self._client.rerank, **kwargs)


def build_client(settings: Settings) -> CohereClient:
    """Return the right client for ``settings``.

    Uses the offline :class:`MockCohereClient` when ``mock_mode`` is set or no API
    key is available (so everything still runs, with the report honestly tagged
    ``mock_mode=True`` via ``is_mock``); otherwise the real wrapper.
    """
    if settings.mock_mode:
        return MockCohereClient()
    if not (settings.cohere.api_key or os.environ.get("CO_API_KEY")):
        _log.warning("cohere.no_key", msg="no API key found; falling back to MockCohereClient")
        return MockCohereClient()
    return CohereClientWrapper(settings=settings.cohere)
