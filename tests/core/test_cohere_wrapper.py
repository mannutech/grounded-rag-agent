"""Cohere wrapper retry/backoff + adapter translation, all offline."""

from __future__ import annotations

from typing import Any

import pytest

from grounded_rag.core.clients import (
    CohereClientWrapper,
    CohereEmbedder,
    CohereReranker,
    MockCohereClient,
    build_client,
)
from grounded_rag.core.config import CohereSettings, Settings

# -- adapters ---------------------------------------------------------------


def test_cohere_embedder_translation() -> None:
    embedder = CohereEmbedder(MockCohereClient(embed_dim=8), dim=8)
    docs = embedder.embed_documents(["a", "b", "c"])
    assert len(docs) == 3
    assert all(len(v) == 8 for v in docs)
    query = embedder.embed_query("a")
    assert len(query) == 8
    assert query == docs[0]  # same text -> same deterministic vector


def test_cohere_reranker_translation() -> None:
    reranker = CohereReranker(MockCohereClient())
    pairs = reranker.rerank(query="q", documents=["d0", "d1", "d2"], top_n=2)
    assert len(pairs) == 2
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in pairs)
    scores = [s for _, s in pairs]
    assert scores == sorted(scores, reverse=True)


# -- retry / backoff --------------------------------------------------------


class _RetryableError(Exception):
    def __init__(self, status_code: int = 429) -> None:
        self.status_code = status_code
        super().__init__("transient")


class _Flaky:
    """A fake SDK client whose ``chat`` fails ``fail_times`` then returns a sentinel."""

    def __init__(self, *, fail_times: int, exc: type[Exception], result: Any) -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.result = result
        self.calls = 0

    def chat(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc()
        return self.result


def _wrapper(client: _Flaky, *, max_retries: int = 3) -> CohereClientWrapper:
    settings = CohereSettings(max_retries=max_retries, backoff_base_s=0.0, backoff_max_s=0.0)
    return CohereClientWrapper(settings=settings, client=client, sleep=lambda _: None)


def test_retries_then_succeeds() -> None:
    flaky = _Flaky(fail_times=2, exc=_RetryableError, result="ok")
    wrapper = _wrapper(flaky)
    assert wrapper.chat(model="m", messages=[]) == "ok"
    assert flaky.calls == 3  # 2 failures + 1 success


def test_gives_up_after_max_retries() -> None:
    flaky = _Flaky(fail_times=99, exc=_RetryableError, result="never")
    wrapper = _wrapper(flaky, max_retries=3)
    with pytest.raises(_RetryableError):
        wrapper.chat(model="m", messages=[])
    assert flaky.calls == 4  # initial + 3 retries


def test_non_retryable_error_raises_immediately() -> None:
    flaky = _Flaky(fail_times=99, exc=ValueError, result="never")
    wrapper = _wrapper(flaky)
    with pytest.raises(ValueError):
        wrapper.chat(model="m", messages=[])
    assert flaky.calls == 1  # not retried


def test_wrapper_is_not_mock() -> None:
    assert _wrapper(_Flaky(fail_times=0, exc=ValueError, result="x")).is_mock is False


# -- client factory ---------------------------------------------------------


def test_build_client_mock_mode() -> None:
    client = build_client(Settings(mock_mode=True))
    assert isinstance(client, MockCohereClient)
    assert client.is_mock is True


def test_build_client_no_key_falls_back_to_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CO_API_KEY", raising=False)
    client = build_client(Settings(mock_mode=False))
    assert isinstance(client, MockCohereClient)
    assert client.is_mock is True
