"""Optional OpenTelemetry spans, behind a flag and the ``otel`` extra.

``span`` is a no-op unless ``enabled`` is True *and* opentelemetry is installed, so
the agent never hard-depends on it and stages are always timed by the loop itself
regardless.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def span(name: str, *, enabled: bool = False) -> Iterator[None]:
    """Open an OTel span named ``name`` when enabled; otherwise do nothing."""
    if not enabled:
        yield
        return
    try:
        from opentelemetry import trace
    except ImportError:
        yield
        return
    tracer = trace.get_tracer("grounded_rag")
    with tracer.start_as_current_span(name):
        yield
