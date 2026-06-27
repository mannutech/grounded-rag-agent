"""Structured logging via ``structlog``.

JSON by default so logs are machine-parseable in CI and production; switchable to
a human-friendly console renderer for local development. Call
:func:`configure_logging` once at process start (the CLI does this); everywhere
else use :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys

import structlog

_SHARED_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    """Configure structlog process-wide.

    Args:
        level: Minimum level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        json_logs: Emit JSON lines when True; a coloured console format otherwise.
    """
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[*_SHARED_PROCESSORS, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger, optionally tagged with ``name``."""
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
