"""Typed, read-only tool registry.

A ``ToolSpec`` couples a pydantic argument model (which both validates the model's
arguments and emits the Cohere JSON schema) with a handler. ``ToolRegistry``
refuses to register anything not marked read-only, so even a successful prompt
injection cannot cause a side effect. ``dispatch`` validates arguments before
calling the handler and turns *any* tool failure into a ``ToolError`` the loop
can feed back to the model instead of crashing the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from grounded_rag.agent.types import AgentRetriever, RetrievedChunk
from grounded_rag.core.config import AgentConfig
from grounded_rag.core.errors import ToolError

# A handler takes validated args + context and returns a list of result
# "documents" (plain dicts). The loop wraps these into Cohere tool-result blocks.
ToolHandler = Callable[[BaseModel, "ToolContext"], list[dict[str, Any]]]


@dataclass
class ToolContext:
    """Run-scoped state shared with tool handlers.

    ``ledger`` is the citation source of truth: ``search_docs`` records every
    returned chunk here by ``chunk_id`` so citations can later be resolved (and
    fabricated ids rejected).
    """

    retriever: AgentRetriever
    config: AgentConfig
    ledger: dict[str, RetrievedChunk] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: argument schema + handler, read-only by contract."""

    name: str
    description: str
    arg_model: type[BaseModel]
    handler: ToolHandler
    read_only: bool = True

    def cohere_schema(self) -> dict[str, Any]:
        """The ``tools=`` entry Cohere ``chat`` expects for this tool."""
        params = self.arg_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolRegistry:
    """Name -> :class:`ToolSpec`, with validation and read-only enforcement."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.read_only:
            raise ToolError(f"refusing to register non-read-only tool: {spec.name!r}")
        if spec.name in self._specs:
            raise ToolError(f"tool already registered: {spec.name!r}")
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._specs)

    def cohere_specs(self) -> list[dict[str, Any]]:
        """All registered tools as Cohere ``tools=`` entries."""
        return [spec.cohere_schema() for spec in self._specs.values()]

    def dispatch(
        self, name: str, raw_args: dict[str, Any], ctx: ToolContext
    ) -> list[dict[str, Any]]:
        """Validate ``raw_args`` and run the handler; all failures -> ``ToolError``."""
        spec = self._specs.get(name)
        if spec is None:
            raise ToolError(f"unknown tool: {name!r}")
        try:
            args = spec.arg_model.model_validate(raw_args)
        except ValidationError as exc:
            raise ToolError(f"invalid arguments for {name!r}: {exc}") from exc
        try:
            return spec.handler(args, ctx)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"tool {name!r} failed: {exc}") from exc
