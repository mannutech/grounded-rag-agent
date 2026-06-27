"""Thin re-export of the canonical :class:`CohereClient` Protocol.

The Protocol itself is defined in :mod:`grounded_rag.core.types` (so it sits at
the root of the dependency DAG). Importing it from here reads more naturally at
call sites that deal with clients.
"""

from __future__ import annotations

from grounded_rag.core.types import CohereClient

__all__ = ["CohereClient"]
