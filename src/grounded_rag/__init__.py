"""grounded-rag-agent.

An evaluation-first, grounded, agentic RAG reference implementation on Cohere's
stack (``embed-english-v3.0`` + ``rerank-v3.5`` + ``command-a-03-2025``).

The package is laid out as a small dependency DAG with no import cycles::

    core  <-  retrieval  <-  agent  <-  eval

``grounded_rag.core`` owns every cross-subsystem type and Protocol; the other
subpackages depend only on ``core`` (and ``eval`` additionally on ``retrieval``),
never on a sibling's internals. Public entry points are re-exported here as the
relevant modules land.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
