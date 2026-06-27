"""The single home of the path-in-embedding decision.

The brief's deliberate choice is to *optionally* prepend a chunk's source path to
the text that gets embedded, on the theory that the path carries signal (a file
named ``retries.md`` helps a query about retries). Keeping that behaviour in one
function makes the A/B honest: ``Chunk.text`` stays pristine everywhere else, and
exactly one place decides what string is embedded.

Note this is an *index-time* decision — it changes the document vectors — so the
evaluation builds two indexes (path-on / path-off) rather than flipping a flag at
query time.
"""

from __future__ import annotations

from grounded_rag.core.types import Chunk


def embed_text_for_chunk(chunk: Chunk, *, embed_file_path: bool) -> str:
    """Return the exact string to embed for ``chunk``.

    Args:
        chunk: The chunk whose text (and optionally path) is embedded.
        embed_file_path: When True, prepend ``"{file_path}\\n\\n"`` to the text.
    """
    if embed_file_path:
        return f"{chunk.file_path}\n\n{chunk.text}"
    return chunk.text
