"""Smoke test: the package imports and exposes a version.

This exists so the very first commit has a green ``make test``; every subsequent
commit keeps the suite green as real modules and their tests are added.
"""

from __future__ import annotations

import grounded_rag


def test_package_has_version() -> None:
    assert isinstance(grounded_rag.__version__, str)
    assert grounded_rag.__version__.count(".") >= 1
