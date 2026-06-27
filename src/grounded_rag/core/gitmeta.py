"""Git provenance for evaluation reports.

Reports embed the commit they were produced at so a ``report.json`` is
reproducible and self-describing. This must never raise: eval may run from a
tarball, a CI cache, or a directory that is not a git checkout at all, in which
case we record ``"unknown"``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_UNKNOWN = "unknown"


def git_sha(*, short: bool = True) -> str:
    """Return the current commit SHA, or ``"unknown"`` if it cannot be determined.

    Args:
        short: Return the abbreviated SHA when True, the full 40-char SHA otherwise.
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            args,
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN
    if out.returncode != 0:
        return _UNKNOWN
    sha = out.stdout.strip()
    return sha or _UNKNOWN
