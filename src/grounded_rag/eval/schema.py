"""Loading and validating the evaluation gold set.

The gold set is JSONL — one :class:`GoldRecord` per line. ``load_gold`` is strict:
malformed JSON, schema violations, duplicate ids, and broken ``must_refuse``
invariants all raise :class:`GoldParseError` with the offending line number, so a
bad gold file fails loudly at load rather than silently skewing metrics. Blank
lines and ``#`` comment lines are allowed.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from grounded_rag.core.errors import GoldParseError
from grounded_rag.core.types import GoldRecord


def load_gold(path: str | Path) -> list[GoldRecord]:
    """Parse and validate ``path`` into a list of :class:`GoldRecord`."""
    text = Path(path).read_text(encoding="utf-8")
    records: list[GoldRecord] = []
    seen_ids: set[str] = set()

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldParseError(line_no, f"invalid JSON: {exc}") from exc
        try:
            record = GoldRecord.model_validate(data)
        except ValidationError as exc:
            raise GoldParseError(line_no, f"schema error: {exc}") from exc

        _check_invariants(record, line_no)
        if record.id in seen_ids:
            raise GoldParseError(line_no, f"duplicate id: {record.id!r}")
        seen_ids.add(record.id)
        records.append(record)

    return records


def _check_invariants(record: GoldRecord, line_no: int) -> None:
    """A must_refuse case has nothing to retrieve or assert against."""
    if not record.must_refuse:
        return
    if record.relevant_doc_ids:
        raise GoldParseError(line_no, "must_refuse record must have empty relevant_doc_ids")
    if record.keypoints:
        raise GoldParseError(line_no, "must_refuse record must have empty keypoints")
    if record.expected_answer is not None:
        raise GoldParseError(line_no, "must_refuse record must have null expected_answer")
