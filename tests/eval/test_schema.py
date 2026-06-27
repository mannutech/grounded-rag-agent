"""Gold-set loading: strict validation with line-numbered errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded_rag.core.errors import GoldParseError
from grounded_rag.core.types import QueryType
from grounded_rag.eval.schema import load_gold

_VALID = """
# a comment line is ignored
{"id": "n1", "question": "What is X?", "type": "normal", "expected_answer": "X is Y", "keypoints": ["Y"], "relevant_doc_ids": ["d1"]}

{"id": "r1", "question": "Tell me tomorrow's lottery numbers", "type": "must_refuse", "must_refuse": true}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "gold.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_valid_records(tmp_path: Path) -> None:
    records = load_gold(_write(tmp_path, _VALID))
    assert [r.id for r in records] == ["n1", "r1"]
    assert records[0].type is QueryType.NORMAL
    assert records[1].must_refuse is True


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "n1", "question": "q", "type": "normal"}\n{bad json}\n')
    with pytest.raises(GoldParseError) as exc:
        load_gold(path)
    assert exc.value.line_no == 2


def test_missing_id_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"question": "q", "type": "normal"}\n')
    with pytest.raises(GoldParseError, match="schema error"):
        load_gold(path)


def test_bad_type_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "x", "question": "q", "type": "nonsense"}\n')
    with pytest.raises(GoldParseError):
        load_gold(path)


def test_must_refuse_invariants_enforced(tmp_path: Path) -> None:
    bad = '{"id": "r1", "question": "q", "type": "must_refuse", "must_refuse": true, "relevant_doc_ids": ["d1"]}\n'
    with pytest.raises(GoldParseError, match="relevant_doc_ids"):
        load_gold(_write(tmp_path, bad))


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    dup = (
        '{"id": "x", "question": "q1", "type": "normal"}\n'
        '{"id": "x", "question": "q2", "type": "normal"}\n'
    )
    with pytest.raises(GoldParseError, match="duplicate id") as exc:
        load_gold(_write(tmp_path, dup))
    assert exc.value.line_no == 2
