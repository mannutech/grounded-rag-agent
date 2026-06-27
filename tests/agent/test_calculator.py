"""Safe calculator: correct arithmetic, and every unsafe input rejected."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from grounded_rag.agent.tool_calculator import safe_calculate


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2+2", 4.0),
        ("3*(4-1)/2", 4.5),
        ("-7%3", 2.0),
        ("2**10", 1024.0),
        ("10/4", 2.5),
        ("7//2", 3.0),
        ("-(3+4)", -7.0),
        ("2 ** -1", 0.5),
    ],
)
def test_valid_arithmetic(expr: str, expected: float) -> None:
    assert safe_calculate(expr) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('rm -rf /')",
        "os.getcwd()",
        "open('x')",
        "(1).__class__",
        "9**9**9",  # exponent-cap DoS
        "[x for x in range(10)]",
        "lambda: 1",
        "1 if True else 2",
        "a + b",  # name
        "1; 2",  # multi-statement
        "",  # empty
        "2 +",  # trailing junk
        "1/0",  # division by zero
        "True + 1",  # bool literal disallowed
        "'a' + 'b'",  # string literal disallowed
    ],
)
def test_unsafe_input_raises_value_error(expr: str) -> None:
    with pytest.raises(ValueError):
        safe_calculate(expr)


def test_never_calls_eval_or_exec() -> None:
    with patch("builtins.eval") as mock_eval, patch("builtins.exec") as mock_exec:
        safe_calculate("3 * (2 + 5)")
        mock_eval.assert_not_called()
        mock_exec.assert_not_called()
