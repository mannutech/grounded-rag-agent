"""A safe arithmetic calculator tool.

``safe_calculate`` evaluates a numeric expression by walking a parsed AST and
permitting *only* numeric literals and a fixed set of binary/unary operators —
never ``eval``/``exec``, never names, attributes, calls, comprehensions, or
subscripts. The ``**`` exponent is capped to prevent CPU-exhaustion via
expressions like ``9**9**9``.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from grounded_rag.agent.tools import ToolContext, ToolSpec

_BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPONENT = 100


def safe_calculate(expression: str) -> float:
    """Evaluate a basic arithmetic ``expression``; raise ``ValueError`` on anything unsafe."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc}") from exc
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError("exponent too large")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("division by zero")
        result: float = op(left, right)
        return result
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ValueError(f"unary operator not allowed: {type(node.op).__name__}")
        return float(unary(_eval_node(node.operand)))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


class CalculatorArgs(BaseModel):
    """Arguments for the calculator tool."""

    expression: str = Field(description="A basic arithmetic expression, e.g. '(3 + 4) * 2'.")


def _calculator_handler(args: BaseModel, ctx: ToolContext) -> list[dict[str, Any]]:
    assert isinstance(args, CalculatorArgs)
    value = safe_calculate(args.expression)
    return [{"tool": "calculator", "expression": args.expression, "result": value}]


def make_calculator_tool() -> ToolSpec:
    """Build the read-only calculator :class:`ToolSpec`."""
    return ToolSpec(
        name="calculator",
        description=(
            "Evaluate a basic arithmetic expression with +, -, *, /, %, ** and parentheses. "
            "No variables, functions, or names — numbers only."
        ),
        arg_model=CalculatorArgs,
        handler=_calculator_handler,
    )
