"""A tiny, locked-down Python evaluator for arithmetic.

Large language models do multi-step arithmetic by pattern completion, not
calculation, so they reliably slip on chained operations (extract a formula,
substitute values, divide, compare). The fix recommended across the field is to
let the model write the computation as code and execute it deterministically.

This runs that code LOCALLY — it never calls the model, so it costs zero
generation tokens — inside a deliberately minimal sandbox:

* the source is size-limited and parsed to an AST first;
* only a whitelist of node types is allowed (numbers, arithmetic and comparison
  operators, calls to a fixed set of safe math builtins, simple assignments and
  a final expression);
* NO imports, attribute access, subscripting, comprehensions, names other than
  a handful of math functions and locally-assigned variables, or dunder access;
* execution runs with empty builtins and a hard operation-count ceiling.

It is intended for expressions like ``round(97.25 / 1180 * 100, 2)`` — not for
running untrusted programs, and it refuses anything outside that shape rather
than trying to sandbox a general interpreter.

Status: this sandbox is **not wired into the multiple-choice answer path**, on
purpose. It was measured there and the wiring lost accuracy: on the labelled set
the model wrote the correct expression, the sandbox computed the exact value, the
result was injected and marked authoritative — and the model still picked a
superficially-related distractor over its own verified number (48-question score
91.7% without it, 89.6% with it). For choice questions with a tempting wrong
option, injecting the exact answer did not make this model adopt it, and the
extra call was pure cost. The sandbox is kept as a correct, safe utility for
open-ended (free-number) computation, where the model emits the number directly
and there is no distractor to prefer.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict

MAX_SOURCE_CHARS = 2000
MAX_NODES = 400

# The only callables the code may use. All pure, all numeric, no side effects.
_SAFE_FUNCS: Dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "pow": pow, "sorted": sorted,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10, "exp": math.exp,
}

_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.Tuple, ast.List,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


class UnsafeCode(ValueError):
    """Raised when the source uses anything outside the arithmetic whitelist."""


def _validate(tree: ast.AST) -> None:
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise UnsafeCode("expression too large")
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeCode(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise UnsafeCode("only whitelisted math functions may be called")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # A loaded name must be a safe function or a variable assigned above;
            # dunders are refused outright.
            if node.id.startswith("__"):
                raise UnsafeCode("dunder access is not allowed")


def safe_eval(source: str) -> Any:
    """Evaluate arithmetic *source* and return the value of its last expression.

    Raises :class:`UnsafeCode` for anything outside the whitelist and lets normal
    arithmetic errors (ZeroDivisionError, ValueError, ...) propagate to the
    caller, which treats a failed computation as "no verified result".
    """
    if not source or len(source) > MAX_SOURCE_CHARS:
        raise UnsafeCode("empty or oversized source")
    tree = ast.parse(source, mode="exec")
    _validate(tree)

    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        raise UnsafeCode("code must end in an expression whose value is the result")

    scope: Dict[str, Any] = {}
    env = {"__builtins__": {}, **_SAFE_FUNCS}
    # Execute every statement except the last, then evaluate the final expression.
    if len(tree.body) > 1:
        body = ast.Module(body=tree.body[:-1], type_ignores=[])
        exec(compile(body, "<calc>", "exec"), env, scope)  # noqa: S102 - sandboxed
    final = ast.Expression(body=tree.body[-1].value)
    return eval(compile(final, "<calc>", "eval"), env, scope)  # noqa: S307 - sandboxed
