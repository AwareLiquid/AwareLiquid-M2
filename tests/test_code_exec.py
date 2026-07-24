"""The arithmetic sandbox must compute correctly AND refuse everything else."""

import pytest

from awareliquid.adapter.code_exec import UnsafeCode, safe_eval


# ---- it computes --------------------------------------------------------

def test_the_x06_style_ratio():
    # market-share distractor vs the real revenue/market ratio
    assert round(safe_eval("97.25 / 1180 * 100"), 2) == 8.24


def test_multi_step_with_assignments():
    src = "cash = 60000 * 0.45\nfee = cash * 0.05\ncash - fee"
    assert safe_eval(src) == 25650.0


def test_whitelisted_functions():
    assert safe_eval("round(max(11.5, 12, 9), 1)") == 12
    assert safe_eval("min(12, 11.5, 9)") == 9


def test_subscripting_is_refused_even_though_it_looks_harmless():
    # Subscript is the first hop of the classic sandbox escape
    # ().__class__.__bases__[0], so it is refused outright.
    with pytest.raises(UnsafeCode):
        safe_eval("sorted([12, 11.5, 9])[-1]")


def test_comparison_returns_bool():
    assert safe_eval("40000 == 35000") is False


# ---- it refuses ---------------------------------------------------------

@pytest.mark.parametrize("src", [
    "__import__('os').system('echo hi')",
    "import os",
    "open('x')",
    "().__class__.__bases__",
    "eval('1+1')",
    "[x for x in range(3)]",
    "globals()",
    "(1).__class__",
    "getattr(1, 'real')",
    "lambda: 1",
])
def test_dangerous_constructs_are_rejected(src):
    with pytest.raises((UnsafeCode, SyntaxError)):
        safe_eval(src)


def test_unknown_names_do_not_resolve_to_builtins():
    # `range`/`print` are not in the whitelist -> NameError, never executed as a
    # real builtin (builtins are emptied).
    with pytest.raises((UnsafeCode, NameError)):
        safe_eval("print(1)")


def test_oversized_source_is_refused():
    with pytest.raises(UnsafeCode):
        safe_eval("1+" * 3000 + "1")


def test_arithmetic_errors_propagate_not_swallowed():
    with pytest.raises(ZeroDivisionError):
        safe_eval("1 / 0")


def test_code_must_end_in_an_expression():
    with pytest.raises(UnsafeCode):
        safe_eval("x = 1")
