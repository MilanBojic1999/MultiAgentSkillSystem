"""Hermetic regression tests for the calculator tool.

Reuse the conftest hermetic-import / environment pattern — no network, no .env,
no live LLM required.
"""

import math

import pytest


# ---------------------------------------------------------------------------
# Precedence & associativity
# ---------------------------------------------------------------------------

PRECEDENCE_CASES = [
    # (expr, expected_text, description)
    ("2 * 3^2", "18", "power before multiply"),
    ("2^3^2", "512", "right-associative power: 2^(3^2)"),
    ("-2^2", "-4", "power before unary minus"),
    ("(-2)^2", "4", "parentheses override unary"),
    ("2^-2", "0.25", "negative exponent"),
    ("-3!", "-6", "factorial before unary minus"),
    ("3!!", "720", "repeated postfix factorial: (3!)! = 6! = 720"),
    ("2 * 3 + 4", "10", "multiply before add"),
    ("2 + 3 * 4", "14", "multiply before add (right)"),
    ("8 / 2 * 4", "16", "left-to-right for same precedence"),
    ("8 - 3 - 1", "4", "left-to-right subtraction"),
    ("10 % 3", "1", "modulo"),
    ("+5", "5", "unary plus is identity"),
    ("--5", "5", "double unary minus"),
    ("(1 + 2) * (3 + 4)", "21", "grouped sub-expressions"),
    ("2 * (3 + 4)", "14", "group right"),
    ("(1 + 2) * 3", "9", "group left"),
    ("2 ^ (1 + 2)", "8", "power of grouped expression"),
    ("-2^2 + 5", "1", "combined precedence"),
    ("2^3! - 10", "54", "factorial binds tighter than power: 2^(3!) - 10 = 2^6 - 10 = 64 - 10"),
]


@pytest.mark.parametrize("expr, expected_text, desc", PRECEDENCE_CASES)
def test_precedence(expr, expected_text, desc):
    from tools.calculator import evaluate_expression

    env = evaluate_expression(expr)
    assert env["ok"], f"Expected success for '{expr}': {env['error']}"
    assert env["result"]["text"] == expected_text, (
        f"{desc}: expected {expected_text}, got {env['result']['text']}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_val(n):
    """Match the calculator's internal formatting for test expectations."""
    if isinstance(n, int):
        return str(n)
    if n == int(n) and abs(n) < 1e15:
        return str(int(n))
    return f"{n:.10g}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSTANT_CASES = [
    ("pi", str(math.pi)),
    ("e", str(math.e)),
    ("2 * pi", _fmt_val(2 * math.pi)),
    ("sqrt2", _fmt_val(math.sqrt(2))),
    ("Inf", "inf"),
]


@pytest.mark.parametrize("expr, expected", CONSTANT_CASES)
def test_constants(expr, expected):
    from tools.calculator import evaluate_expression

    env = evaluate_expression(expr)
    assert env["ok"], f"Expected success for '{expr}': {env['error']}"
    # Float constants — compare as floats with tolerance
    result_val = float(env["result"]["text"])
    expected_val = float(expected)
    assert math.isclose(result_val, expected_val, rel_tol=1e-9), (
        f"Constant {expr}: expected ~{expected}, got {env['result']['text']}"
    )


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

FUNCTION_CASES = [
    ("sqrt(16)", "4"),
    ("abs(-3.5)", "3.5"),
    ("exp(0)", "1"),
    ("sin(0)", "0"),
    ("cos(0)", "1"),
    ("log(1)", "0"),
    ("log2(8)", "3"),
    ("log10(100)", "2"),
    ("ceil(2.3)", "3"),
    ("floor(2.9)", "2"),
    ("hypot(3, 4)", "5"),
    ("max(1, 5, 3)", "5"),
    ("min(1, 5, 3)", "1"),
    ("atan2(1, 1)", _fmt_val(math.atan2(1, 1))),
    ("pow(2, 3)", "8"),
    ("round(3.14159, 2)", "3.14"),
    ("sign(-10)", "-1"),
    ("sign(0)", "0"),
    ("sign(3)", "1"),
    ("cbrt(8)", "2"),
    ("cbrt(-8)", "-2"),
    ("log(100, 10)", "2"),    # two-arg log
]


@pytest.mark.parametrize("expr, expected", FUNCTION_CASES)
def test_functions(expr, expected):
    from tools.calculator import evaluate_expression

    env = evaluate_expression(expr)
    assert env["ok"], f"Expected success for '{expr}': {env['error']}"
    # For float results, compare numerically
    try:
        expected_num = float(expected)
        result_num = float(env["result"]["text"])
        assert math.isclose(result_num, expected_num, rel_tol=1e-9), (
            f"{expr}: expected ~{expected}, got {env['result']['text']}"
        )
    except ValueError:
        assert env["result"]["text"] == expected


# ---------------------------------------------------------------------------
# Exact integer arithmetic
# ---------------------------------------------------------------------------

EXACT_INTEGER_CASES = [
    ("fact(5)", "120"),
    ("fact(0)", "1"),
    ("fact(20)", str(math.factorial(20))),
    ("nCr(10, 3)", str(math.comb(10, 3))),
    ("nPr(10, 3)", str(math.perm(10, 3))),
    ("gcd(48, 18)", "6"),
    ("lcm(12, 18)", "36"),
    ("gcd(100, 75, 25)", "25"),
    ("lcm(4, 6, 8)", "24"),
    # r > n returns 0 for nCr / nPr
    ("nCr(5, 10)", "0"),
    ("nPr(5, 10)", "0"),
    # Large factorial — exact integer, no float overflow
    ("fact(50)", str(math.factorial(50))),
    # Large combinatorics
    ("nCr(100, 3)", str(math.comb(100, 3))),
    ("nPr(100, 3)", str(math.perm(100, 3))),
    # Integer literal stays int, operations on ints stay int when possible
    ("2 + 3", "5"),
    ("10 - 7", "3"),
    ("4 * 5", "20"),
    ("2^10", "1024"),
    ("17 % 5", "2"),
]


@pytest.mark.parametrize("expr, expected", EXACT_INTEGER_CASES)
def test_exact_integer(expr, expected):
    from tools.calculator import evaluate_expression

    env = evaluate_expression(expr)
    assert env["ok"], f"Expected success for '{expr}': {env['error']}"
    assert env["result"]["text"] == expected, (
        f"{expr}: expected {expected}, got {env['result']['text']}"
    )
    if expected != "0":
        assert env["result"]["kind"] == "integer"


# ---------------------------------------------------------------------------
# Kind metadata
# ---------------------------------------------------------------------------

def test_result_kind():
    from tools.calculator import evaluate_expression

    assert evaluate_expression("42")["result"]["kind"] == "integer"
    assert evaluate_expression("3.14")["result"]["kind"] == "real"
    assert evaluate_expression("22/7")["result"]["kind"] == "real"
    assert evaluate_expression("2 + 3")["result"]["kind"] == "integer"
    assert evaluate_expression("2 + 3.0")["result"]["kind"] == "real"


# ---------------------------------------------------------------------------
# Scientific notation
# ---------------------------------------------------------------------------

SCINOT_CASES = [
    ("1e5", "100000"),
    ("1.5e3", "1500"),
    ("1e-3", "0.001"),
    ("2.5E2", "250"),
    ("1e+10", "10000000000"),
]


@pytest.mark.parametrize("expr, expected", SCINOT_CASES)
def test_scientific_notation(expr, expected):
    from tools.calculator import evaluate_expression

    env = evaluate_expression(expr)
    assert env["ok"], f"Expected success for '{expr}': {env['error']}"
    # Compare numerically for float representations
    assert math.isclose(float(env["result"]["text"]), float(expected), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# include_steps
# ---------------------------------------------------------------------------

def test_include_steps_enabled():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("2 + 3 * 4", include_steps=True)
    assert env["ok"]
    assert len(env["steps"]) > 0
    # Should show the multiplication step before the addition step
    steps_text = " ".join(env["steps"])
    assert "3 * 4" in steps_text
    assert "2 + 12" in steps_text


def test_include_steps_disabled():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("2 + 3 * 4", include_steps=False)
    assert env["ok"]
    assert env["steps"] == []


def test_steps_include_constants():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("pi", include_steps=True)
    assert env["ok"]
    assert any("pi =" in s for s in env["steps"])


def test_steps_include_functions():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("sqrt(16)", include_steps=True)
    assert env["ok"]
    assert any("sqrt(16)" in s for s in env["steps"])


def test_steps_include_factorial():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("5!", include_steps=True)
    assert env["ok"]
    assert any("5! =" in s for s in env["steps"])


def test_steps_include_unary():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("-5", include_steps=True)
    assert env["ok"]
    assert any("-5 =" in s for s in env["steps"])


# ---------------------------------------------------------------------------
# Strict diagnostics — error codes and positions
# ---------------------------------------------------------------------------

ERROR_CASES = [
    # (expr, error_code, substring_in_message)
    ("", "blank_input", "blank"),
    ("1..2", "invalid_number", "Invalid number"),
    ("1e", "invalid_number", "missing exponent"),
    ("1e+", "invalid_number", "missing exponent"),
    ("foo", "unknown_identifier", "foo"),
    ("@", "unexpected_char", "@"),
    ("(1 + 2", "missing_token", "Expected"),
    (")", "unexpected_token", ")"),
    ("1 +", "missing_token", "Expected"),
    ("1 + +", "missing_token", "end"),
    ("sqrt()", "invalid_arguments", "sqrt()"),
    ("atan2(1)", "invalid_arguments", "atan2()"),
    ("max()", "invalid_arguments", "max()"),
    ("fact(3.5)", "integer_required", "integer"),
    ("fact(-1)", "domain_error", ""),
    ("gcd(3.5, 2)", "integer_required", "integer"),
    ("nCr(3.5, 1)", "integer_required", "integer"),
    ("sqrt(-1)", "domain_error", ""),
    ("log(0)", "domain_error", ""),
    ("1 / 0", "divide_by_zero", ""),
    # (-2)^0.5 — real domain error
    ("(-2)^0.5", "domain_error", ""),
]


@pytest.mark.parametrize("expr, expected_code, msg_substr", ERROR_CASES)
def test_error_diagnostics(expr, expected_code, msg_substr):
    from tools.calculator import evaluate_expression

    env = evaluate_expression(expr)
    assert not env["ok"], f"Expected error for '{expr}', got success"
    assert env["result"] is None
    assert env["error"]["code"] == expected_code, (
        f"Expected code '{expected_code}', got '{env['error']['code']}'"
    )
    if msg_substr:
        assert msg_substr.lower() in env["error"]["message"].lower(), (
            f"Message should contain '{msg_substr}': {env['error']['message']}"
        )
    assert env["steps"] == []


def test_error_position():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("1 + @ 2")
    assert not env["ok"]
    assert env["error"]["position"] == 4  # the '@'


def test_error_position_unknown_identifier():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("1 + unknown_var")
    assert not env["ok"]
    assert env["error"]["code"] == "unknown_identifier"
    assert env["error"]["position"] == 4  # start of 'unknown_var'


# ---------------------------------------------------------------------------
# Integer-domain rejection (exact values, not rounding)
# ---------------------------------------------------------------------------

def test_integer_domain_factorial_fractional():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("fact(3.5)")
    assert not env["ok"]
    assert env["error"]["code"] == "integer_required"


def test_integer_domain_factorial_inf():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("fact(Inf)")
    assert not env["ok"]
    assert env["error"]["code"] == "integer_required"


def test_integer_domain_gcd_fractional():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("gcd(3.5, 2)")
    assert not env["ok"]
    assert env["error"]["code"] == "integer_required"


def test_integer_domain_ncr_negative():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("nCr(5, -1)")
    assert not env["ok"]
    # math.comb raises ValueError for negative args
    assert env["error"]["code"] == "domain_error"


def test_integer_domain_ncr_negative_n():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("nCr(-5, 2)")
    assert not env["ok"]
    assert env["error"]["code"] == "domain_error"


# ---------------------------------------------------------------------------
# Exact integer acceptance — integral floats (5.0) are OK
# ---------------------------------------------------------------------------

def test_exact_integral_float_accepted_for_fact():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("fact(5.0)")
    assert env["ok"], f"fact(5.0) should succeed, got: {env['error']}"
    assert env["result"]["text"] == "120"


def test_exact_integral_float_accepted_for_gcd():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("gcd(48.0, 18)")
    assert env["ok"], f"gcd(48.0, 18) should succeed, got: {env['error']}"
    assert env["result"]["text"] == "6"


# ---------------------------------------------------------------------------
# Right-associative power — explicit cases
# ---------------------------------------------------------------------------

def test_power_right_associative():
    from tools.calculator import evaluate_expression

    # 2^(3^2) = 2^9 = 512
    env = evaluate_expression("2^3^2")
    assert env["ok"]
    assert env["result"]["text"] == "512"


def test_power_vs_unary():
    from tools.calculator import evaluate_expression

    # -2^2 = -(2^2) = -4
    env = evaluate_expression("-2^2")
    assert env["ok"]
    assert env["result"]["text"] == "-4"


def test_power_group_override():
    from tools.calculator import evaluate_expression

    # (-2)^2 = 4
    env = evaluate_expression("(-2)^2")
    assert env["ok"]
    assert env["result"]["text"] == "4"


# ---------------------------------------------------------------------------
# Repeated factorial
# ---------------------------------------------------------------------------

def test_repeated_factorial():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("3!!!")
    assert env["ok"]
    # 3! = 6, 6! = 720, 720! is huge — verify it's a big integer
    assert env["result"]["kind"] == "integer"
    val = int(env["result"]["text"])
    assert val == math.factorial(math.factorial(math.factorial(3)))


# ---------------------------------------------------------------------------
# nCr / nPr: r > n returns 0
# ---------------------------------------------------------------------------

def test_ncr_r_gt_n_returns_zero():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("nCr(3, 7)")
    assert env["ok"]
    assert env["result"]["text"] == "0"


def test_npr_r_gt_n_returns_zero():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("nPr(3, 7)")
    assert env["ok"]
    assert env["result"]["text"] == "0"


# ---------------------------------------------------------------------------
# LangChain @tool contract — calculate.invoke({...})
# ---------------------------------------------------------------------------

def test_calculate_invoke_success():
    from tools.calculator import calculate

    result = calculate.invoke({"expr": "2 + 2"})
    assert result["ok"]
    assert result["result"]["text"] == "4"
    assert result["error"] is None


def test_calculate_invoke_error():
    from tools.calculator import calculate

    result = calculate.invoke({"expr": "1 / 0"})
    assert not result["ok"]
    assert result["error"]["code"] == "divide_by_zero"


def test_calculate_invoke_with_steps():
    from tools.calculator import calculate

    result = calculate.invoke({"expr": "3 + 4", "include_steps": True})
    assert result["ok"]
    assert len(result["steps"]) > 0


def test_calculate_invoke_default_no_steps():
    from tools.calculator import calculate

    result = calculate.invoke({"expr": "3 + 4"})
    assert result["ok"]
    assert result["steps"] == []


def test_calculate_invoke_no_stdout(capsys):
    from tools.calculator import calculate

    calculate.invoke({"expr": "3 + 4"})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ---------------------------------------------------------------------------
# TOOL_REGISTRY discovery
# ---------------------------------------------------------------------------

def test_calculate_in_tool_registry():
    from tools import TOOL_REGISTRY

    assert "calculate" in TOOL_REGISTRY, (
        "calculate should be auto-discovered in TOOL_REGISTRY"
    )
    from langchain_core.tools import BaseTool
    assert isinstance(TOOL_REGISTRY["calculate"], BaseTool)


# ---------------------------------------------------------------------------
# CLI — main(argv)
# ---------------------------------------------------------------------------

def test_main_success(capsys):
    from tools.calculator import main

    rc = main(["calculator.py", "2 * 3^2"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "18" in captured.out


def test_main_with_steps(capsys):
    from tools.calculator import main

    rc = main(["calculator.py", "2 + 3 * 4"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Steps:" in captured.out


def test_main_error(capsys):
    from tools.calculator import main

    rc = main(["calculator.py", "1 / 0"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.out
    assert "division by zero" in captured.out.lower()


def test_main_blank(capsys):
    from tools.calculator import main

    rc = main(["calculator.py", ""])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.out


def test_main_invalid_number(capsys):
    from tools.calculator import main

    rc = main(["calculator.py", "1..2"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.out


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------

def test_overflow_power():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("10^10000")
    # Python big ints can represent this, but string conversion may hit
    # the 4300-digit safety limit.  Either outcome is correct as long as
    # the calculator does not crash.
    if env["ok"]:
        # Result is either an integer or a placeholder for huge ints.
        assert env["result"]["kind"] in ("integer", "real")
    else:
        assert env["error"]["code"] in ("overflow",)


# ---------------------------------------------------------------------------
# Unary minus on factorial — explicit
# ---------------------------------------------------------------------------

def test_unary_minus_factorial():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("-3!")
    assert env["ok"]
    # -3! = -(3!) = -6
    assert env["result"]["text"] == "-6"


def test_unary_minus_grouped_factorial():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("-(3!)")
    assert env["ok"]
    assert env["result"]["text"] == "-6"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_number():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("42")
    assert env["ok"]
    assert env["result"]["text"] == "42"
    assert env["result"]["kind"] == "integer"


def test_single_float():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("3.14")
    assert env["ok"]
    assert env["result"]["text"] == "3.14"


def test_negative_literal():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("-42")
    assert env["ok"]
    assert env["result"]["text"] == "-42"


# ---------------------------------------------------------------------------
# Comma-separated expressions
# ---------------------------------------------------------------------------

def test_comma_separated_basic():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("1, 2, 3")
    assert env["ok"]
    assert env["result"] is None
    assert "results" in env
    assert len(env["results"]) == 3
    assert env["results"][0]["text"] == "1"
    assert env["results"][1]["text"] == "2"
    assert env["results"][2]["text"] == "3"


def test_comma_separated_expressions():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("sqrt(pi/2), sqrt(5*pi/2), sqrt(9*pi/2)")
    assert env["ok"]
    assert "results" in env
    assert len(env["results"]) == 3
    for r in env["results"]:
        assert r["kind"] == "real"


def test_comma_separated_mixed_kinds():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("5, 3.14, 2 + 3")
    assert env["ok"]
    assert len(env["results"]) == 3
    assert env["results"][0]["kind"] == "integer"
    assert env["results"][1]["kind"] == "real"
    assert env["results"][2]["kind"] == "integer"


def test_comma_separated_functions():
    from tools.calculator import evaluate_expression

    env = evaluate_expression("fact(5), nCr(10, 3)")
    assert env["ok"]
    assert len(env["results"]) == 2
    assert env["results"][0]["text"] == "120"
    assert env["results"][1]["text"] == str(math.comb(10, 3))


def test_comma_separated_within_function_is_not_split():
    """Commas inside function calls are arguments, not separators."""
    from tools.calculator import evaluate_expression

    env = evaluate_expression("max(1, 5, 3)")
    assert env["ok"]
    # Single expression — commas are inside max()
    assert "result" in env
    assert env["result"]["text"] == "5"


def test_single_expression_still_uses_result():
    """Single expressions use 'result', not 'results', for backward compat."""
    from tools.calculator import evaluate_expression

    env = evaluate_expression("42")
    assert env["ok"]
    assert "result" in env
    assert env["result"]["text"] == "42"
    # The singular 'result' key is present for single expressions.
    # 'results' (plural) is absent — callers shouldn't have to check both.
    # (It's only present for multi-expression inputs.)
