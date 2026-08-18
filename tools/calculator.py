"""
Expression Calculator
Parses and evaluates mathematical expression strings using a recursive descent parser.

Supported:
  Constants : pi, e, phi, tau, sqrt2, sqrt3, ln2, ln10, Inf
  Functions : sqrt, cbrt, abs, log, log2, log10, exp,
              sin, cos, tan, asin, acos, atan, atan2,
              sinh, cosh, tanh, ceil, floor, round, sign,
              max, min, pow, hypot, fact, gcd, lcm, nCr, nPr
  Operators : + - * / ^ % ! (postfix factorial)
  Grouping  : ( )

Usage:
  python calculator.py
  python calculator.py "2 + 3 * sqrt(10) - e / pi"
"""

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Constants & Functions
# ---------------------------------------------------------------------------

CONSTANTS: Dict[str, float] = {
    "pi":    math.pi,
    "e":     math.e,
    "phi":   (1 + math.sqrt(5)) / 2,
    "tau":   math.tau,
    "sqrt2": math.sqrt(2),
    "sqrt3": math.sqrt(3),
    "ln2":   math.log(2),
    "ln10":  math.log(10),
    "log2e": math.log2(math.e),
    "log10e": math.log10(math.e),
    "Inf":   math.inf,
    "inf":   math.inf,
}


def _fact(n: float) -> float:
    if math.isnan(n):
        return math.nan
    if math.isinf(n):
        if n > 0:
            return math.inf
        raise ValueError("Factorial of negative number")
    n = round(n)
    if n < 0:
        raise ValueError("Factorial of negative number")
    if n > 170:
        return math.inf
    result = 1
    for i in range(2, n + 1):
        result *= i
    return float(result)


def _gcd(a: float, b: float) -> float:
    return float(math.gcd(round(abs(a)), round(abs(b))))


def _lcm(a: float, b: float) -> float:
    return float(math.lcm(round(abs(a)), round(abs(b))))


def _ncr(n: float, r: float) -> float:
    n, r = round(n), round(r)
    if n < 0 or r < 0 or r > n:
        return 0.0
    try:
        return float(math.comb(n, r))
    except OverflowError:
        return math.inf


def _npr(n: float, r: float) -> float:
    n, r = round(n), round(r)
    if n < 0 or r < 0 or r > n:
        return 0.0
    try:
        return float(math.perm(n, r))
    except OverflowError:
        return math.inf


def _round(x: float, ndigits: Optional[float] = None) -> float:
    if ndigits is None:
        return float(round(x))
    return float(round(x, round(ndigits)))


FUNCTIONS: Dict[str, Callable] = {
    "sqrt":  math.sqrt,
    "cbrt":  lambda x: math.copysign(abs(x) ** (1/3), x),
    "abs":   abs,
    "log":   math.log,
    "log2":  math.log2,
    "log10": math.log10,
    "exp":   math.exp,
    "sin":   math.sin,
    "cos":   math.cos,
    "tan":   math.tan,
    "asin":  math.asin,
    "acos":  math.acos,
    "atan":  math.atan,
    "atan2": math.atan2,
    "sinh":  math.sinh,
    "cosh":  math.cosh,
    "tanh":  math.tanh,
    "ceil":  math.ceil,
    "floor": math.floor,
    "round": _round,
    "sign":  lambda x: math.copysign(1, x) if x != 0 else 0.0,
    "max":   max,
    "min":   min,
    "pow":   math.pow,
    "hypot": math.hypot,
    "fact":  _fact,
    "gcd":   _gcd,
    "lcm":   _lcm,
    "nCr":   _ncr,
    "nPr":   _npr,
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class Token:
    type: str   # 'num' | 'const' | 'fn' | 'op' | 'lparen' | 'rparen' | 'comma'
    value: Any = None


def tokenize(expr: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0

    while i < len(expr):
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        # Number (int or float, optional scientific notation)
        if ch.isdigit() or (ch == '.' and i + 1 < len(expr) and expr[i+1].isdigit()):
            j = i
            seen_dot = False
            while j < len(expr) and (expr[j].isdigit() or (expr[j] == '.' and not seen_dot)):
                if expr[j] == '.':
                    seen_dot = True
                j += 1
            if j < len(expr) and expr[j] in ('e', 'E'):
                k = j + 1
                if k < len(expr) and expr[k] in ('+', '-'):
                    k += 1
                if k < len(expr) and expr[k].isdigit():
                    j = k
                    while j < len(expr) and expr[j].isdigit():
                        j += 1
            tokens.append(Token('num', float(expr[i:j])))
            i = j
            continue

        # Identifier (function or constant)
        if ch.isalpha() or ch == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            name = expr[i:j]
            if name in FUNCTIONS:
                tokens.append(Token('fn', name))
            elif name in CONSTANTS:
                tokens.append(Token('const', name))
            else:
                raise SyntaxError(f"Unknown identifier: '{name}'")
            i = j
            continue

        if ch == '*' and i + 1 < len(expr) and expr[i+1] == '*':
            tokens.append(Token('op', '**'))
            i += 2
        elif ch in '+-*/^%!':
            tokens.append(Token('op', ch))
            i += 1
        elif ch == '(':
            tokens.append(Token('lparen'))
            i += 1
        elif ch == ')':
            tokens.append(Token('rparen'))
            i += 1
        elif ch == ',':
            tokens.append(Token('comma'))
            i += 1
        else:
            raise SyntaxError(f"Unexpected character: '{ch}'")

    return tokens


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------
# Grammar:
#   expr     → add_sub
#   add_sub  → mul_div  ( ('+' | '-') mul_div )*
#   mul_div  → unary    ( ('*' | '/' | '%') unary )*
#   unary    → ('-' | '+') unary | power
#   power    → postfix  ( ('^' | '**') unary )?     (right-associative)
#   postfix  → primary '!'*
#   primary  → NUMBER | CONST | FUNCTION '(' args ')' | '(' expr ')'
#   args     → expr (',' expr)*

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.steps: List[str] = []

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type_: str) -> Token:
        tok = self.peek()
        if tok is None or tok.type != type_:
            raise SyntaxError(f"Expected '{type_}', got {tok}")
        return self.consume()

    # -- Entry point ---------------------------------------------------------

    def parse(self) -> float:
        result = self.parse_expr()
        if self.pos < len(self.tokens):
            raise SyntaxError(f"Unexpected token: '{self.peek().value}'")
        return result

    # -- Grammar rules -------------------------------------------------------

    def parse_expr(self) -> float:
        return self.parse_add_sub()

    def parse_add_sub(self) -> float:
        left = self.parse_mul_div()
        while (t := self.peek()) and t.type == 'op' and t.value in '+-':
            op = self.consume().value
            right = self.parse_mul_div()
            result = left + right if op == '+' else left - right
            self.steps.append(f"  {_fmt(left)} {op} {_fmt(right)} = {_fmt(result)}")
            left = result
        return left

    def parse_mul_div(self) -> float:
        left = self.parse_unary()
        while (t := self.peek()) and t.type == 'op' and t.value in ('*', '/', '%'):
            op = self.consume().value
            right = self.parse_unary()
            if op == '*':   result = left * right
            elif op == '/': result = left / right
            elif op == '%': result = left % right
            self.steps.append(f"  {_fmt(left)} {op} {_fmt(right)} = {_fmt(result)}")
            left = result
        return left

    def parse_unary(self) -> float:
        t = self.peek()
        if t and t.type == 'op' and t.value in ('-', '+'):
            op = self.consume().value
            val = self.parse_unary()
            return -val if op == '-' else val
        return self.parse_power()

    def parse_power(self) -> float:
        left = self.parse_postfix()
        if (t := self.peek()) and t.type == 'op' and t.value in ('^', '**'):
            op = self.consume().value
            right = self.parse_unary()
            result = left ** right
            self.steps.append(f"  {_fmt(left)} {op} {_fmt(right)} = {_fmt(result)}")
            return result
        return left

    def parse_postfix(self) -> float:
        val = self.parse_primary()
        while (t := self.peek()) and t.type == 'op' and t.value == '!':
            self.consume()
            result = _fact(val)
            self.steps.append(f"  fact({_fmt(val)}) = {_fmt(result)}")
            val = result
        return val

    def parse_primary(self) -> float:
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected end of expression")

        if t.type == 'num':
            self.consume()
            return t.value

        if t.type == 'const':
            self.consume()
            val = CONSTANTS[t.value]
            self.steps.append(f"  {t.value} = {_fmt(val)}")
            return val

        if t.type == 'fn':
            name = self.consume().value
            self.expect('lparen')
            args = []
            if not (self.peek() and self.peek().type == 'rparen'):
                args.append(self.parse_expr())
                while self.peek() and self.peek().type == 'comma':
                    self.consume()
                    args.append(self.parse_expr())
            self.expect('rparen')
            result = FUNCTIONS[name](*args)
            arg_str = ', '.join(_fmt(a) for a in args)
            self.steps.append(f"  {name}({arg_str}) = {_fmt(result)}")
            return result

        if t.type == 'lparen':
            self.consume()
            val = self.parse_expr()
            self.expect('rparen')
            return val

        raise SyntaxError(f"Unexpected token: '{t.value}' in tokens {self.tokens[:self.pos]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(n: float) -> str:
    """Format a number for display."""
    if not math.isfinite(n):
        return str(n)
    if n == int(n) and abs(n) < 1e15:
        return str(int(n))
    return f"{n:.8g}"

def _evaluate(expr: str, verbose: bool = True) -> float:
    tokens = tokenize(expr)
    parser = Parser(tokens)
    result = parser.parse()

    if verbose:
        print(f"\nExpression : {expr}")
        if parser.steps:
            print("Steps      :")
            for step in parser.steps:
                print(step)
        print(f"Result     : {_fmt(result)}\n")

    return result


@tool
def calculate(expr: str, verbose: bool = True):
    """
    Parse and evaluate a mathematical expression string. Used for calculating mathematical equations. Use this tool over reasoning with numbers. Use parentheses to group operations and control order of evaluation. Use functions for more complex calculations. Note: this is calculator syntax, not Python — '^' (or '**') is exponentiation and '!' is postfix factorial; '//' is not supported.

    Args:
        expr:    The expression to evaluate, e.g. "2 + 3 * sqrt(10) - e / pi"
        verbose: If True, print step-by-step breakdown.
        Operators : + - * / ^ (power) % ! (factorial)
        Constants : pi, e, phi, tau, sqrt2, ln2, ...
        Functions : sqrt, sin, cos, log, exp, fact, nCr, ...

    Returns:
        The numeric result as a float, or an error message string if the
        expression is invalid.
    """

    try:
        return _evaluate(expr, verbose)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as err:
        return f"Error: {err}"


# ---------------------------------------------------------------------------
# CLI / Interactive REPL
# ---------------------------------------------------------------------------

def repl():
    print("Expression Calculator  (type 'quit' or Ctrl-C to exit)")
    print("Operators : + - * / ^ % !")
    print("Constants : pi, e, phi, tau, sqrt2, ln2, ...")
    print("Functions : sqrt, sin, cos, log, exp, fact, nCr, ...\n")
    while True:
        try:
            expr = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not expr:
            continue
        if expr.lower() in ('quit', 'exit', 'q'):
            print("Bye.")
            break
        try:
            _evaluate(expr)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as err:
            print(f"Error: {err}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        expr = " ".join(sys.argv[1:])
        try:
            _evaluate(expr)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as err:
            print(f"Error: {err}")
            sys.exit(1)
    else:
        repl()
