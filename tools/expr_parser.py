"""
Shared Expression Parser
========================

Recursive-descent parser that builds an AST for mathematical expressions.
Used by both ``tools/calculator.py`` and ``tools/plotting.py``.

Two evaluation paths:
  ``evaluate_expression(expr)`` — immediate numeric result (calculator)
  ``compile_expression(expr, variables, namespace)`` — Python callable (plotting)

Precedence (tightest → loosest):
  postfix factorial (!)  →  power (^, right-associative)  →  unary (+/-)
  →  multiplication / division / modulo (* / %)  →  addition / subtraction (+ -)

Distinguishing examples:
  ``2 * 3^2 = 18``        (power before multiply)
  ``2^3^2 = 512``         (right-associative: 2^(3^2))
  ``-2^2 = -4``           (power before unary minus)
  ``(-2)^2 = 4``          (parentheses override)
  ``2^-2 = 0.25``         (negative exponent)
  ``-3! = -6``            (factorial before unary minus)

Supported:
  Constants : pi, e, phi, tau, sqrt2, sqrt3, ln2, ln10, Inf
  Functions : sqrt, cbrt, abs, log, log2, log10, exp,
              sin, cos, tan, asin, acos, atan, atan2,
              sinh, cosh, tanh, ceil, floor, round, sign,
              max, min, pow, hypot, fact, gcd, lcm, nCr, nPr
  Operators : + - * / ^ % ! (postfix factorial)
  Grouping  : ( )
  Variables: any identifier not in FUNCTIONS or CONSTANTS can be treated as a
             variable when listed in the *variables* set passed to the parser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ErrorCode(Enum):
    """Stable machine-readable error codes for the result envelope."""
    BLANK_INPUT = "blank_input"
    INVALID_NUMBER = "invalid_number"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    UNEXPECTED_CHAR = "unexpected_char"
    UNEXPECTED_TOKEN = "unexpected_token"
    MISSING_TOKEN = "missing_token"
    INVALID_ARGUMENTS = "invalid_arguments"
    INTEGER_REQUIRED = "integer_required"
    DOMAIN_ERROR = "domain_error"
    DIVIDE_BY_ZERO = "divide_by_zero"
    OVERFLOW = "overflow"


class CalculatorError(Exception):
    """Raised for user-expression failures that the tool envelope normalizes."""

    def __init__(self, code: ErrorCode, message: str, position: int | None = None) -> None:
        self.code = code
        self.message = message
        self.position = position
        super().__init__(message)


# ---------------------------------------------------------------------------
# Constants
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


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

Numeric = Union[int, float]


def _is_exact_integer(n: Numeric) -> bool:
    """True when *n* is an ``int`` or an integral finite ``float``."""
    if isinstance(n, int):
        return True
    if isinstance(n, float) and math.isfinite(n):
        return n == int(n)
    return False


def _require_integer(n: Numeric, *, label: str = "argument") -> int:
    """Return *n* as an ``int``, or raise ``CalculatorError``."""
    if isinstance(n, int):
        return n
    if isinstance(n, float):
        if not math.isfinite(n):
            raise CalculatorError(ErrorCode.INTEGER_REQUIRED,
                                  f"{label} must be a finite integer, got {n}")
        if n != int(n):
            raise CalculatorError(ErrorCode.INTEGER_REQUIRED,
                                  f"{label} must be an integer, got {n}")
        return int(n)
    raise CalculatorError(ErrorCode.INTEGER_REQUIRED,
                          f"{label} must be an integer, got {type(n).__name__}")


def _fmt(n: Numeric) -> str:
    """Format a number for display — exact for ints, round-trip-safe for floats."""
    if isinstance(n, int):
        try:
            return str(n)
        except ValueError:
            # Python 3.11+ limits integer string conversion (default 4300 digits).
            return f"<{n.bit_length() // 3 + 1}-digit integer>"
    if isinstance(n, complex):
        return str(n)
    if not math.isfinite(n):
        return str(n)
    # Exact-integer float within a display-safe range → show as int.
    if n == int(n) and abs(n) < 1e15:
        return str(int(n))
    # Round-trip-safe representation (no trailing noise).
    return f"{n:.10g}"


# ---------------------------------------------------------------------------
# Function arity metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FuncDef:
    """Wrapper around a math function with its arity constraints."""
    fn: Callable[..., Numeric]
    min_args: int = 1
    max_args: int | None = 1  # ``None`` means unlimited


FUNCTIONS: Dict[str, FuncDef] = {
    # One required argument
    "sqrt":  FuncDef(math.sqrt),
    "cbrt":  FuncDef(lambda x: math.copysign(abs(x) ** (1 / 3), x)),
    "abs":   FuncDef(abs),
    "exp":   FuncDef(math.exp),
    "log2":  FuncDef(math.log2),
    "log10": FuncDef(math.log10),
    "sin":   FuncDef(math.sin),
    "cos":   FuncDef(math.cos),
    "tan":   FuncDef(math.tan),
    "asin":  FuncDef(math.asin),
    "acos":  FuncDef(math.acos),
    "atan":  FuncDef(math.atan),
    "sinh":  FuncDef(math.sinh),
    "cosh":  FuncDef(math.cosh),
    "tanh":  FuncDef(math.tanh),
    "ceil":  FuncDef(math.ceil),
    "floor": FuncDef(math.floor),
    "sign":  FuncDef(lambda x: math.copysign(1, x) if x != 0 else 0),
    "fact":  FuncDef(lambda n: math.factorial(_require_integer(n, label="fact argument"))),

    # Two required arguments
    "atan2": FuncDef(math.atan2, min_args=2, max_args=2),
    "pow":   FuncDef(math.pow, min_args=2, max_args=2),
    "nCr":   FuncDef(lambda n, r: math.comb(_require_integer(n, label="n"),
                                             _require_integer(r, label="r")),
                     min_args=2, max_args=2),
    "nPr":   FuncDef(lambda n, r: math.perm(_require_integer(n, label="n"),
                                             _require_integer(r, label="r")),
                     min_args=2, max_args=2),

    # One required, one optional
    "log":   FuncDef(math.log, min_args=1, max_args=2),
    "round": FuncDef(round, min_args=1, max_args=2),

    # Variadic (≥1)
    "hypot": FuncDef(math.hypot, min_args=1, max_args=None),
    "max":   FuncDef(max, min_args=1, max_args=None),
    "min":   FuncDef(min, min_args=1, max_args=None),

    # Variadic (≥2)
    "gcd":   FuncDef(lambda *args: math.gcd(*(_require_integer(a, label=f"gcd arg {i+1}")
                                                for i, a in enumerate(args))),
                     min_args=2, max_args=None),
    "lcm":   FuncDef(lambda *args: math.lcm(*(_require_integer(a, label=f"lcm arg {i+1}")
                                                for i, a in enumerate(args))),
                     min_args=2, max_args=None),
}


def _validate_function_args(name: str, func_def: FuncDef, n_args: int) -> None:
    """Raise ``CalculatorError`` when *n_args* is out of range for the function."""
    if n_args < func_def.min_args:
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENTS,
            f"{name}() takes at least {func_def.min_args} argument(s), got {n_args}",
        )
    if func_def.max_args is not None and n_args > func_def.max_args:
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENTS,
            f"{name}() takes at most {func_def.max_args} argument(s), got {n_args}",
        )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """A lexical token with source position for diagnostics."""
    type: str          # 'num' | 'const' | 'fn' | 'var' | 'op' | 'lparen' | 'rparen' | 'comma'
    value: Any = None
    lexeme: str = ""
    position: int = -1


def tokenize(expr: str, *, variables: frozenset[str] = frozenset()) -> List[Token]:
    """Scan *expr* into tokens.  Raises ``CalculatorError`` on malformed input.

    *variables* — identifiers to treat as variables (token type ``'var'``)
    rather than rejecting as unknown.
    """
    tokens: List[Token] = []
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]

        if ch.isspace():
            i += 1
            continue

        # Number — int, float, or scientific notation.
        if ch.isdigit() or (ch == '.' and i + 1 < n and expr[i + 1].isdigit()):
            start = i
            # Mantissa: digits and at most one dot.
            dots = 0
            while i < n and (expr[i].isdigit() or expr[i] == '.'):
                if expr[i] == '.':
                    dots += 1
                    if dots > 1:
                        raise CalculatorError(
                            ErrorCode.INVALID_NUMBER,
                            f"Invalid number literal: '{expr[start:i + 1]}'",
                            position=start,
                        )
                i += 1
            # Optional exponent part.
            if i < n and expr[i] in ('e', 'E'):
                i += 1
                if i < n and expr[i] in ('+', '-'):
                    i += 1
                if i == n or not expr[i].isdigit():
                    raise CalculatorError(
                        ErrorCode.INVALID_NUMBER,
                        f"Invalid number literal — missing exponent digits: '{expr[start:i]}'",
                        position=start,
                    )
                while i < n and expr[i].isdigit():
                    i += 1

            lexeme = expr[start:i]
            try:
                # Preserve integer literals as int when no decimal point / exponent.
                if '.' in lexeme or 'e' in lexeme or 'E' in lexeme:
                    val: Numeric = float(lexeme)
                else:
                    val = int(lexeme)
            except (ValueError, OverflowError):
                raise CalculatorError(
                    ErrorCode.INVALID_NUMBER,
                    f"Invalid number literal: '{lexeme}'",
                    position=start,
                )
            tokens.append(Token('num', val, lexeme=lexeme, position=start))
            continue

        # Identifier — function, constant, or variable.
        if ch.isalpha() or ch == '_':
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            name = expr[start:i]
            if name in FUNCTIONS:
                tokens.append(Token('fn', name, lexeme=name, position=start))
            elif name in CONSTANTS:
                tokens.append(Token('const', name, lexeme=name, position=start))
            elif name in variables:
                tokens.append(Token('var', name, lexeme=name, position=start))
            else:
                raise CalculatorError(
                    ErrorCode.UNKNOWN_IDENTIFIER,
                    f"Unknown identifier: '{name}'",
                    position=start,
                )
            continue

        # Single-character tokens.
        if ch in '+-*/^%!':
            tokens.append(Token('op', ch, lexeme=ch, position=i))
            i += 1
        elif ch == '(':
            tokens.append(Token('lparen', lexeme='(', position=i))
            i += 1
        elif ch == ')':
            tokens.append(Token('rparen', lexeme=')', position=i))
            i += 1
        elif ch == ',':
            tokens.append(Token('comma', lexeme=',', position=i))
            i += 1
        else:
            raise CalculatorError(
                ErrorCode.UNEXPECTED_CHAR,
                f"Unexpected character: '{ch}'",
                position=i,
            )

    return tokens


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

class ASTNode:
    """Base class for all AST nodes."""

    def evaluate(self, variables: dict[str, Numeric] | None = None,
                 steps: list[str] | None = None) -> Numeric:
        """Evaluate this node and return a numeric result.

        *variables* — optional mapping of variable name → value.
        *steps* — when provided, evaluation steps are appended to this list.
        """
        raise NotImplementedError

    def to_python(self) -> str:
        """Return a Python expression string that evaluates to this node's value.

        The returned string references function/constant/variable names directly;
        the caller must supply a namespace that resolves those names when the
        string is compiled via :func:`compile_expression`.
        """
        raise NotImplementedError


class Number(ASTNode):
    """Numeric literal."""

    def __init__(self, value: Numeric, lexeme: str = "") -> None:
        self.value = value
        self.lexeme = lexeme

    def evaluate(self, variables=None, steps=None) -> Numeric:
        return self.value

    def to_python(self) -> str:
        if isinstance(self.value, int):
            return str(self.value)
        # Float: use repr for round-trip safety, but special-case inf/nan
        # so they resolve via the namespace (e.g. "Inf" → float('inf')).
        if math.isinf(self.value):
            return "Inf" if self.value > 0 else "(-Inf)"
        if math.isnan(self.value):
            return "NaN"
        return repr(self.value)


class Constant(ASTNode):
    """Named mathematical constant (pi, e, Inf, …)."""

    def __init__(self, name: str, value: Numeric) -> None:
        self.name = name
        self.value = value

    def evaluate(self, variables=None, steps=None) -> Numeric:
        if steps is not None:
            steps.append(f"  {self.name} = {_fmt(self.value)}")
        return self.value

    def to_python(self) -> str:
        return self.name


class Variable(ASTNode):
    """A free variable (e.g. *x* in a plotting expression)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, variables=None, steps=None) -> Numeric:
        if variables is None or self.name not in variables:
            raise CalculatorError(
                ErrorCode.UNKNOWN_IDENTIFIER,
                f"Undefined variable: '{self.name}'",
            )
        return variables[self.name]

    def to_python(self) -> str:
        return self.name


class UnaryOp(ASTNode):
    """Unary plus / minus."""

    def __init__(self, op: str, operand: ASTNode) -> None:
        self.op = op
        self.operand = operand

    def evaluate(self, variables=None, steps=None) -> Numeric:
        inner = self.operand.evaluate(variables, steps)
        if self.op == '-':
            result = -inner
            if steps is not None:
                steps.append(f"  -{_fmt(inner)} = {_fmt(result)}")
            return result
        return inner  # unary plus is identity

    def to_python(self) -> str:
        if self.op == '-':
            return f"(-{self.operand.to_python()})"
        return self.operand.to_python()


class BinaryOp(ASTNode):
    """Binary arithmetic operator: + - * / ^ %"""

    def __init__(self, op: str, left: ASTNode, right: ASTNode) -> None:
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, variables=None, steps=None) -> Numeric:
        left_val = self.left.evaluate(variables, steps)
        right_val = self.right.evaluate(variables, steps)

        if self.op == '+':
            result = left_val + right_val
        elif self.op == '-':
            result = left_val - right_val
        elif self.op == '*':
            result = left_val * right_val
        elif self.op == '/':
            try:
                result = left_val / right_val
            except ZeroDivisionError:
                raise CalculatorError(ErrorCode.DIVIDE_BY_ZERO, "Division by zero")
        elif self.op == '%':
            result = left_val % right_val
        elif self.op == '^':
            try:
                result = left_val ** right_val
            except OverflowError:
                raise CalculatorError(ErrorCode.OVERFLOW,
                                      f"Overflow: {_fmt(left_val)} ^ {_fmt(right_val)}")
            except ValueError as e:
                raise CalculatorError(ErrorCode.DOMAIN_ERROR, str(e))
            if isinstance(result, complex):
                raise CalculatorError(
                    ErrorCode.DOMAIN_ERROR,
                    f"Cannot raise {_fmt(left_val)} to {_fmt(right_val)}: result is complex",
                )
        else:
            raise CalculatorError(ErrorCode.UNEXPECTED_TOKEN,
                                  f"Unexpected operator: '{self.op}'")

        if steps is not None:
            steps.append(f"  {_fmt(left_val)} {self.op} {_fmt(right_val)} = {_fmt(result)}")
        return result

    def to_python(self) -> str:
        left_py = self.left.to_python()
        right_py = self.right.to_python()
        if self.op == '^':
            return f"({left_py} ** {right_py})"
        return f"({left_py} {self.op} {right_py})"


class FunctionCall(ASTNode):
    """Function invocation: ``sin(x)``, ``max(a, b, c)``, …"""

    def __init__(self, name: str, args: list[ASTNode]) -> None:
        self.name = name
        self.args = args

    def evaluate(self, variables=None, steps=None) -> Numeric:
        func_def = FUNCTIONS[self.name]
        arg_vals = [a.evaluate(variables, steps) for a in self.args]
        _validate_function_args(self.name, func_def, len(arg_vals))

        try:
            result = func_def.fn(*arg_vals)
        except ValueError as e:
            raise CalculatorError(ErrorCode.DOMAIN_ERROR, f"{self.name}: {e}")
        except ZeroDivisionError:
            raise CalculatorError(ErrorCode.DIVIDE_BY_ZERO,
                                  f"{self.name}: division by zero")
        except OverflowError:
            raise CalculatorError(ErrorCode.OVERFLOW, f"{self.name}: overflow")

        if steps is not None:
            arg_str = ', '.join(_fmt(a) for a in arg_vals)
            steps.append(f"  {self.name}({arg_str}) = {_fmt(result)}")
        return result

    def to_python(self) -> str:
        arg_py = ', '.join(a.to_python() for a in self.args)
        return f"{self.name}({arg_py})"


class Factorial(ASTNode):
    """Postfix factorial: ``x!``"""

    def __init__(self, operand: ASTNode) -> None:
        self.operand = operand

    def evaluate(self, variables=None, steps=None) -> Numeric:
        val = self.operand.evaluate(variables, steps)
        v_int = _require_integer(val, label="factorial argument")
        try:
            result: Numeric = math.factorial(v_int)
        except ValueError as e:
            raise CalculatorError(ErrorCode.DOMAIN_ERROR, str(e))
        if steps is not None:
            steps.append(f"  {_fmt(val)}! = {_fmt(result)}")
        return result

    def to_python(self) -> str:
        return f"fact({self.operand.to_python()})"


# ---------------------------------------------------------------------------
# Recursive-descent parser (builds AST)
# ---------------------------------------------------------------------------
# Grammar (precedence tightest → loosest):
#   expr     → add_sub
#   add_sub  → mul_div  ( ('+' | '-') mul_div )*
#   mul_div  → power    ( ('*' | '/' | '%') power )*
#   power    → unary    ( '^' power )?            # right-associative
#   unary    → ('+' | '-') unary  |  postfix
#   postfix  → primary  ( '!' )*
#   primary  → NUM | CONST | VAR | FN '(' args ')' | '(' expr ')'
#   args     → expr (',' expr)*


class Parser:
    """Recursive-descent parser that builds an AST."""

    def __init__(self, tokens: List[Token], source: str = "") -> None:
        self.tokens = tokens
        self.pos = 0
        self.source = source

    # -- helpers ---------------------------------------------------------------

    def peek(self) -> Optional[Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type_: str) -> Token:
        tok = self.peek()
        if tok is None:
            raise CalculatorError(
                ErrorCode.MISSING_TOKEN,
                f"Expected '{type_}', but expression ended",
            )
        if tok.type != type_:
            raise CalculatorError(
                ErrorCode.UNEXPECTED_TOKEN,
                f"Expected '{type_}', got '{tok.lexeme}'",
                position=tok.position,
            )
        return self.consume()

    # -- entry point -----------------------------------------------------------

    def parse(self) -> list[ASTNode]:
        """Parse one or more comma-separated expressions.  Returns AST roots."""
        if not self.tokens:
            raise CalculatorError(ErrorCode.BLANK_INPUT, "Expression is blank")
        roots = [self.parse_expr()]
        while (t := self.peek()) and t.type == 'comma':
            self.consume()
            roots.append(self.parse_expr())
        if self.pos < len(self.tokens):
            extra = self.peek()
            raise CalculatorError(
                ErrorCode.UNEXPECTED_TOKEN,
                f"Unexpected token: '{extra.lexeme}'",
                position=extra.position,
            )
        return roots

    # -- grammar rules ---------------------------------------------------------

    def parse_expr(self) -> ASTNode:
        return self.parse_add_sub()

    def parse_add_sub(self) -> ASTNode:
        left = self.parse_mul_div()
        while (t := self.peek()) and t.type == 'op' and t.value in '+-':
            op = self.consume().value
            right = self.parse_mul_div()
            left = BinaryOp(op, left, right)
        return left

    def parse_mul_div(self) -> ASTNode:
        left = self.parse_unary()
        while (t := self.peek()) and t.type == 'op' and t.value in '*/%':
            op = self.consume().value
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_power(self) -> ASTNode:
        """Power — right-associative: ``2^3^2`` → ``2^(3^2)``."""
        left = self.parse_postfix()
        if (t := self.peek()) and t.type == 'op' and t.value == '^':
            self.consume()
            right = self.parse_unary()  # unary so 2^-2 works; unary→power preserves right-assoc
            left = BinaryOp('^', left, right)
        return left

    def parse_unary(self) -> ASTNode:
        """Unary plus/minus — applied after power (lower precedence)."""
        t = self.peek()
        if t and t.type == 'op' and t.value == '-':
            self.consume()
            inner = self.parse_unary()
            return UnaryOp('-', inner)
        if t and t.type == 'op' and t.value == '+':
            self.consume()
            return self.parse_unary()
        return self.parse_power()

    def parse_postfix(self) -> ASTNode:
        """Postfix factorial — tightest binding."""
        node = self.parse_primary()
        while (t := self.peek()) and t.type == 'op' and t.value == '!':
            self.consume()
            node = Factorial(node)
        return node

    def parse_primary(self) -> ASTNode:
        t = self.peek()
        if t is None:
            raise CalculatorError(ErrorCode.MISSING_TOKEN,
                                  "Unexpected end of expression")

        # Number literal
        if t.type == 'num':
            self.consume()
            return Number(t.value, lexeme=t.lexeme)

        # Named constant
        if t.type == 'const':
            self.consume()
            return Constant(t.value, CONSTANTS[t.value])

        # Variable
        if t.type == 'var':
            self.consume()
            return Variable(t.value)

        # Function call
        if t.type == 'fn':
            name = self.consume().value
            self.expect('lparen')
            args: list[ASTNode] = []
            if not (self.peek() and self.peek().type == 'rparen'):
                args.append(self.parse_expr())
                while self.peek() and self.peek().type == 'comma':
                    self.consume()
                    args.append(self.parse_expr())
            self.expect('rparen')
            return FunctionCall(name, args)

        # Parenthesised sub-expression
        if t.type == 'lparen':
            self.consume()
            node = self.parse_expr()
            self.expect('rparen')
            return node

        raise CalculatorError(
            ErrorCode.UNEXPECTED_TOKEN,
            f"Unexpected token: '{t.lexeme}'",
            position=t.position,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_expression(expr: str, *, variables: Iterable[str] = ()) -> list[ASTNode]:
    """Tokenize and parse *expr*, returning a list of AST roots.

    One root per comma-separated expression.  *variables* is an iterable of
    identifier names to treat as free variables rather than rejecting as
    unknown (useful for plotting: ``variables=["x"]``).
    """
    tokens = tokenize(expr, variables=frozenset(variables))
    parser = Parser(tokens, source=expr)
    return parser.parse()


def evaluate_expression(expr: str, *, include_steps: bool = False,
                        variables: dict[str, Numeric] | None = None) -> dict:
    """Parse and evaluate *expr*.  Returns a stable JSON-compatible envelope.

    Single-expression success::

        {"ok": true, "expression": …, "result": {"text": …, "kind": …},
         "error": null, "steps": […]}

    Multi-expression success (comma-separated)::

        {"ok": true, "expression": …, "results": [{"text": …, "kind": …}, …],
         "error": null, "steps": […]}

    Error envelope::

        {"ok": false, "expression": …, "result": null,
         "error": {"code": …, "message": …, "position": …}, "steps": []}

    *variables* — optional values for free variables in the expression
    (e.g. ``{"x": 2.5}``).  When used, variable identifiers are accepted by
    the tokenizer.
    """
    var_names = list(variables.keys()) if variables else []
    try:
        roots = parse_expression(expr, variables=var_names)
        steps: list[str] = [] if include_steps else []
        values = [r.evaluate(variables, steps) for r in roots]

        def _to_result(v: Numeric) -> dict:
            return {
                "text": _fmt(v),
                "kind": "integer" if isinstance(v, int) else "real",
            }

        if len(values) == 1:
            return {
                "ok": True,
                "expression": expr,
                "result": _to_result(values[0]),
                "error": None,
                "steps": steps if include_steps else [],
            }

        return {
            "ok": True,
            "expression": expr,
            "result": None,
            "results": [_to_result(v) for v in values],
            "error": None,
            "steps": steps if include_steps else [],
        }
    except CalculatorError as e:
        return {
            "ok": False,
            "expression": expr,
            "result": None,
            "error": {
                "code": e.code.value,
                "message": e.message,
                "position": e.position,
            },
            "steps": [],
        }


def compile_expression(expr: str, *,
                       variables: Iterable[str] = (),
                       namespace: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Parse *expr* and compile it to a Python callable.

    Returns a function that accepts the named *variables* as keyword arguments
    and evaluates the expression using the callables and constants in
    *namespace*.

    Example::

        >>> import numpy as np
        >>> ns = {"sin": np.sin, "cos": np.cos, "pi": np.pi}
        >>> fn = compile_expression("sin(x) + cos(2*x)", variables=["x"], namespace=ns)
        >>> import numpy as np
        >>> x = np.linspace(-10, 10, 5)
        >>> fn(x)  # returns numpy array

    For immediate (scalar) evaluation use :func:`evaluate_expression` instead.
    """
    var_set = list(variables)
    roots = parse_expression(expr, variables=var_set)

    if len(roots) != 1:
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENTS,
            "compile_expression expects a single expression, "
            f"got {len(roots)} (comma-separated)",
        )

    py_code = roots[0].to_python()
    lambda_code = f"lambda {', '.join(var_set)}: {py_code}"

    ns: dict[str, Any] = {"__builtins__": {}}
    if namespace:
        ns.update(namespace)

    try:
        return eval(lambda_code, ns)
    except SyntaxError as e:
        raise CalculatorError(
            ErrorCode.INVALID_ARGUMENTS,
            f"Failed to compile expression: {e}",
        ) from e
