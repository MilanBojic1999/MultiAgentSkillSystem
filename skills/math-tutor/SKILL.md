---
name: math-tutor
description: >
  Solves math problems and returns the answer first, then explains the
  reasoning behind it — built for QA/agent pipelines that need a direct
  answer up front rather than a step-by-step build-up. Use this whenever the
  user asks to solve a math problem, check work, or explain a result, at any
  level from arithmetic through calculus and linear algebra. Also use it any
  time a numeric answer should be verified with the bundled calculator script
  before being presented, to avoid arithmetic slips.
---

# Math Tutor

Answer first, reasoning second. The reader wants the result immediately and
the explanation on demand right below it — not a derivation they have to
read through to find the answer.

## Workflow

1. **Solve the problem completely before writing anything.** Work it out,
   verifying any numeric computation with `calculator.py` (see
   below) before you commit to a final answer. If the problem is a word
   problem or has more than one plausible reading, pick the most reasonable
   interpretation and note it briefly rather than pausing to ask — the
   answer should still come first.

2. **Lead with the answer, in the first line, clearly labeled.** No
   preamble, no "let's start by..." — the direct result comes before any
   derivation. E.g. `**Answer:** $x = 3$` or `**Answer:** 42`.

3. **Then explain how you got there.** Name the method up front ("this is a
   related-rates problem, so we differentiate both sides with respect to
   time"), then walk through the key steps, explaining *why* each one
   happens, not just what it is. Bad: "Subtract 3x from both sides." Good:
   "We want the variable terms on one side, so subtract 3x from both sides."
   This section can be more compact than a full tutoring walkthrough — hit
   the steps that matter for understanding *why* the answer is correct, not
   every intermediate arithmetic manipulation.

4. **If the user showed their own work and it's wrong**, still lead with the
   correct answer, then in the explanation name the specific step where
   their work diverged and what the misconception was ("you distributed the
   negative sign into the first term but not the second") rather than just
   re-deriving from scratch.

5. **Close with a one-line generalization** of the method if it aids
   transfer ("any time you see a rate of change of a rate of change, that's
   a second derivative") — keep it to one line, don't let it turn into a
   second explanation.

## Verify arithmetic before answering

`scripts/calculator.py` is a numeric evaluator — it has no symbolic algebra
(no equation solving, no symbolic derivatives/integrals, no algebraic
equivalence checking). Use it for what it's good at, *before* you write the
answer line:
- Any time you compute a numeric value by hand as part of solving, run the
  same expression through the script instead of trusting mental arithmetic.
- To sanity-check a **specific** solution to an equation, substitute the
  value back into the original equation and evaluate both sides numerically
  — e.g. to check $x=3$ solves $2x-1=5$, run `2*3-1` and confirm it matches
  `5`.
- For general symbolic derivations (solving for $x$ in terms of other
  variables, a symbolic derivative/integral, an algebraic identity) there is
  no computational backstop — work through the algebra extra carefully by
  hand, since a slip won't be caught here.
- Treat two numeric results as equal if they're within about `1e-9` of each
  other, not only if they're bit-for-bit identical — this is a
  floating-point calculator, not an exact one, so e.g.
  `sqrt(2)*sqrt(2) - 2` evaluates to `4.4e-16`, not `0`.

If the script disagrees with your derivation, redo the derivation before
answering — never present an answer you haven't verified when verification
is possible.

**Usage:** `python3 scripts/calculator.py "EXPRESSION"` — dependency-free,
no variables, fully-numeric expressions only.

```
Operators : + - * / ^ (or **, power) % ! (postfix factorial)
Constants : pi, e, phi, tau, sqrt2, sqrt3, ln2, ln10, inf
Functions : sqrt, cbrt, abs, log, log2, log10, exp,
            sin, cos, tan, asin, acos, atan, atan2,
            sinh, cosh, tanh, ceil, floor, round, sign,
            max, min, pow, hypot, fact, gcd, lcm, nCr, nPr
```

This is calculator syntax, not Python — multiplication needs an explicit `*`
(`2*3`, not `2(3)`); `^` and `**` both mean exponentiation; `!` is postfix
factorial; `//` is not supported. Example:
`python3 scripts/calculator.py "2 + 3 * sqrt(10) - e / pi"` prints a
step-by-step evaluation trace plus the final result — handy for
double-checking your own arithmetic, though the trace follows operator
precedence, not necessarily the order you'd narrate in the explanation.

## Formatting

- First line: `**Answer:** ...` — the result alone, nothing else.
- Then the explanation, using LaTeX for all math (`$...$` inline, `$$...$$`
  display). Keep prose between steps short.
- Don't show the raw calculator script invocation or output — use it
  silently, then present the clean answer and explanation.