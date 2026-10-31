# Derivative Calculator

Command-line derivative solver for Calc 1 and 2, with numbered step-by-step rules.

## Setup

```
pip install sympy
python derivative_calculator.py
```

## Features

1. Derivative with steps
2. Higher-order derivatives
3. Derivative with respect to another variable (partial)
4. Implicit differentiation
5. Parametric curves
6. Polar curves
7. Logarithmic differentiation
8. Slope at a point and tangent line

## Input

- Powers: `^` or `**`
- Implicit multiplication: `3x` means `3*x`
- Functions: `ln(x)`, `e^x`, `sin(x)`, `arcsin(x)`, `sec(x)`
- Implicit mode takes a full equation, e.g. `x^2 + y^2 = 25`

## Use as a module

```python
import derivative_calculator as d

d.basic("x^2 sin(3x)")
d.parametric("t^2", "t^3 - 3t")
```
