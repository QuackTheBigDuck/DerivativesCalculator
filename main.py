import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

x, y, t, theta = sp.symbols("x y t theta")

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

NAMES = {
    "e": sp.E,
    "pi": sp.pi,
    "ln": sp.log,
    "arcsin": sp.asin,
    "arccos": sp.acos,
    "arctan": sp.atan,
    "arcsec": sp.asec,
    "arccsc": sp.acsc,
    "arccot": sp.acot,
    "abs": sp.Abs,
}
for _name in ("x", "y", "t", "theta", "u", "a", "b", "c", "k", "n", "r"):
    NAMES[_name] = sp.Symbol(_name)


def parse(text):
    return parse_expr(text.strip(), local_dict=dict(NAMES), transformations=TRANSFORMS)


def _collect_steps(f, v, out):
    if not f.has(v):
        out.append(f"Constant rule:  d/d{v}[{f}] = 0")
        return

    if f == v:
        out.append(f"Identity rule:  d/d{v}[{v}] = 1")
        return

    if f.is_Add:
        out.append(f"Sum rule:  differentiate each term of  {f}  separately")
        for term in f.args:
            _collect_steps(term, v, out)
        return

    if f.is_Mul:
        num, den = sp.fraction(f)
        if den.has(v):
            out.append(
                f"Quotient rule with  u = {num},  w = {den}:\n"
                f"          d/d{v}[u/w] = (u'w - u w') / w^2"
            )
            _collect_steps(num, v, out)
            _collect_steps(den, v, out)
            return

        const, rest = f.as_independent(v)
        if const != 1:
            out.append(f"Constant multiple rule:  pull out the constant {const}")
            _collect_steps(rest, v, out)
            return

        u = f.args[0]
        w = sp.Mul(*f.args[1:])
        out.append(
            f"Product rule with  u = {u},  w = {w}:\n"
            f"          d/d{v}[u w] = u'w + u w'"
        )
        _collect_steps(u, v, out)
        _collect_steps(w, v, out)
        return

    if f.is_Pow:
        base, exp = f.args
        if not exp.has(v):
            if base == v:
                out.append(f"Power rule:  d/d{v}[{v}^n] = n {v}^(n-1),  here n = {exp}")
            else:
                out.append(
                    f"Power rule + chain rule:  d/d{v}[u^n] = n u^(n-1) u',\n"
                    f"          here u = {base},  n = {exp}"
                )
                _collect_steps(base, v, out)
        elif not base.has(v):
            if base == sp.E:
                out.append(f"Exponential rule:  d/d{v}[e^u] = e^u u',  here u = {exp}")
            else:
                out.append(
                    f"Exponential rule:  d/d{v}[a^u] = a^u ln(a) u',\n"
                    f"          here a = {base},  u = {exp}"
                )
            if exp != v:
                _collect_steps(exp, v, out)
        else:
            out.append(
                f"Logarithmic differentiation (variable in base AND exponent):\n"
                f"          y = {base}^({exp})  ->  ln y = ({exp}) ln({base})\n"
                f"          y'/y = d/d{v}[({exp}) ln({base})]"
            )
            _collect_steps(exp, v, out)
            _collect_steps(base, v, out)
        return

    if len(f.args) == 1:
        inner = f.args[0]
        U = sp.Symbol("u")
        outer = sp.diff(f.func(U), U)
        fname = f.func.__name__
        if inner == v:
            out.append(f"Known derivative:  d/d{v}[{f}] = {outer.subs(U, v)}")
        else:
            out.append(
                f"Chain rule:  d/d{v}[{fname}(u)] = ({outer}) u',  where u = {inner}"
            )
            _collect_steps(inner, v, out)
        return

    out.append(f"Differentiate  {f}  directly")


def show_steps(f, v):
    steps = []
    _collect_steps(f, v, steps)
    print()
    print(f"Find  d/d{v} [ {f} ]")
    print()
    for i, s in enumerate(steps, 1):
        print(f"Step {i}:  {s}")
        print()
    raw = sp.diff(f, v)
    ans = sp.simplify(raw)
    print(f"Step {len(steps) + 1}:  Combine and simplify")
    print()
    print(f"          {raw}")
    print()
    print(f"Answer:   d/d{v}[{f}] = {ans}")
    print()
    return ans


def basic(expr_text, var="x"):
    v = sp.Symbol(var)
    return show_steps(parse(expr_text), v)


def higher_order(expr_text, n, var="x"):
    v = sp.Symbol(var)
    f = parse(expr_text)
    print()
    current = f
    for k in range(1, n + 1):
        current = sp.simplify(sp.diff(current, v))
        print(f"Step {k}:  derivative number {k}")
        print()
        print(f"          f^({k})({var}) = {current}")
        print()
    return current


def implicit(eq_text):
    if "=" in eq_text:
        lhs, rhs = eq_text.split("=", 1)
        F = parse(lhs) - parse(rhs)
    else:
        F = parse(eq_text)
    Fx, Fy = sp.diff(F, x), sp.diff(F, y)
    dydx = sp.simplify(-Fx / Fy)
    d2 = sp.simplify(sp.diff(dydx, x) + sp.diff(dydx, y) * dydx)
    print()
    print(f"Write the curve as  F(x, y) = {F} = 0")
    print()
    print(f"Step 1:  Partial derivative in x:   F_x = {Fx}")
    print()
    print(f"Step 2:  Partial derivative in y:   F_y = {Fy}")
    print()
    print("Step 3:  Use  dy/dx = -F_x / F_y")
    print()
    print(f"Answer:  dy/dx = {dydx}")
    print()
    print("Step 4:  Differentiate again, replacing y' with dy/dx")
    print()
    print(f"Answer:  d2y/dx2 = {d2}")
    print()
    return dydx


def parametric(x_text, y_text):
    X, Y = parse(x_text), parse(y_text)
    dx, dy = sp.diff(X, t), sp.diff(Y, t)
    dydx = sp.simplify(dy / dx)
    d2 = sp.simplify(sp.diff(dydx, t) / dx)
    print()
    print(f"Step 1:  dx/dt = {dx}")
    print()
    print(f"Step 2:  dy/dt = {dy}")
    print()
    print("Step 3:  dy/dx = (dy/dt) / (dx/dt)")
    print()
    print(f"Answer:  dy/dx = {dydx}")
    print()
    print("Step 4:  d2y/dx2 = [ d/dt (dy/dx) ] / (dx/dt)")
    print()
    print(f"Answer:  d2y/dx2 = {d2}")
    print()
    return dydx


def polar(r_text):
    r = parse(r_text)
    rp = sp.diff(r, theta)
    num = rp * sp.sin(theta) + r * sp.cos(theta)
    den = rp * sp.cos(theta) - r * sp.sin(theta)
    dydx = sp.simplify(num / den)
    print()
    print(f"Step 1:  r = {r}")
    print()
    print(f"Step 2:  dr/dtheta = {rp}")
    print()
    print("Step 3:  Use x = r cos(theta), y = r sin(theta):")
    print("          dy/dx = (r' sin(theta) + r cos(theta)) / (r' cos(theta) - r sin(theta))")
    print()
    print(f"Answer:  dy/dx = {dydx}")
    print()
    return dydx


def log_diff(expr_text, var="x"):
    v = sp.Symbol(var)
    f = parse(expr_text)
    lnf = sp.expand_log(sp.log(f), force=True)
    dlnf = sp.diff(lnf, v)
    result = sp.simplify(f * dlnf)
    print()
    print(f"Step 1:  Let y = {f}")
    print()
    print(f"Step 2:  Take ln of both sides and expand:   ln y = {lnf}")
    print()
    print(f"Step 3:  Differentiate both sides:   y'/y = {dlnf}")
    print()
    print("Step 4:  Multiply by y:   y' = y * (y'/y)")
    print()
    print(f"Answer:  dy/d{var} = {result}")
    print()
    return result


def at_point(expr_text, a_text, var="x"):
    v = sp.Symbol(var)
    f = parse(expr_text)
    a = parse(a_text)
    fp = sp.diff(f, v)
    slope = sp.simplify(fp.subs(v, a))
    y0 = sp.simplify(f.subs(v, a))
    print()
    print(f"Step 1:  f'({var}) = {sp.simplify(fp)}")
    print()
    print(f"Step 2:  Slope at {var} = {a}:   f'({a}) = {slope}   (about {sp.N(slope, 6)})")
    print()
    print(f"Step 3:  Point on curve:   f({a}) = {y0}")
    print()
    line = sp.simplify(y0 + slope * (v - a))
    print(f"Answer:  Tangent line:   y = {line}")
    print()
    return slope


MENU = """
==========================================
   DERIVATIVE CALCULATOR  (Calc 1 & 2)
==========================================
 1) Derivative with steps
 2) Higher-order derivative
 3) Derivative w.r.t. another variable (partial)
 4) Implicit differentiation
 5) Parametric  (x(t), y(t))
 6) Polar  r = f(theta)
 7) Logarithmic differentiation
 8) Slope at a point + tangent line
 q) Quit
"""


def main():
    while True:
        print(MENU)
        choice = input("Choose: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("Bye!")
            break
        try:
            if choice == "1":
                basic(input("f(x) = "))
            elif choice == "2":
                f = input("f(x) = ")
                n = int(input("Order n = "))
                higher_order(f, n)
            elif choice == "3":
                f = input("f = ")
                var = input("Differentiate with respect to: ").strip() or "x"
                basic(f, var)
            elif choice == "4":
                implicit(input("Equation (e.g. x^2 + y^2 = 25): "))
            elif choice == "5":
                parametric(input("x(t) = "), input("y(t) = "))
            elif choice == "6":
                polar(input("r(theta) = "))
            elif choice == "7":
                log_diff(input("f(x) = "))
            elif choice == "8":
                f = input("f(x) = ")
                a = input("x = ")
                at_point(f, a)
            else:
                print("Please pick 1-8 or q.")
        except Exception as err:
            print(f"\nCouldn't do that one: {err}\n")


if __name__ == "__main__":
    main()