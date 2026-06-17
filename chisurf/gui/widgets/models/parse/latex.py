"""Utilities for converting parse-model formulas to LaTeX."""

from __future__ import annotations

import ast
import re

from latexify import ast_utils
from latexify.codegen import ExpressionCodegen


def sanitize_latex_for_mathtext(latex: str) -> str:
    """Return LaTeX that Matplotlib's mathtext parser can render.

    Parameters
    ----------
    latex : str
        LaTeX expression produced by a converter.

    Returns
    -------
    str
        LaTeX expression with commands unsupported by Matplotlib mathtext
        rewritten or removed.
    """
    if not latex:
        return latex

    result = re.sub(r"\\mathopen\s*\{\}", "", latex)
    result = re.sub(r"\\mathclose\s*\{\}", "", result)
    return _replace_vfrac(result)


def _replace_vfrac(latex: str) -> str:
    """Replace latexify's vertical fraction command with a normal fraction."""
    result: list[str] = []
    index = 0

    while index < len(latex):
        if not latex.startswith("\\vfrac", index):
            result.append(latex[index])
            index += 1
            continue

        index += len("\\vfrac")
        first_group, index = _parse_latex_group(latex, index)
        numerator, index = _parse_latex_group(latex, index)
        denominator, index = _parse_latex_group(latex, index)

        if first_group is None or numerator is None or denominator is None:
            result.append("\\vfrac")
            continue

        result.append(f"\\frac{{{numerator}}}{{{denominator}}}")

    return "".join(result)


def _parse_latex_group(latex: str, start: int) -> tuple[str | None, int]:
    """Parse the next braced LaTeX group.

    Parameters
    ----------
    latex : str
        LaTeX expression.
    start : int
        Index where the group search should begin.

    Returns
    -------
    tuple[str | None, int]
        Parsed group content without the surrounding braces and the index after
        the group, or ``(None, start)`` if no group is present.
    """
    index = start
    while index < len(latex) and latex[index].isspace():
        index += 1

    if index >= len(latex) or latex[index] != "{":
        return None, start

    depth = 0
    while index < len(latex):
        if latex[index] == "\\":
            index += 2
            continue
        if latex[index] == "{":
            depth += 1
        elif latex[index] == "}":
            depth -= 1
            if depth == 0:
                return latex[start + 1 : index], index + 1
        index += 1

    return None, start


def convert_python_expression_to_latex(expression: str) -> str:
    r"""Convert a Python expression string to LaTeX.

    Parameters
    ----------
    expression : str
        Python expression to convert. NumPy-qualified calls such as
        ``np.exp(x)`` are accepted by latexify and rendered as the underlying
        mathematical function.

    Returns
    -------
    str
        LaTeX expression, or the original expression if conversion fails.

    Examples
    --------
    >>> convert_python_expression_to_latex("a / (b + c)")
    '\\frac{a}{b + c}'
    """
    expression = (expression or "").strip()
    if not expression:
        return expression

    try:
        node = ast_utils.parse_expr(expression)
        return ExpressionCodegen(use_math_symbols=True).visit(node)
    except Exception:
        return _convert_python_expression_to_latex_fallback(expression)


def _convert_python_expression_to_latex_fallback(expression: str) -> str:
    """Convert a Python expression using a small local AST fallback.

    Parameters
    ----------
    expression : str
        Python expression to convert.

    Returns
    -------
    str
        Best-effort LaTeX expression, or the original expression if parsing
        fails.
    """
    clean_expression = expression.replace("np.", "").strip()
    if not clean_expression:
        return clean_expression

    func_latex_map = {
        "sin": r"\sin",
        "cos": r"\cos",
        "tan": r"\tan",
        "asin": r"\arcsin",
        "acos": r"\arccos",
        "atan": r"\arctan",
        "sinh": r"\sinh",
        "cosh": r"\cosh",
        "tanh": r"\tanh",
        "asinh": r"\mathrm{asinh}",
        "acosh": r"\mathrm{acosh}",
        "atanh": r"\mathrm{atanh}",
        "log": r"\log",
        "erf": r"\mathrm{erf}",
    }
    const_map = {
        "pi": r"\pi",
        "E": "e",
        "I": "i",
        "inf": r"\infty",
        "nan": r"\mathrm{NaN}",
    }

    def convert(node: ast.AST, in_pow_base: bool = False, parent_op: type[ast.operator] | None = None) -> str:
        if isinstance(node, ast.Expression):
            return convert(node.body)

        if isinstance(node, ast.BinOp):
            left = convert(node.left, parent_op=type(node.op))
            right = convert(node.right, parent_op=type(node.op))
            if isinstance(node.op, ast.Add):
                result = f"{left} + {right}"
            elif isinstance(node.op, ast.Sub):
                result = f"{left} - {right}"
            elif isinstance(node.op, ast.Mult):
                result = rf"{left} \, {right}"
            elif isinstance(node.op, ast.Div):
                result = rf"\frac{{{left}}}{{{right}}}"
            elif isinstance(node.op, ast.Pow):
                base = convert(node.left, in_pow_base=True, parent_op=type(node.op))
                exponent = convert(node.right, parent_op=type(node.op))
                result = f"{{{base}}}^{{{exponent}}}"
            elif isinstance(node.op, ast.Mod):
                result = rf"{left} \bmod {right}"
            else:
                result = rf"({left} \circ {right})"

            needs_wrap = in_pow_base or (
                parent_op in (ast.Mult, ast.Div)
                and isinstance(node.op, (ast.Add, ast.Sub))
            )
            return rf"\left({result}\right)" if needs_wrap else result

        if isinstance(node, ast.UnaryOp):
            operand = convert(node.operand)
            if isinstance(node.op, ast.USub):
                return f"-{operand}"
            if isinstance(node.op, ast.UAdd):
                return operand
            return f"({operand})"

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            else:
                func_name = None

            args = [convert(arg) for arg in node.args]
            if func_name == "sqrt" and args:
                return rf"\sqrt{{{args[0]}}}" if len(args) == 1 else rf"\sqrt[{args[1]}]{{{args[0]}}}"
            if func_name == "abs" and args:
                return rf"\left|{args[0]}\right|"
            if func_name == "exp" and len(args) == 1:
                return f"e^{{{args[0]}}}"
            if func_name == "factorial" and args:
                return f"{{{args[0]}}}!"
            if func_name == "gamma":
                return r"\Gamma(" + ", ".join(args) + ")"

            latex_name = func_latex_map.get(func_name)
            if latex_name:
                return f"{latex_name}(" + ", ".join(args) + ")"
            if func_name is not None:
                return rf"\mathrm{{{func_name}}}(" + ", ".join(args) + ")"
            return f"{convert(node.func)}(" + ", ".join(args) + ")"

        if isinstance(node, ast.Name):
            return const_map.get(node.id, node.id)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            if node.value is None:
                return r"\mathrm{None}"
            return str(node.value)

        if isinstance(node, (ast.List, ast.Tuple)):
            elements = [convert(el) for el in node.elts]
            start, end = ("[", "]") if isinstance(node, ast.List) else ("(", ")")
            return start + ", ".join(elements) + end

        if isinstance(node, ast.Attribute):
            return node.attr

        return ""

    try:
        tree = ast.parse(clean_expression, mode="eval")
        result = convert(tree)
    except Exception:
        return expression

    return result if result else expression
