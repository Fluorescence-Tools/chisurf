"""Tests for parse-model expression LaTeX conversion."""

from __future__ import annotations

import pytest

from chisurf.gui.widgets.models.parse.latex import convert_python_expression_to_latex


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("a / (b + c)", r"\frac{a}{b + c}"),
        ("np.exp(-x / tau)", r"\exp \mathopen{}\left( \frac{-x}{\tau} \mathclose{}\right)"),
        ("sqrt(x**2 + y**2)", r"\sqrt{ x^{2} + y^{2} }"),
        ("sin(alpha) + cos(beta)", r"\sin \alpha + \cos \beta"),
    ],
)
def test_convert_python_expression_to_latex(expression: str, expected: str) -> None:
    """Expression strings are converted with latexify-py."""
    assert convert_python_expression_to_latex(expression) == expected


def test_convert_python_expression_to_latex_keeps_invalid_expression() -> None:
    """Invalid Python expressions fall back to their original text."""
    expression = "a +"

    assert convert_python_expression_to_latex(expression) == expression
