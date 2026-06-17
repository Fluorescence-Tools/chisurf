"""Tests for parse-model expression LaTeX conversion."""

from __future__ import annotations

import pytest

from chisurf.gui.widgets.models.parse.latex import (
    convert_python_expression_to_latex,
    sanitize_latex_for_mathtext,
)


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


def test_sanitize_latex_for_mathtext_rewrites_latexify_output() -> None:
    """LaTeX emitted by latexify is made compatible with Matplotlib mathtext."""
    expression = (
        r"b + \vfrac{}{1}{\mathopen{}\left( N \mathclose{}\right)} "
        r"\mathopen{}\left(1 + \frac{x}{\mathopen{}\left(\mathbf{td}\right)^2 - 1}\mathclose{}\right) "
        r"\mathopen{}\left(1 + \mathbf{t} \mathopen{}\left(\frac{1}{\mathbf{s}^2(2\mathbf{x})}\mathclose{}\right) "
        r"\mathopen{}\left(\mathbf{a} \cdot \exp \mathopen{}\left(\frac{\mathbf{t}}{\mathbf{s}}\right)\mathclose{}\right)\right)"
    )

    assert sanitize_latex_for_mathtext(expression) == (
        r"b + \frac{1}{\left( N \right)} "
        r"\left(1 + \frac{x}{\left(\mathbf{td}\right)^2 - 1}\right) "
        r"\left(1 + \mathbf{t} \left(\frac{1}{\mathbf{s}^2(2\mathbf{x})}\right) "
        r"\left(\mathbf{a} \cdot \exp \left(\frac{\mathbf{t}}{\mathbf{s}}\right)\right)\right)"
    )
