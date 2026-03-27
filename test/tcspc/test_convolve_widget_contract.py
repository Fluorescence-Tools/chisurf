from __future__ import annotations

import ast
from pathlib import Path


def test_convolve_widget_exposes_on_unload_irf_handler_contract():
    path = Path("chisurf/models/tcspc/widgets/convolve.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConvolveWidget"
    )
    method_names = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "onUnloadIRF" in method_names


def test_lifetime_widget_exposes_button_handler_contracts():
    path = Path("chisurf/models/tcspc/widgets/lifetime.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LifetimeWidget"
    )
    method_names = {
        node.name
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "onAddLifetime" in method_names
    assert "onRemoveLifetime" in method_names
    assert "onAbsoluteAmplitudes" in method_names
