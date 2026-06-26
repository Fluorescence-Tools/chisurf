"""Architectural boundary: the data/spec layers must stay GUI-free.

The model layer (computation, equations, parameters) and the declarative
editor vocabulary (:mod:`chisurf.core.dataspec`, lifted out under PRD-40) are
kept strictly separate from the presentation layer (Qt widgets, plots). An
object describes its editor through plain ``dataspec`` data and never imports a
GUI toolkit. This test enforces that boundary statically via AST so a stray
``import qtpy`` (or ``import chisurf.gui``) fails CI instead of silently
re-coupling the two sides.
"""
from __future__ import annotations

import ast
import pathlib

import chisurf.core.dataspec
import chisurf.core.models

#: Module prefixes that the data/spec layers are forbidden to import.
FORBIDDEN_PREFIXES = ("qtpy", "PyQt5", "PyQt6", "PySide2", "PySide6", "chisurf.gui")

#: Package roots that must stay GUI-free.
GUI_FREE_ROOTS = {
    "chisurf/core/models": pathlib.Path(chisurf.core.models.__file__).parent,
    "chisurf/core/dataspec": pathlib.Path(chisurf.core.dataspec.__file__).parent,
}


def _forbidden_imports(source: str, filename: str):
    """Yield ``(lineno, module)`` for every forbidden import in ``source``."""
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_PREFIXES):
                    yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith(FORBIDDEN_PREFIXES):
                yield node.lineno, module


def test_core_data_layers_do_not_import_gui():
    """No file under the GUI-free roots imports a GUI toolkit or chisurf.gui."""
    violations = []
    for root in GUI_FREE_ROOTS.values():
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for lineno, module in _forbidden_imports(source, str(path)):
                violations.append(f"{path}:{lineno}: imports '{module}'")

    assert not violations, (
        "chisurf/core/models and chisurf/core/dataspec must not depend on any "
        "GUI toolkit (strict compute/UI split). Offending imports:\n  "
        + "\n  ".join(violations)
    )
