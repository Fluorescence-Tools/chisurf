from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass


@dataclass
class CodeSymbol:
    """Navigable code symbol extracted from an editor document."""

    name: str
    kind: str
    line: int
    column: int = 0
    end_line: int | None = None
    parent: str = ""
    path: str = ""

    @property
    def display_name(self) -> str:
        """Return the symbol label shown in navigation widgets."""
        prefix = f"{self.parent}." if self.parent else ""
        return f"{prefix}{self.name}"


class _PythonSymbolVisitor(ast.NodeVisitor):
    """Collect Python symbols while preserving class nesting."""

    def __init__(self, path: str = "") -> None:
        self.path = path
        self.symbols: list[CodeSymbol] = []
        self._parents: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        parent = ".".join(self._parents)
        self.symbols.append(
            CodeSymbol(
                name=node.name,
                kind="class",
                line=node.lineno,
                column=node.col_offset,
                end_line=getattr(node, "end_lineno", None),
                parent=parent,
                path=self.path,
            )
        )
        self._parents.append(node.name)
        self.generic_visit(node)
        self._parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)
        self.generic_visit(node)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = ".".join(self._parents)
        self.symbols.append(
            CodeSymbol(
                name=node.name,
                kind="method" if parent else "function",
                line=node.lineno,
                column=node.col_offset,
                end_line=getattr(node, "end_lineno", None),
                parent=parent,
                path=self.path,
            )
        )


def extract_python_symbols(text: str, path: str = "") -> list[CodeSymbol]:
    """Extract navigable Python symbols from *text*."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _extract_python_symbols_with_regex(text, path)

    visitor = _PythonSymbolVisitor(path)
    visitor.visit(tree)
    symbols = visitor.symbols
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            title = stripped[4:].strip() or "Section"
            symbols.append(CodeSymbol(title, "section", line_number, 0, path=path))
    return sorted(symbols, key=lambda symbol: (symbol.line, symbol.column))


def _extract_python_symbols_with_regex(text: str, path: str = "") -> list[CodeSymbol]:
    """Extract symbols from incomplete Python source using simple line matching."""
    symbols: list[CodeSymbol] = []
    class_stack: list[tuple[int, str]] = []
    pattern = re.compile(r"^(\s*)(class|def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)")
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            title = stripped[4:].strip() or "Section"
            symbols.append(CodeSymbol(title, "section", line_number, 0, path=path))
            continue
        match = pattern.match(line)
        if match is None:
            continue
        indent = len(match.group(1))
        keyword = match.group(2)
        name = match.group(3)
        while class_stack and class_stack[-1][0] >= indent:
            class_stack.pop()
        parent = ".".join(item[1] for item in class_stack)
        if keyword == "class":
            symbols.append(CodeSymbol(name, "class", line_number, indent, parent=parent, path=path))
            class_stack.append((indent, name))
        else:
            kind = "method" if parent else "function"
            symbols.append(CodeSymbol(name, kind, line_number, indent, parent=parent, path=path))
    return symbols


def find_project_root(path: str | pathlib.Path | None = None) -> pathlib.Path:
    """Return the nearest project root for *path*."""
    start = pathlib.Path(path or pathlib.Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return start


__all__ = ["CodeSymbol", "extract_python_symbols", "find_project_root"]
