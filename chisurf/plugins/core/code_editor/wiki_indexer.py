from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class APISymbol:
    """Metadata for a ChiSurf API symbol extracted from source."""

    name: str
    qualname: str
    module: str
    path: str
    line: int
    kind: str
    signature: str
    docstring: str | None
    source: str


class ChisurfAPIIndexer:
    """Build an AST-based API index for the ChiSurf codebase."""

    def __init__(self, repo_root: str | Path | None = None) -> None:
        """Initialize the indexer.

        Parameters
        ----------
        repo_root : str or pathlib.Path, optional
            Repository root. Defaults to the parent of the ``chisurf`` package.
        """
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[4]).resolve()
        self.package_root = self.repo_root / "chisurf"
        self.wiki_root = self.repo_root / "llm-wiki" / "wiki" / "api"

    @property
    def index_path(self) -> Path:
        """Return the JSON API index path."""
        return self.wiki_root / "index.json"

    def build(self) -> Path:
        """Build and write the API index.

        Returns
        -------
        pathlib.Path
            Path to the generated JSON index.
        """
        symbols = [asdict(symbol) for symbol in self.iter_symbols()]
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(self.repo_root),
            "symbols": symbols,
        }
        self.wiki_root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._write_examples_markdown(symbols)
        return self.index_path

    def iter_symbols(self) -> list[APISymbol]:
        """Extract public functions, classes, and methods from ChiSurf source files.

        Returns
        -------
        list[APISymbol]
            Extracted API symbol metadata.
        """
        symbols: list[APISymbol] = []
        for path in sorted(self.package_root.rglob("*.py")):
            if "__pycache__" in path.parts or any(part == "test" for part in path.parts):
                continue
            module = self._module_name(path)
            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (OSError, SyntaxError):
                continue

            lines = source.splitlines()
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"):
                        continue
                    class_symbol = self._class_symbol(node, module, path, lines)
                    symbols.append(class_symbol)
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name.startswith("_") and child.name != "__init__":
                                continue
                            symbols.append(
                                self._function_symbol(
                                    child,
                                    module,
                                    path,
                                    lines,
                                    class_symbol.qualname,
                                )
                            )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_"):
                        continue
                    symbols.append(self._function_symbol(node, module, path, lines))
        return symbols

    def _module_name(self, path: Path) -> str:
        """Return the dotted module name for a source path."""
        rel = path.relative_to(self.package_root).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _class_symbol(
        self,
        node: ast.ClassDef,
        module: str,
        path: Path,
        lines: list[str],
    ) -> APISymbol:
        """Create metadata for a class node."""
        return APISymbol(
            name=node.name,
            qualname=f"{module}.{node.name}",
            module=module,
            path=self._display_path(path),
            line=node.lineno,
            kind="class",
            signature=f"class {node.name}",
            docstring=ast.get_docstring(node),
            source=self._source_snippet(lines, node),
        )

    def _function_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module: str,
        path: Path,
        lines: list[str],
        class_qualname: str | None = None,
    ) -> APISymbol:
        """Create metadata for a function or method node."""
        qualname = f"{class_qualname}.{node.name}" if class_qualname else f"{module}.{node.name}"
        return APISymbol(
            name=node.name,
            qualname=qualname,
            module=module,
            path=self._display_path(path),
            line=node.lineno,
            kind="method" if class_qualname else "function",
            signature=self._signature(node),
            docstring=ast.get_docstring(node),
            source=self._source_snippet(lines, node),
        )

    def _signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Return a compact source signature for a function node."""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = ""
        return f"{prefix} {node.name}({args})"

    def _source_snippet(self, lines: list[str], node: ast.AST) -> str:
        """Return a bounded source snippet around an AST node."""
        start = max(0, node.lineno - 1)
        end = min(len(lines), start + 80)
        return "\n".join(lines[start:end])

    def _display_path(self, path: Path) -> str:
        """Return a repo-relative path string."""
        try:
            return path.relative_to(self.repo_root).as_posix()
        except ValueError:
            return path.as_posix()

    def _write_examples_markdown(self, symbols: list[dict[str, Any]]) -> None:
        """Write a compact markdown overview of indexed API symbols."""
        content = ["# ChiSurf API Index", "", "Generated API symbols from the current source tree.", ""]
        for symbol in symbols[:300]:
            content.append(f"- `{symbol['qualname']}` — {symbol['path']}:{symbol['line']}")
        self.wiki_root.mkdir(parents=True, exist_ok=True)
        (self.wiki_root / "examples.md").write_text("\n".join(content), encoding="utf-8")


def build_api_index(repo_root: str | Path | None = None) -> Path:
    """Build the ChiSurf API index.

    Parameters
    ----------
    repo_root : str or pathlib.Path, optional
        Repository root.

    Returns
    -------
    pathlib.Path
        Path to the generated JSON index.
    """
    return ChisurfAPIIndexer(repo_root).build()
