#!/usr/bin/env python

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class FittingParameterVisitor(ast.NodeVisitor):
    """AST visitor that records calls to FittingParameter()."""

    def __init__(self, module: str, rel_path: str) -> None:
        self.module = module
        self.rel_path = rel_path
        self._stack: List[Dict[str, str]] = []
        self.results: List[Dict[str, Any]] = []

    # context helpers -----------------------------------------------------
    def _push(self, kind: str, name: str) -> None:
        self._stack.append({"kind": kind, "name": name})

    def _pop(self) -> None:
        if self._stack:
            self._stack.pop()

    def _current(self, kind: str) -> Optional[str]:
        for frame in reversed(self._stack):
            if frame.get("kind") == kind:
                return frame.get("name")
        return None

    # node visitors -------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # type: ignore[override]
        self._push("class", node.name)
        self.generic_visit(node)
        self._pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # type: ignore[override]
        self._push("func", node.name)
        self.generic_visit(node)
        self._pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # type: ignore[override]
        self._push("func", node.name)
        self.generic_visit(node)
        self._pop()

    def visit_Call(self, node: ast.Call) -> Any:  # type: ignore[override]
        if self._is_fitting_parameter_call(node.func):
            info = self._extract_call_info(node)
            if info is not None:
                self.results.append(info)
        self.generic_visit(node)

    # helpers -------------------------------------------------------------
    @staticmethod
    def _is_fitting_parameter_call(func: ast.expr) -> bool:
        # FittingParameter(...)
        if isinstance(func, ast.Name) and func.id == "FittingParameter":
            return True
        # chisurf.fitting.parameter.FittingParameter(...), or similar
        if isinstance(func, ast.Attribute) and func.attr == "FittingParameter":
            return True
        return False

    @staticmethod
    def _const_value(node: Optional[ast.AST]) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        return None

    def _extract_call_info(self, node: ast.Call) -> Optional[Dict[str, Any]]:
        def kw(name: str) -> Any:
            for k in node.keywords:
                if k.arg == name:
                    return self._const_value(k.value)
            return None

        name = kw("name")
        if not isinstance(name, str) or not name:
            return None

        description = kw("description")
        label_text = kw("label_text")
        fixed = kw("fixed")
        bounds_on = kw("bounds_on")
        lb = kw("lb")
        ub = kw("ub")

        return {
            "name": name,
            "module": self.module,
            "file": self.rel_path,
            "line": int(getattr(node, "lineno", -1)),
            "class": self._current("class"),
            "function": self._current("func"),
            "description_in_code": description if isinstance(description, str) else None,
            "label_text": label_text if isinstance(label_text, str) else None,
            "fixed": bool(fixed) if isinstance(fixed, bool) else None,
            "bounds_on": bool(bounds_on) if isinstance(bounds_on, bool) else None,
            "lb": float(lb) if isinstance(lb, (int, float)) else None,
            "ub": float(ub) if isinstance(ub, (int, float)) else None,
        }


def _load_existing_registry(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if isinstance(data, dict) and "parameters" in data and isinstance(data["parameters"], dict):
        return data["parameters"]
    if isinstance(data, dict):
        return data
    return {}


def _save_registry(path: Path, parameters: Dict[str, Any]) -> None:
    doc = {"version": 1, "parameters": parameters}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp.replace(path)


def _merge_results(existing: Dict[str, Any], discovered: List[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(existing) if isinstance(existing, dict) else {}

    for info in discovered:
        name = info.get("name")
        if not isinstance(name, str) or not name:
            continue

        entry = params.get(name)
        if not isinstance(entry, dict):
            entry = {"description": "", "keywords": [], "aliases": [], "label_texts": [], "sources": []}

        # Preserve existing description/keywords/aliases/label_texts if present
        desc = entry.get("description")
        if not isinstance(desc, str):
            desc = ""
        code_desc = info.get("description_in_code")
        if not desc and isinstance(code_desc, str) and code_desc:
            desc = code_desc

        keywords = entry.get("keywords") or []
        if not isinstance(keywords, list):
            keywords = []

        aliases = entry.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = []

        label_texts = entry.get("label_texts") or []
        if not isinstance(label_texts, list):
            label_texts = []
        lt = info.get("label_text")
        if isinstance(lt, str) and lt and lt not in label_texts:
            label_texts.append(lt)

        # Merge source info (avoid exact duplicates)
        sources = entry.get("sources") or []
        if not isinstance(sources, list):
            sources = []
        src_key = (info.get("file"), info.get("line"))
        seen = {(s.get("file"), s.get("line")) for s in sources if isinstance(s, dict)}
        if src_key not in seen:
            sources.append({
                "module": info.get("module"),
                "file": info.get("file"),
                "line": info.get("line"),
                "class": info.get("class"),
                "function": info.get("function"),
                "description_in_code": info.get("description_in_code"),
                "label_text": info.get("label_text"),
                "fixed": info.get("fixed"),
                "bounds_on": info.get("bounds_on"),
                "lb": info.get("lb"),
                "ub": info.get("ub"),
            })

        entry["description"] = desc
        entry["keywords"] = keywords
        entry["aliases"] = aliases
        entry["label_texts"] = label_texts
        entry["sources"] = sources

        params[name] = entry

    return params


def _discover_fitting_parameters(source_root: Path, include_tests: bool = False) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # Package root ("chisurf") is the default scan target; optionally add tests
    scan_roots: List[Path] = [source_root]
    if include_tests:
        tests_dir = source_root.parent / "test"
        if tests_dir.is_dir():
            scan_roots.append(tests_dir)

    for root in scan_roots:
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError:
                continue

            # Compute module and relative path for reporting
            try:
                rel = path.relative_to(source_root.parent)
            except ValueError:
                rel = path
            rel_str = str(rel).replace("\\", "/")
            parts = list(rel.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module = ".".join(parts)

            visitor = FittingParameterVisitor(module=module, rel_path=rel_str)
            visitor.visit(tree)
            results.extend(visitor.results)

    return results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the chisurf codebase for FittingParameter(...) calls and "
            "update a JSON registry of fitting-parameter metadata."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Path to the chisurf package root (defaults to this file's parent).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path (defaults to chisurf/settings/constants/"
            "parameter_registry.json)."
        ),
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Also scan the test/ directory for FittingParameter uses.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    here = Path(__file__).resolve()
    default_root = here.parent.parent  # chisurf package directory
    source_root = Path(args.root).resolve() if args.root is not None else default_root

    if args.output is not None:
        out_path = Path(args.output).resolve()
    else:
        out_path = source_root / "settings" / "constants" / "parameter_registry.json"

    discovered = _discover_fitting_parameters(source_root, include_tests=args.include_tests)
    existing = _load_existing_registry(out_path)
    merged = _merge_results(existing, discovered)
    _save_registry(out_path, merged)


if __name__ == "__main__":
    main()
