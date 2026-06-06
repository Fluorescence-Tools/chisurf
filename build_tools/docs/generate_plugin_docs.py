#!/usr/bin/env python
"""Generate plugin documentation for Sphinx from plugin __init__.py files.

Outputs to docs/plugins.md, which is included in the Sphinx toctree.

Usage:
    python build_tools/docs/generate_plugin_docs.py
"""

import ast
import pathlib
import sys
from typing import List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _get_plugin_name(package_path: pathlib.Path) -> str | None:
    init_py = package_path / "__init__.py"
    if not init_py.exists():
        return None
    source = init_py.read_text(encoding="utf-8")
    import re
    m = re.search(r'name\s*=\s*[\'"]([^\'"]*)[\'"]', source)
    return m.group(1) if m else None


def _get_docstring(package_path: pathlib.Path) -> str | None:
    init_py = package_path / "__init__.py"
    if not init_py.exists():
        return None
    source = init_py.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_py))
    return ast.get_docstring(tree)


def _get_all_plugins() -> List[Tuple[str, str, str]]:
    import pkgutil
    import chisurf.plugins

    plugin_root = pathlib.Path(chisurf.plugins.__file__).resolve().parent
    modules = [name for _, name, _ in pkgutil.iter_modules(chisurf.plugins.__path__) if not name.startswith("_")]

    result = []
    for name in modules:
        pkg_path = plugin_root / name
        plugin_name = _get_plugin_name(pkg_path) or name
        doc = _get_docstring(pkg_path) or "No description available."
        result.append((name, plugin_name, doc))

    result.sort(key=lambda x: x[1])
    return result


def _group_by_category(plugins):
    categories = {}
    for mod_name, full_name, doc in plugins:
        if ":" in full_name:
            cat, label = full_name.split(":", 1)
        else:
            cat, label = "Uncategorized", full_name
        categories.setdefault(cat, []).append((mod_name, label, doc))
    return dict(sorted(categories.items()))


def _display_label(label: str) -> str:
    label = label.strip()
    return label or "Unnamed"


def generate_plugin_md(output_path: str | pathlib.Path) -> str:
    plugins = _get_all_plugins()
    categories = _group_by_category(plugins)

    lines = []
    lines.append("# Plugin Reference\n")
    lines.append("Auto-generated documentation for all ChiSurf plugins.\n")

    lines.append("## Table of Contents\n")
    for cat in categories:
        lines.append(f"- {cat}")
        for _, label, _ in categories[cat]:
            lines.append(f"  - {_display_label(label)}")
    lines.append("")

    for cat in categories:
        lines.append(f"## {cat}\n")
        for index, (mod_name, label, doc) in enumerate(categories[cat]):
            lines.append(f"### {_display_label(label)}\n")
            lines.append(f"*Module: `{mod_name}`*\n")
            for para in doc.strip().split("\n\n"):
                stripped = para.strip()
                if stripped:
                    lines.append(stripped)
                    lines.append("")
            if index < len(categories[cat]) - 1:
                lines.append("---\n")

    text = "\n".join(lines)
    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Plugin docs written to {path}")
    return text


if __name__ == "__main__":
    output = REPO_ROOT / "docs" / "plugins.md"
    generate_plugin_md(output)
