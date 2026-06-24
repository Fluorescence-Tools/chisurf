"""PRD-23 Task 4: widget construction must be read-only (no MFDB writes/opens).

A class of bugs (the FCS dialog writing to MFDB on construction; opening an
``MFDatabase`` in ``__init__`` triggering schema reconcile/migration writes) comes
from doing I/O during widget construction. This static guard parses every GUI module
and fails if any class ``__init__`` *directly* calls an MFDB write/open entrypoint.

It is AST-only (no Qt needed, runs anywhere). Calls inside nested functions/lambdas
defined in ``__init__`` are deferred callbacks and are intentionally not flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: MFDB write / connection-open entrypoints that must not run during construction.
FORBIDDEN_CALLS = frozenset(
    {
        "register_result",
        "register_raw_measurement",
        "register_operation",
        "set_object_sample_id",
        "save_setup",
        "add_setup",
        "add_sample",
        "add_artifact",
        "add_operation",
        "add_parameter",
        "MFDatabase",
        "reconcile_schema",
        "bootstrap_operation_parameter_defs",
    }
)

#: Known, accepted exceptions — keep empty. Add only with a tracking note.
ALLOWLIST: frozenset[str] = frozenset()

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GUI_ROOTS = ("chisurf/plugins", "chisurf/gui")


def _is_gui_module(path: Path) -> bool:
    sp = path.as_posix()
    return (
        "/gui" in sp
        or sp.endswith("tool.py")
        or "wizard" in sp
        or "widget" in sp
    )


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


_DEFERRED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _direct_nodes(body: list[ast.stmt]):
    """Yield nodes reachable in ``__init__`` without entering nested defs/lambdas."""
    stack = list(body)
    while stack:
        node = stack.pop()
        # A nested def/lambda/class is a deferred scope: skip it and its body.
        if isinstance(node, _DEFERRED_SCOPES):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _violations_in_file(path: Path) -> list[tuple[str, int]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            for sub in _direct_nodes(node.body):
                if isinstance(sub, ast.Call):
                    name = _call_name(sub)
                    if name in FORBIDDEN_CALLS:
                        hits.append((name, sub.lineno))
    return hits


def _scan() -> dict[str, list[tuple[str, int]]]:
    found: dict[str, list[tuple[str, int]]] = {}
    for root in _GUI_ROOTS:
        for path in (_REPO_ROOT / root).rglob("*.py"):
            if not _is_gui_module(path):
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in ALLOWLIST:
                continue
            hits = _violations_in_file(path)
            if hits:
                found[rel] = hits
    return found


def test_no_mfdb_writes_in_widget_init():
    violations = _scan()
    if violations:
        lines = [
            f"  {f}: " + ", ".join(f"{name}@{ln}" for name, ln in hits)
            for f, hits in sorted(violations.items())
        ]
        pytest.fail(
            "Widget __init__ must be read-only (PRD-23). MFDB write/open calls "
            "found during construction:\n" + "\n".join(lines)
        )


def test_guard_detects_a_planted_violation(tmp_path):
    """The guard itself catches a construction-time MFDB write (meta-test)."""
    planted = tmp_path / "gui_bad.py"
    planted.write_text(
        "class W:\n"
        "    def __init__(self):\n"
        "        register_result(kind='x')\n",
        encoding="utf-8",
    )
    assert _violations_in_file(planted) == [("register_result", 3)]


def test_guard_ignores_deferred_callbacks(tmp_path):
    """Calls in nested callbacks defined in __init__ are deferred, not flagged."""
    planted = tmp_path / "gui_ok.py"
    planted.write_text(
        "class W:\n"
        "    def __init__(self):\n"
        "        def on_click():\n"
        "            register_result(kind='x')\n"
        "        self._cb = on_click\n",
        encoding="utf-8",
    )
    assert _violations_in_file(planted) == []
