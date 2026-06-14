from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path
from typing import Any


def validate_writes(
    writes: list[tuple[str, str]],
    editor: object | None = None,
    timeout_ms: int = 5000,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Validate generated code with py_compile and ruff.

    Parameters
    ----------
    writes : list[tuple[str, str]]
        Filename and content pairs to validate.
    editor : object, optional
        CodeEditor instance providing ``ruff_runner``.
    timeout_ms : int, optional
        Ruff timeout in milliseconds.

    Returns
    -------
    list[tuple[str, list[dict[str, Any]]]]
        Diagnostics grouped by filename.
    """
    issues_by_file: list[tuple[str, list[dict[str, Any]]]] = []
    for filename, content in writes:
        diagnostics: list[dict[str, Any]] = []
        diagnostics.extend(_compile_check_content(content, filename))
        diagnostics.extend(_ruff_check_content(content, filename, editor, timeout_ms))
        if diagnostics:
            issues_by_file.append((filename, diagnostics))
    return issues_by_file


def _compile_check_content(content: str, filename: str) -> list[dict[str, Any]]:
    """Run py_compile on generated content and return diagnostics."""
    suffix = Path(filename).suffix or ".py"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    try:
        try:
            py_compile.compile(str(temp_path), doraise=True)
        except py_compile.PyCompileError as exc:
            return [_diagnostic_from_py_compile_error(exc, filename)]
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
    return []


def _diagnostic_from_py_compile_error(exc: py_compile.PyCompileError, filename: str) -> dict[str, Any]:
    """Convert a PyCompileError into a normalized diagnostic."""
    exc_value = getattr(exc, "exc_value", None)
    line = getattr(exc_value, "lineno", None) or getattr(exc_value, "args", [None])[0]
    column = getattr(exc_value, "offset", None)
    if isinstance(column, int) and column > 0:
        column -= 1
    message = str(exc)
    if not message:
        message = str(exc_value) if exc_value is not None else "py_compile failed"
    return {
        "path": filename,
        "line": int(line or 0),
        "column": int(column or 0),
        "code": "E999",
        "message": message,
        "severity": "error",
    }


def _ruff_check_content(
    content: str,
    filename: str,
    editor: object | None,
    timeout_ms: int,
) -> list[dict[str, Any]]:
    """Run ruff on generated content and return normalized diagnostics."""
    runner = getattr(editor, "ruff_runner", None) if editor is not None else None
    if runner is None:
        return []
    settings = {}
    try:
        from chisurf.plugins.core.code_editor.text_editor import get_editor_settings

        settings = get_editor_settings()
    except ImportError:
        pass
    hint_path = filename if Path(filename).suffix else "untitled.py"
    result = runner.check(
        path=hint_path,
        content=content,
        extra_args=list(settings.get("ruff_extra_args", [])),
        timeout_ms=int(settings.get("ruff_timeout_ms", timeout_ms)),
    )
    if not isinstance(result, dict):
        return []
    diagnostics = result.get("diagnostics", [])
    if diagnostics and hasattr(editor, "_show_ruff_result"):
        editor._show_ruff_result(result)
    return diagnostics
