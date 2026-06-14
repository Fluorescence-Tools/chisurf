from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class RuffRunner:
    """Run Ruff checks and fixes for editor documents."""

    def __init__(self, command: str | None = None, timeout_ms: int | None = None) -> None:
        self.command = command or self._default_command()
        self.timeout_ms = timeout_ms

    @staticmethod
    def _default_command() -> str:
        """Return the default ruff executable name."""
        return shutil.which("ruff") or "ruff"

    def check(
        self,
        path: str | Path,
        content: str | None = None,
        extra_args: list[str] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Run ``ruff check`` and return normalized diagnostics."""
        args = [self.command, "check", "--output-format", "json"]
        args.extend(self._extra_args(extra_args))
        if content is not None:
            args.extend(["--stdin-filename", str(Path(path).resolve()), "-"])
            return self._run_json(args, content, timeout_ms)
        args.append(str(path))
        return self._run_json(args, None, timeout_ms)

    def fix(
        self,
        path: str | Path,
        content: str | None = None,
        extra_args: list[str] | None = None,
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """Run ``ruff check --fix`` and return diagnostics plus fixed content when possible."""
        args = [self.command, "check", "--fix", "--exit-zero"]
        args.extend(self._extra_args(extra_args))
        if content is not None:
            args.extend(["--stdin-filename", str(Path(path).resolve()), "-"])
            try:
                result = self._run(args, content, timeout_ms)
            except Exception as exc:
                return {
                    "ok": False,
                    "path": str(path),
                    "diagnostics": [
                        {
                            "path": "",
                            "line": 0,
                            "column": 0,
                            "end_line": 0,
                            "end_column": 0,
                            "code": "ruff",
                            "message": str(exc),
                            "severity": "error",
                        }
                    ],
                    "command": args,
                    "error": str(exc),
                }
            fixed_content = self._fixed_content_from_stdout(result.stdout)
            diagnostics = self._diagnostics_from_output(result.stdout, result.stderr)
            return {
                "ok": True,
                "path": str(path),
                "diagnostics": diagnostics,
                "changed": fixed_content != content,
                "fixed_content": fixed_content if fixed_content != content else None,
                "command": result.args,
                "returncode": result.returncode,
            }

        source = Path(path)
        original = source.read_text(encoding="utf-8")
        args.append(str(path))
        try:
            result = self._run(args, None, timeout_ms)
        except Exception as exc:
            return {
                "ok": False,
                "path": str(path),
                "diagnostics": [
                    {
                        "path": "",
                        "line": 0,
                        "column": 0,
                        "end_line": 0,
                        "end_column": 0,
                        "code": "ruff",
                        "message": str(exc),
                        "severity": "error",
                    }
                ],
                "command": args,
                "error": str(exc),
            }
        fixed = source.read_text(encoding="utf-8") if source.exists() else original
        diagnostics = self._diagnostics_from_output(result.stdout, result.stderr)
        return {
            "ok": True,
            "path": str(path),
            "diagnostics": diagnostics,
            "changed": fixed != original,
            "fixed_content": fixed if fixed != original else None,
            "command": result.args,
            "returncode": result.returncode,
        }

    def _extra_args(self, extra_args: list[str] | None) -> list[str]:
        return [str(arg) for arg in extra_args or []]

    def _run_json(
        self,
        args: list[str],
        content: str | None,
        timeout_ms: int | None,
    ) -> dict[str, Any]:
        try:
            result = self._run(args, content, timeout_ms)
        except Exception as exc:
            return {
                "ok": False,
                "path": self._stdin_filename(args) or (args[-1] if args else ""),
                "diagnostics": [
                    {
                        "path": "",
                        "line": 0,
                        "column": 0,
                        "end_line": 0,
                        "end_column": 0,
                        "code": "ruff",
                        "message": str(exc),
                        "severity": "error",
                    }
                ],
                "command": args,
                "error": str(exc),
            }
        try:
            diagnostics = json.loads(result.stdout or "[]")
        except json.JSONDecodeError:
            diagnostics = self._diagnostics_from_output(result.stdout, result.stderr)
        if not isinstance(diagnostics, list):
            diagnostics = []
        return {
            "ok": True,
            "path": args[-1] if args[-1] != "-" else self._stdin_filename(args),
            "diagnostics": self._normalize_diagnostics(diagnostics),
            "command": result.args,
            "returncode": result.returncode,
        }

    def _run(
        self,
        args: list[str],
        content: str | None,
        timeout_ms: int | None,
    ) -> subprocess.CompletedProcess[str]:
        timeout = None if timeout_ms is None else max(0.001, timeout_ms / 1000)
        return subprocess.run(
            args,
            input=content,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    @staticmethod
    def _stdin_filename(args: list[str]) -> str:
        try:
            index = args.index("--stdin-filename")
            return args[index + 1]
        except ValueError:
            return ""

    @staticmethod
    def _fixed_content_from_stdout(stdout: str) -> str:
        stripped = stdout.strip()
        if not stripped:
            return ""
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                json.loads(stripped)
                return ""
            except json.JSONDecodeError:
                return stdout
        return stdout

    @staticmethod
    def _diagnostics_from_output(stdout: str, stderr: str) -> list[dict[str, Any]]:
        try:
            data = json.loads(stdout or "[]")
            if isinstance(data, list):
                return RuffRunner._normalize_diagnostics(data)
        except json.JSONDecodeError:
            pass
        text = stderr or stdout
        if not text.strip():
            return []
        return [
            {
                "path": "",
                "line": 0,
                "column": 0,
                "end_line": 0,
                "end_column": 0,
                "code": "ruff",
                "message": text.strip(),
                "severity": "error",
            }
        ]

    @staticmethod
    def _normalize_diagnostics(data: list[Any]) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            range_info = item.get("location", {}) if isinstance(item.get("location"), dict) else {}
            end_range = (
                item.get("end_location", {}) if isinstance(item.get("end_location"), dict) else {}
            )
            diagnostics.append(
                {
                    "path": str(item.get("filename", "")),
                    "line": int(range_info.get("row", 0) or 0),
                    "column": int((range_info.get("column", 1) or 1) - 1),
                    "end_line": int(end_range.get("row", 0) or 0),
                    "end_column": int((end_range.get("column", 1) or 1) - 1),
                    "code": str(item.get("code", "")),
                    "message": str(item.get("message", "")),
                    "severity": str(item.get("severity", "error")),
                }
            )
        return diagnostics
