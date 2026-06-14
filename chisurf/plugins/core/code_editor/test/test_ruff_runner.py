from __future__ import annotations

from chisurf.plugins.core.code_editor.ruff_runner import RuffRunner


class Completed:
    """Small subprocess.CompletedProcess stand-in for ruff output tests."""

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.args = ["ruff", "check"]


def test_ruff_runner_normalizes_json_diagnostics(monkeypatch) -> None:
    """Ruff JSON output is normalized into editor-friendly diagnostics."""
    stdout = """[
        {
            "filename": "/tmp/sample.py",
            "code": "F401",
            "message": "`os` imported but unused",
            "location": {"row": 2, "column": 1},
            "end_location": {"row": 2, "column": 3}
        }
    ]"""
    monkeypatch.setattr(RuffRunner, "_run", lambda self, args, content, timeout: Completed(stdout))

    result = RuffRunner().check("/tmp/sample.py", "import os\n", timeout_ms=1000)

    assert result["ok"] is True
    assert result["diagnostics"][0]["code"] == "F401"
    assert result["diagnostics"][0]["line"] == 2
    assert result["diagnostics"][0]["column"] == 0


def test_ruff_runner_uses_stdin_filename(monkeypatch) -> None:
    """Unsaved editor content is checked with --stdin-filename."""
    seen = {}

    def fake_run(self, args, content, timeout):
        seen["args"] = args
        seen["content"] = content
        return Completed("[]")

    monkeypatch.setattr(RuffRunner, "_run", fake_run)

    RuffRunner().check("/tmp/sample.py", "x = 1\n")

    assert seen["content"] == "x = 1\n"
    assert "--stdin-filename" in seen["args"]
    assert seen["args"][-1] == "-"


def test_ruff_runner_reports_missing_ruff(monkeypatch) -> None:
    """A missing or failing ruff executable returns structured diagnostics."""

    def fake_run(self, args, content, timeout):
        raise FileNotFoundError("ruff")

    monkeypatch.setattr(RuffRunner, "_run", fake_run)

    result = RuffRunner(command="ruff").check("/tmp/sample.py", "x = 1\n")

    assert result["ok"] is False
    assert result["diagnostics"][0]["code"] == "ruff"
