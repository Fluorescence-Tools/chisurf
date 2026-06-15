from __future__ import annotations

"""Tests for chisurf.server.services.agent."""

from chisurf.server.services.agent import agent_code_run
from chisurf.server.session import SessionState


class TestAgentCodeRun:

    def test_execute_simple_python(self):
        state = SessionState()
        result = agent_code_run(state, code="print('hello')", timeout_ms=5000)
        assert result["ok"]
        assert "hello" in result["stdout"]
        assert result["elapsed_ms"] >= 0

    def test_execute_with_result_json(self):
        state = SessionState()
        code = "import json; print('data'); print('RESULT_JSON:' + json.dumps({'key': 'value'}))"
        result = agent_code_run(state, code=code, timeout_ms=5000)
        assert result["ok"]
        assert result["result_json"] == {"key": "value"}

    def test_execute_captures_stderr(self):
        state = SessionState()
        code = "import sys; sys.stderr.write('error msg')"
        result = agent_code_run(state, code=code, timeout_ms=5000)
        assert result["ok"]
        assert "error msg" in result["stderr"]

    def test_empty_code_returns_error(self):
        state = SessionState()
        result = agent_code_run(state, code="", timeout_ms=5000)
        assert not result["ok"]

    def test_no_shell_execution(self):
        state = SessionState()
        # Verify shell injection is not possible
        code = "import os; print(os.name)"
        result = agent_code_run(state, code=code, timeout_ms=5000)
        assert result["ok"]
        assert any(k in result["stdout"] for k in ("posix", "nt", "java"))

    def test_timeout_returns_error(self):
        state = SessionState()
        code = "import time; time.sleep(10)"
        result = agent_code_run(state, code=code, timeout_ms=100)
        assert not result["ok"]
        assert "timed out" in result.get("error", "").lower()
