from __future__ import annotations

"""Tests for chisurf.plugins.core.code_editor.agent_runtime."""

import json
from unittest.mock import MagicMock

import pytest

from chisurf.plugins.core.code_editor.agent_runtime import (
    ALLOWED_TOOLS,
    BLOCKED_TOOLS,
    TOOLS_REQUIRING_PARAMS,
    AgentMode,
    AgentObservation,
    AgentRunConfig,
    AgentRunState,
    AgentRuntime,
    AgentToolCall,
    AgentToolRegistry,
    _compare_metrics,
    _extract_metrics,
    parse_llm_response,
)


class TestParseLlmResponse:

    def test_parse_message_response(self):
        action, error = parse_llm_response(
            '{"type": "message", "content": "Hello"}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert error is None
        assert action["type"] == "message"
        assert action["content"] == "Hello"

    def test_parse_tool_call_response(self):
        action, error = parse_llm_response(
            '{"type": "tool_call", "tool": "fit.run", "params": {"fit_uid": "abc"}}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert error is None
        assert action["type"] == "tool_call"
        assert action["tool"] == "fit.run"

    def test_parse_final_response(self):
        action, error = parse_llm_response(
            '{"type": "final", "content": "Done"}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert error is None
        assert action["type"] == "final"

    def test_invalid_json(self):
        action, error = parse_llm_response(
            "not json",
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "invalid JSON" in error

    def test_non_object_json(self):
        action, error = parse_llm_response(
            '"string"',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "must be a JSON object" in error

    def test_missing_type(self):
        action, error = parse_llm_response(
            '{"tool": "fit.run"}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "missing 'type'" in error

    def test_unknown_type(self):
        action, error = parse_llm_response(
            '{"type": "invalid"}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "unknown type" in error

    def test_tool_call_missing_tool(self):
        action, error = parse_llm_response(
            '{"type": "tool_call", "params": {}}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "missing 'tool'" in error

    def test_unknown_tool(self):
        action, error = parse_llm_response(
            '{"type": "tool_call", "tool": "unknown.tool", "params": {}}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "unknown tool" in error

    def test_blocked_tool(self):
        action, error = parse_llm_response(
            '{"type": "tool_call", "tool": "fit.clear", "params": {}}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "blocked tool" in error

    def test_missing_required_params(self):
        action, error = parse_llm_response(
            '{"type": "tool_call", "tool": "parameter.set_value", "params": {}}',
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert action is None
        assert "missing required parameter" in error

    def test_code_fence_stripping(self):
        action, error = parse_llm_response(
            "```json\n{\"type\": \"message\", \"content\": \"ok\"}\n```",
            ALLOWED_TOOLS, BLOCKED_TOOLS,
        )
        assert error is None
        assert action["type"] == "message"


class TestMetricComparison:

    def test_chi2r_improves(self):
        improved, reason = _compare_metrics(
            {"chi2r": 1.0, "chi2": 2.0},
            {"chi2r": 2.0, "chi2": 4.0},
        )
        assert improved

    def test_chi2r_worsens(self):
        improved, reason = _compare_metrics(
            {"chi2r": 2.0, "chi2": 4.0},
            {"chi2r": 1.0, "chi2": 2.0},
        )
        assert not improved

    def test_fallback_to_chi2(self):
        improved, reason = _compare_metrics(
            {"chi2": 3.0},
            {"chi2": 5.0},
        )
        assert improved

    def test_non_finite_chi2r(self):
        import math
        improved, reason = _compare_metrics(
            {"chi2r": float("nan")},
            {"chi2r": 1.0},
        )
        assert not improved
        assert "non-finite" in reason

    def test_first_measurement_improves(self):
        improved, reason = _compare_metrics(
            {"chi2r": 1.0},
            None,
        )
        assert improved
        assert "first" in reason

    def test_no_comparable_metrics(self):
        improved, reason = _compare_metrics(
            {"other": 1.0},
            {"other": 2.0},
        )
        assert not improved
        assert "no comparable" in reason

    def test_extract_metrics(self):
        result = {"ok": True, "metrics": {"chi2r": 1.5, "chi2": 2.0}}
        metrics = _extract_metrics(result)
        assert metrics["chi2r"] == 1.5

    def test_extract_metrics_bad_result(self):
        assert _extract_metrics({"ok": False}) is None


class TestCodeRunDeadlockPrevention:

    def test_code_run_is_local_not_via_rpc(self):
        """Verify agent.code.run is executed locally, not via ChisurfClient.call."""
        client = MagicMock()
        config = AgentRunConfig(code_run_enabled=True)
        registry = AgentToolRegistry(client, None, config)

        result = registry.execute(AgentToolCall(
            "agent.code.run",
            {"code": "print('hello')", "timeout_ms": 5000},
        ))
        assert result.get("ok")
        assert "hello" in result.get("stdout", "")
        # Should NOT have called client.call (which would deadlock)
        client.call.assert_not_called()

    def test_code_run_env_vars_present(self):
        """Verify runtime-local code.run receives CHISURF_RPC env vars."""
        client = MagicMock()
        config = AgentRunConfig(code_run_enabled=True)
        registry = AgentToolRegistry(client, None, config)

        code = "import os; print(os.environ.get('CHISURF_RPC_HOST', 'missing'))"
        result = registry.execute(AgentToolCall(
            "agent.code.run",
            {"code": code, "timeout_ms": 5000},
        ))
        assert "127.0.0.1" in result.get("stdout", "")


class TestToolRegistry:

    def test_execute_allowed_tool(self):
        client = MagicMock()
        client.call.return_value = {"ok": True}
        config = AgentRunConfig()
        registry = AgentToolRegistry(client, None, config)
        result = registry.execute(AgentToolCall("fit.run", {"fit_uid": "abc"}))
        assert result["ok"]
        client.call.assert_called_once_with("fit.run", {"fit_uid": "abc"})

    def test_code_run_disabled(self):
        client = MagicMock()
        config = AgentRunConfig(code_run_enabled=False)
        registry = AgentToolRegistry(client, None, config)
        result = registry.execute(AgentToolCall("agent.code.run", {"code": "print(1)"}))
        assert not result["ok"]
        assert "disabled" in result.get("error", "")


class TestAgentRuntime:

    def test_autonomous_fit_loop_completes(self):
        events = []

        def event_cb(ev, data):
            events.append((ev, data))

        def llm_call(messages):
            """Fake LLM that simulates a complete fit loop."""
            if "session.describe" in str(messages[-1]):
                return json.dumps({"type": "tool_call", "tool": "session.describe", "params": {}})
            if "fit.get" in str(messages[-1]):
                return json.dumps({"type": "tool_call", "tool": "session.describe", "params": {}})
            return json.dumps({"type": "final", "content": "All done."})

        registry = MagicMock()
        registry.execute.side_effect = [
            {"ok": True, "fits": [{"index": 0, "uid": "fit-1", "name": "TestFit"}]},  # session.describe
            {"ok": True},  # fit.get / session.describe again
        ]

        config = AgentRunConfig(mode=AgentMode.AUTONOMOUS_FIT, max_tool_iterations=10)
        runtime = AgentRuntime(
            config=config,
            tool_registry=registry,
            event_callback=event_cb,
            llm_call_fn=llm_call,
        )
        runtime.start("Fit this data")

        events_types = [ev for ev, _ in events]
        assert "message.started" in events_types
        assert "agent.completed" in events_types

    def test_cancellation_stops_loop(self):
        events = []
        call_count = 0
        import threading as _threading

        def event_cb(ev, data):
            events.append((ev, data))

        def llm_call(messages):
            nonlocal call_count
            call_count += 1
            import time
            time.sleep(0.02)
            return json.dumps({"type": "tool_call", "tool": "session.describe", "params": {}})

        registry = MagicMock()
        registry.execute.return_value = {"ok": True, "fits": []}

        config = AgentRunConfig(mode=AgentMode.AUTONOMOUS_FIT, max_tool_iterations=100)
        runtime = AgentRuntime(
            config=config,
            tool_registry=registry,
            event_callback=event_cb,
            llm_call_fn=llm_call,
        )

        def cancel_later():
            runtime.cancel()

        _threading.Timer(0.05, cancel_later).start()

        runtime.start("Do stuff")

        events_types = [ev for ev, _ in events]
        assert "agent.cancelled" in events_types

    def test_invalid_json_retry(self):
        events = []
        call_log = []

        def event_cb(ev, data):
            events.append((ev, data))

        def llm_call(messages):
            call_log.append(messages[-1]["role"])
            if len(call_log) == 1:
                return "bad json"
            return json.dumps({"type": "final", "content": "ok"})

        registry = MagicMock()
        config = AgentRunConfig(mode=AgentMode.CHISURF_TOOLS, invalid_json_retries=2)
        runtime = AgentRuntime(
            config=config,
            tool_registry=registry,
            event_callback=event_cb,
            llm_call_fn=llm_call,
        )
        runtime.start("test")
        # Should have retried on bad json
        assert "agent.completed" in [ev for ev, _ in events]
        assert len(call_log) >= 2

    def test_blocked_tool_rejected(self):
        events = []

        def event_cb(ev, data):
            events.append((ev, data))

        call_count = [0]

        def llm_call(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({"type": "tool_call", "tool": "fit.clear", "params": {}})
            return json.dumps({"type": "final", "content": "done"})

        registry = MagicMock()
        config = AgentRunConfig(mode=AgentMode.CHISURF_TOOLS, invalid_json_retries=1)
        runtime = AgentRuntime(
            config=config,
            tool_registry=registry,
            event_callback=event_cb,
            llm_call_fn=llm_call,
        )
        runtime.start("test")

        events_types = [ev for ev, _ in events]
        assert "tool.failed" in events_types

    def test_rollback_after_worsening_metrics(self):
        """Verify that 3 consecutive worsening iterations trigger rollback."""
        events = []
        call_log: list[AgentToolCall] = []

        def event_cb(ev, data):
            events.append((ev, data))

        def llm_call(messages):
            return json.dumps({"type": "final", "content": "all done"})

        registry = MagicMock()
        registry.execute.side_effect = [
            # Iteration 1: fit.run succeeds, diagnostics shows worsening
            {"ok": True},                                                         # fit.run
            {"ok": True, "metrics": {"chi2r": 5.0}, "parameters": []},           # auto diagnostics
            {"ok": True, "snapshot": {"parameters": [{"name": "p1", "value": 1}]}},  # auto snapshot (improved)
            # Iteration 2: fit.run succeeds, diagnostics shows worsening
            {"ok": True},                                                         # fit.run
            {"ok": True, "metrics": {"chi2r": 6.0}, "parameters": []},           # auto diagnostics
            # Iteration 3: fit.run succeeds, diagnostics shows worsening
            {"ok": True},                                                         # fit.run
            {"ok": True, "metrics": {"chi2r": 7.0}, "parameters": []},           # auto diagnostics
        ]
        config = AgentRunConfig(mode=AgentMode.AUTONOMOUS_FIT, max_tool_iterations=10)
        runtime = AgentRuntime(
            config=config,
            tool_registry=registry,
            event_callback=event_cb,
            llm_call_fn=llm_call,
        )
        runtime._state.selected_fit_uid = "fit-1"
        runtime._state.selected_fit_name = "Test"
        runtime._state.best_metrics = {"chi2r": 4.0}
        runtime._state.best_snapshot = {"parameters": [{"name": "p1", "value": 1}]}
        runtime._state.worsening_count = 1
        runtime._state.initial_metrics = {"chi2r": 4.0}
        runtime._state.fit_runs = 3

        runtime._handle_fit_completion({"ok": True})
        # worsening_count should now be 2
        assert runtime._state.worsening_count == 2

        runtime._handle_fit_completion({"ok": True})
        # worsening_count should now be 3, loop should stop on next _should_stop check
        assert runtime._state.worsening_count == 3

        runtime._finalize()
        events_types = [ev for ev, _ in events]
        assert "fit.rollback.completed" in events_types
