"""Tests for ServiceDispatcher monitoring callbacks."""

from __future__ import annotations

from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState


class TestServiceDispatcherMonitor:
    """Class-level monitoring callbacks on ServiceDispatcher."""

    def setup_method(self):
        self._events: list[tuple] = []
        self._cb = self._capture
        ServiceDispatcher.add_monitor(self._cb)

    def teardown_method(self):
        ServiceDispatcher.remove_monitor(self._cb)

    def _capture(self, method, params, result, elapsed):
        self._events.append((method, params, result.get("ok"), elapsed))

    def test_notifies_on_success(self):
        d = ServiceDispatcher(SessionState())
        d.register("test.ok", lambda p: {"ok": True, "result": 42})
        result = d.dispatch("test.ok", {"x": 1})
        assert result["ok"] is True
        assert len(self._events) == 1
        meth, params, ok, elapsed = self._events[0]
        assert meth == "test.ok"
        assert params == {"x": 1}
        assert ok is True
        assert elapsed >= 0

    def test_notifies_on_method_not_found(self):
        d = ServiceDispatcher(SessionState())
        result = d.dispatch("does.not.exist")
        assert result["ok"] is False
        assert len(self._events) == 1
        meth, _, ok, elapsed = self._events[0]
        assert meth == "does.not.exist"
        assert ok is False
        assert elapsed >= 0

    def test_notifies_on_handler_error(self):
        d = ServiceDispatcher(SessionState())

        def failing_handler(params):
            raise ValueError("boom")

        d.register("test.boom", failing_handler)
        result = d.dispatch("test.boom")
        assert result["ok"] is False
        assert result["error_code"] == "INTERNAL_ERROR"
        assert len(self._events) == 1
        _, _, ok, _ = self._events[0]
        assert ok is False

    def test_notifies_multiple_instances(self):
        d1 = ServiceDispatcher(SessionState())
        d2 = ServiceDispatcher(SessionState())
        d1.register("a", lambda p: {"ok": True})
        d2.register("b", lambda p: {"ok": True})
        d1.dispatch("a")
        d2.dispatch("b")
        assert len(self._events) == 2

    def test_remove_monitor_stops_notifications(self):
        ServiceDispatcher.remove_monitor(self._cb)
        d = ServiceDispatcher(SessionState())
        d.register("x", lambda p: {"ok": True})
        d.dispatch("x")
        assert len(self._events) == 0

    def test_monitor_exception_does_not_crash_dispatch(self):
        def bad_cb(m, p, r, e):
            raise RuntimeError("monitor crash")

        ServiceDispatcher.add_monitor(bad_cb)
        d = ServiceDispatcher(SessionState())
        d.register("safe", lambda p: {"ok": True})
        result = d.dispatch("safe")
        ServiceDispatcher.remove_monitor(bad_cb)
        assert result["ok"] is True
        assert len(self._events) == 1  # our good cb still fired

    def test_notifies_elapsed_time_is_positive(self):
        import time

        d = ServiceDispatcher(SessionState())
        d.register("slow", lambda p: time.sleep(0.01) or {"ok": True})
        d.dispatch("slow")
        _, _, _, elapsed = self._events[0]
        assert elapsed >= 0.01
