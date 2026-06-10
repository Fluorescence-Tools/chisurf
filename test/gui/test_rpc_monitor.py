"""Tests for the RPCMonitorWidget.

These tests require a Qt display server.  They are skipped when
``DISPLAY`` is unset (headless / CI) or when the Qt platform plugin
cannot be loaded.
"""

from __future__ import annotations

import os

import pytest

from chisurf.server.dispatcher import ServiceDispatcher
from chisurf.server.session import SessionState

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="Requires a Qt display server",
)


@pytest.mark.usefixtures("qapp")
class TestRPCMonitorWidget:
    """RPCMonitorWidget displays RPC calls in a QTableWidget."""

    def test_create_and_destroy(self, qapp):
        """Widget can be created and cleaned up."""
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        assert w._table is not None
        assert w._table.columnCount() == 6
        assert w._callback in ServiceDispatcher._monitors
        if w._callback is not None:
            ServiceDispatcher.remove_monitor(w._callback)
        assert w._callback not in ServiceDispatcher._monitors

    def test_clear_removes_rows(self, qapp):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        try:
            w._on_call_recorded("test.a", {}, {"ok": True}, 0.01)
            w._on_call_recorded("test.b", {}, {"ok": True}, 0.02)
            assert w._table.rowCount() == 2
            w.clear()
            assert w._table.rowCount() == 0
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)

    def test_records_call_properties(self, qapp):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        try:
            w._on_call_recorded(
                "my.method", {"x": 1, "y": "hello"}, {"ok": True, "result": 42}, 0.015
            )
            assert w._table.rowCount() == 1
            assert w._table.item(0, 2).text() == "method"
            assert w._table.item(0, 3).text() == "method"
            assert "15" in w._table.item(0, 4).text()
            assert w._table.item(0, 5).text() == "OK"
            assert w._table.item(0, 5).foreground().color().green() > 100
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)

    def test_records_error_status(self, qapp):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        try:
            w._on_call_recorded(
                "fail.method",
                {},
                {"ok": False, "error_code": "INTERNAL_ERROR"},
                0.005,
            )
            assert w._table.item(0, 5).text() == "ERR(INTERNAL_ERROR)"
            r = w._table.item(0, 5).foreground().color().red()
            g = w._table.item(0, 3).foreground().color().green()
            assert r > g
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)

    def test_filter_method_name(self, qapp):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        try:
            w._on_call_recorded("alpha", {}, {"ok": True}, 0.0)
            w._on_call_recorded("beta", {}, {"ok": True}, 0.0)
            w._on_call_recorded("gamma", {}, {"ok": True}, 0.0)
            assert w._table.rowCount() == 3
            w._filter_edit.setText("beta")
            assert w._table.isRowHidden(0) is True
            assert w._table.isRowHidden(1) is False
            assert w._table.isRowHidden(2) is True
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)

    def test_pause_suspends_updates(self, qapp):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        try:
            w._pause_btn.setChecked(True)
            assert w._suspend_updates is True
            w._on_call_recorded("paused", {}, {"ok": True}, 0.0)
            assert w._table.rowCount() == 0
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)

    def test_max_rows_enforced(self, qapp):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        try:
            w.max_rows = 10
            for i in range(15):
                w._on_call_recorded(f"m{i}", {}, {"ok": True}, 0.0)
            assert w._table.rowCount() == 10
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)

    def test_integration_with_dispatcher_monitor(self, qapp, qtbot):
        from chisurf.gui.widgets.rpc_monitor import RPCMonitorWidget

        w = RPCMonitorWidget()
        d = ServiceDispatcher(SessionState())
        try:
            d.register("test.integration", lambda p: {"ok": True, "result": "done"})
            d.dispatch("test.integration", {"key": "val"})
            qtbot.wait(50)
            assert w._table.rowCount() >= 1
            assert w._table.item(0, 1).text() == "test"
            assert w._table.item(0, 2).text() == "integration"
        finally:
            if w._callback is not None:
                ServiceDispatcher.remove_monitor(w._callback)
