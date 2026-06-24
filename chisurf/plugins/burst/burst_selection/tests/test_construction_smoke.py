"""PRD-23 Task 3: mandatory construction smoke test for Burst Selection.

The other GUI tests build the tool via ``__new__`` (bypassing construction), so a
missing import or a side-effect-on-init regression would slip through. This builds
the real widget offscreen and asserts it constructs and reuses the shared
``ChisurfDockTool`` base.
"""

from __future__ import annotations

import os

import pytest

try:
    from qtpy import QtWidgets
except ImportError:
    QtWidgets = None  # type: ignore[assignment]

from chisurf.gui.widgets.tools import ChisurfDockTool


_needs_qt = pytest.mark.skipif(QtWidgets is None, reason="Qt bindings not available")
_needs_offscreen = pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "") != "offscreen",
    reason="Set QT_QPA_PLATFORM=offscreen for headless test",
)


@_needs_qt
@_needs_offscreen
def test_tool_constructs_without_crash() -> None:
    """The BurstSelectionTool constructs without crashing (read-only init)."""
    from chisurf.plugins.burst.burst_selection.gui.tool import BurstSelectionTool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tool = BurstSelectionTool()
    try:
        assert tool.windowTitle() == "Burst Selection"
        assert hasattr(tool, "_client")
        # reuses the shared dockable-tool base (PRD-23)
        assert isinstance(tool, ChisurfDockTool)
        assert tool.acceptDrops() is True
        # no MFDB connection was opened on construction
        assert tool._mfdb_db is None
    finally:
        tool.close()


@_needs_qt
@_needs_offscreen
def test_tool_drag_drop_dispatches_to_add_paths(monkeypatch) -> None:
    """The base window drop hook routes to the tool's ``_add_paths``."""
    from chisurf.plugins.burst.burst_selection.gui.tool import BurstSelectionTool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tool = BurstSelectionTool()
    try:
        captured = []
        monkeypatch.setattr(tool, "_add_paths", lambda paths: captured.append(paths))
        tool.on_paths_dropped(["x.ptu"])
        assert captured == [["x.ptu"]]
    finally:
        tool.close()
