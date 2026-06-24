"""GUI construction smoke test for the Micro-time Shifter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

# Skip if Qt bindings are unavailable
try:
    from qtpy import QtWidgets
except ImportError:
    QtWidgets = None  # type: ignore[assignment]

from chisurf.plugins.tttr.tttr_microtime_shifter.gui.tool import MicrotimeShifterTool


@pytest.mark.skipif(QtWidgets is None, reason="Qt bindings not available")
@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "") != "offscreen",
    reason="Set QT_QPA_PLATFORM=offscreen for headless test",
)
def test_tool_constructs_without_crash() -> None:
    """The MicrotimeShifterTool constructs without crashing."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    tool = MicrotimeShifterTool()
    assert tool.windowTitle() == "Micro-time Shifter"
    assert hasattr(tool, "_client")
    assert hasattr(tool, "plot")
    # reuses the shared dockable-tool base (PRD-23)
    from chisurf.gui.widgets.tools import ChisurfDockTool

    assert isinstance(tool, ChisurfDockTool)
    assert tool.acceptDrops() is True
    tool.close()


@pytest.mark.skipif(QtWidgets is None, reason="Qt bindings not available")
@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "") != "offscreen",
    reason="Set QT_QPA_PLATFORM=offscreen for headless test",
)
def test_tool_has_required_children() -> None:
    """The tool has the expected child widgets."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    tool = MicrotimeShifterTool()
    assert tool.centralWidget() is not None
    assert tool.plot is not None
    assert len(tool.findChildren(QtWidgets.QToolBar)) >= 1
    tool.close()


@pytest.mark.skipif(QtWidgets is None, reason="Qt bindings not available")
@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "") != "offscreen",
    reason="Set QT_QPA_PLATFORM=offscreen for headless test",
)
def test_tool_file_list_operations(tmp_path: Path) -> None:
    """The tool can add, select and clear files in its list."""
    from unittest.mock import MagicMock
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    
    tool = MicrotimeShifterTool()
    tool._on_file_path = MagicMock()

    # Create dummy files
    f1 = tmp_path / "test1.ptu"
    f1.write_bytes(b"data1")
    f2 = tmp_path / "test2.ptu"
    f2.write_bytes(b"data2")

    # Add paths
    tool._add_paths([f1, f2])
    assert len(tool._file_paths) == 2
    assert tool.file_list.count() == 2
    assert Path(tool.file_list.item(0).text()) == f1.resolve()

    # Select the second item
    tool.file_list.setCurrentRow(1)
    tool._on_file_selected()
    tool._on_file_path.assert_called_with(str(f2.resolve()))

    # Clear the list
    tool._clear_file_list()
    assert len(tool._file_paths) == 0
    assert tool.file_list.count() == 0
    tool.close()


@pytest.mark.skipif(QtWidgets is None, reason="Qt bindings not available")
@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "") != "offscreen",
    reason="Set QT_QPA_PLATFORM=offscreen for headless test",
)
def test_tool_reset_button_and_logy() -> None:
    """The tool reset buttons set shifts to 0, and logy updates line scale."""
    import numpy as np
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    tool = MicrotimeShifterTool()

    # Manually configure state to simulate a loaded file with one channel
    tool._n_mt = 4096
    tool._channel_shifts = {0: 10}
    tool._routing_channels = [0]
    tool._trigger_level = 200
    tool.trigger_level_line.setValue(200)

    # Build controls for the manually configured state
    tool._build_shift_controls()

    # Assert reset button callback works for channel 0
    buttons = tool.findChildren(QtWidgets.QPushButton)
    reset_btn = None
    for btn in buttons:
        if btn.text() == "↻":
            reset_btn = btn
            break

    assert reset_btn is not None

    # Click reset
    reset_btn.click()
    assert tool._channel_shifts[0] == 0

    # Test logy toggle
    assert tool.logy_action.isChecked() is False
    assert tool.trigger_level_line.value() == 200

    # Toggle logy on
    tool.logy_action.setChecked(True)
    assert np.isclose(tool.trigger_level_line.value(), np.log10(200))

    # Toggle logy off
    tool.logy_action.setChecked(False)
    assert tool.trigger_level_line.value() == 200

    tool.close()


@pytest.mark.skipif(QtWidgets is None, reason="Qt bindings not available")
@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM", "") != "offscreen",
    reason="Set QT_QPA_PLATFORM=offscreen for headless test",
)
def test_save_dialog_always_processes_all_files(tmp_path: Path) -> None:
    """_open_save_dialog always uses all loaded files, not just selected ones."""
    from unittest.mock import MagicMock
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    tool = MicrotimeShifterTool()
    tool._db = MagicMock(return_value=None)  # local file mode
    tool._client = MagicMock()

    # Add paths
    f1 = tmp_path / "test1.ptu"
    f1.write_bytes(b"data1")
    f2 = tmp_path / "test2.ptu"
    f2.write_bytes(b"data2")
    tool._add_paths([f1, f2])

    # Select one of the files
    tool.file_list.setCurrentRow(0)

    # Mock QFileDialog.getExistingDirectory
    QtWidgets.QFileDialog.getExistingDirectory = MagicMock(return_value=str(tmp_path / "out"))
    QtWidgets.QMessageBox.information = MagicMock()

    # Call save
    tool._open_save_dialog()

    # Check client.apply was called with all paths, not just f1
    tool._client.apply.assert_called_once()
    called_args = tool._client.apply.call_args[1]
    assert len(called_args["file_paths"]) == 2
    assert f1.resolve() in called_args["file_paths"]
    assert f2.resolve() in called_args["file_paths"]

    tool.close()



