"""Dockable TTTR LUT Tools plugin window."""

from __future__ import annotations

import json

from qtpy import QtCore, QtWidgets

import chisurf as cs
from chisurf.gui.widgets.dock_area import DockArea

from .settings_panel import TTTRSettingsPanel
from .tac_lut_panel import TACLinearizationPanel


class TTRLutToolsWidget(QtWidgets.QMainWindow):
    """Combined TTTR microtime LUT computation and settings workspace."""

    def __init__(self) -> None:
        """Create the combined LUT tools window."""
        super().__init__()

        self.setWindowTitle("TTTR LUT Tools")
        self.resize(1100, 700)

        self.dock_area = DockArea(self)
        self.dock_area.setNewTabButtonVisible(False)
        self.dock_area.setTabsClosable(False)
        self.dock_area.setContextMenuEnabled(True)
        self.setCentralWidget(self.dock_area)

        self.tac_panel = TACLinearizationPanel()
        self.settings_panel = TTTRSettingsPanel()
        self.dock_area.addTab(self.tac_panel, "Compute Microtime LUT")
        self.dock_area.addTab(self.settings_panel, "Create LUT Settings")

        self.statusBar().showMessage("Create a LUT, then assign it to channels in Create LUT Settings.")
        self.dock_area.layoutChanged.connect(self.save_dock_layout_state)
        self.restore_dock_layout_state()

    def _lut_tools_settings(self) -> QtCore.QSettings:
        """Return QSettings for the LUT tools dock layout."""
        settings_path = cs.core.settings.get_path("settings") / "tttr_lut_tools_dock_layout.ini"
        return QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    def _widget_key(self, widget: QtWidgets.QWidget) -> str:
        """Return a stable key for dock layout persistence."""
        if widget is self.tac_panel:
            return "compute_microtime_lut"
        if widget is self.settings_panel:
            return "create_lut_settings"
        return self.dock_area.tabText(self.dock_area.indexOf(widget))

    def get_dock_layout_state(self) -> dict[str, object]:
        """Return the current dock layout state."""
        return self.dock_area.get_layout_state(key_func=self._widget_key)

    def save_dock_layout_state(self) -> None:
        """Persist the current dock layout."""
        try:
            if self.dock_area.count() <= 0:
                return
            state = self.get_dock_layout_state()
            settings = self._lut_tools_settings()
            settings.setValue("dock_layout", json.dumps(state, sort_keys=True))
            settings.sync()
        except Exception as exc:
            cs.logging.warning(f"Failed to save TTTR LUT tools dock layout: {exc}")

    def restore_dock_layout_state(self) -> None:
        """Restore the saved dock layout."""
        try:
            settings = self._lut_tools_settings()
            value = settings.value("dock_layout")
            if isinstance(value, str):
                state = json.loads(value)
            elif isinstance(value, dict):
                state = value
            else:
                return
            self.dock_area.set_layout_state(
                state,
                key_func=self._widget_key,
                emit_change=False,
            )
        except Exception as exc:
            cs.logging.warning(f"Failed to restore TTTR LUT tools dock layout: {exc}")


if __name__ == "plugin":
    window = TTRLutToolsWidget()
    window.show()
