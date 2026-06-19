"""Dock-based editor workspace for fps.json files."""

from __future__ import annotations

import json
import traceback
from typing import Any

from qtpy import QtCore, QtWidgets

import chisurf as cs
import chisurf.gui.widgets as gui_widgets
import chisurf.gui.widgets.general as gui_general
from chisurf.gui.widgets.dock_area import DockArea
from chisurf.plugins.core.code_editor import SimpleCodeEditor

from chisurf.plugins.chimol.chimol.renderer.view import MolView
from ..core.model import FpsJsonModel
from .distance_panel import DistancePanel
from .flexfit_panel import FlexFitPanel
from .position_panel import PositionPanel


class FpsJsonEditor(QtWidgets.QWidget):
    """Coordinator widget for editing and inspecting fps.json configurations."""

    name = "FpsJsonEditor"

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        client: Any | None = None,
    ) -> None:
        """Create the dock-based editor workspace.

        Parameters
        ----------
        parent : QWidget, optional
            Parent Qt widget.
        client : object, optional
            RPC client used by panels for backend calls.
        """
        super().__init__(parent)
        self._client = client or self._make_default_client()
        self._model = FpsJsonModel()

        self.mol_view_3d = MolView()
        self.position_panel = PositionPanel(client=self._client, mol_view_3d=self.mol_view_3d)
        self.distance_panel = DistancePanel(
            position_panel=self.position_panel,
            mol_view_3d=self.mol_view_3d
        )
        self.flexfit_panel = FlexFitPanel()
        self.json_tab_widget = self._build_json_panel()

        self.dock_area = DockArea(self)
        self.dock_area.setNewTabButtonVisible(False)
        self.dock_area.setTabsClosable(False)
        self.dock_area.setContextMenuEnabled(True)
        self.dock_area.addTab(self.position_panel, "Positions")
        self.dock_area.addTab(self.distance_panel, "Distances")
        self.dock_area.addTab(self.flexfit_panel, "FlexFit")
        self.dock_area.addTab(self.json_tab_widget, "JSON")
        self.dock_area.addTab(self.mol_view_3d, "3D View")
        self.dock_area.layoutChanged.connect(self.save_dock_layout_state)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dock_area, 1)

        self._connect_panels()
        self._refresh_ui()
        self.restore_dock_layout_state()

    @staticmethod
    def _make_default_client():
        """Create a default FpsJsonEditorClient with local in-process services."""
        from ..api.client import FpsJsonEditorClient
        return FpsJsonEditorClient()

    def _build_json_panel(self) -> QtWidgets.QWidget:
        """Create the JSON text editor panel."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.json_editor = SimpleCodeEditor(language="JSON")
        layout.addWidget(self.json_editor, 1)

        return widget

    def _connect_panels(self) -> None:
        """Connect panel signals to the editor model."""
        self.position_panel.position_added.connect(self._on_position_added)
        self.position_panel.position_removed.connect(self._on_position_removed)

        self.distance_panel.distance_added.connect(self._on_distance_added)
        self.distance_panel.distance_removed.connect(self._on_distance_removed)
        self.distance_panel.distance_modified.connect(self._on_distance_modified)
        self.distance_panel.score_set_added.connect(self._on_score_set_added)
        self.distance_panel.score_set_removed.connect(self._on_score_set_removed)

        self.flexfit_panel.flexfit_changed.connect(self._on_flexfit_changed)

    def _settings(self) -> QtCore.QSettings:
        """Return QSettings for the editor dock layout."""
        settings_path = cs.core.settings.get_path("settings") / "fps_json_editor_dock_layout.ini"
        return QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    def _widget_key(self, widget: QtWidgets.QWidget) -> str:
        """Return a stable key for dock layout persistence."""
        if widget is self.position_panel:
            return "positions"
        if widget is self.distance_panel:
            return "distances"
        if widget is self.flexfit_panel:
            return "flexfit"
        if widget is self.json_tab_widget:
            return "json"
        if widget is self.mol_view_3d:
            return "3d_view"
        return self.dock_area.tabText(self.dock_area.indexOf(widget))

    def get_dock_layout_state(self) -> dict[str, object]:
        """Return the current dock layout state."""
        return self.dock_area.get_layout_state(key_func=self._widget_key)

    def save_dock_layout_state(self) -> None:
        """Persist the current dock layout."""
        try:
            if self.dock_area.count() <= 0:
                return
            settings = self._settings()
            settings.setValue(
                "dock_layout",
                json.dumps(self.get_dock_layout_state(), sort_keys=True),
            )
            settings.sync()
        except Exception as exc:
            cs.logging.warning(f"Failed to save FPS JSON Editor dock layout: {exc}")

    def restore_dock_layout_state(self) -> None:
        """Restore the saved dock layout."""
        try:
            value = self._settings().value("dock_layout")
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
            cs.logging.warning(f"Failed to restore FPS JSON Editor dock layout: {exc}")

    @property
    def fps_json_payload(self) -> dict:
        """Get the current model state as a payload dictionary."""
        return self._model.fps_json_payload

    @fps_json_payload.setter
    def fps_json_payload(self, payload: dict) -> None:
        """Set the model state using a payload dictionary."""
        self._model.fps_json_payload = payload
        self._refresh_ui()

    @property
    def positions(self) -> dict[str, dict[str, Any]]:
        """Return the model positions."""
        return self._model.positions

    @property
    def distances(self) -> dict[str, dict[str, Any]]:
        """Return the model distances."""
        return self._model.distances

    @property
    def score_sets(self) -> dict[str, dict[str, Any]]:
        """Return the model score sets."""
        return self._model.score_sets

    @property
    def extra_sections(self) -> dict[str, Any]:
        """Return extra top-level fps.json sections."""
        return self._model.extra_sections

    def _on_position_added(self, name: str, params: dict) -> None:
        self._model.add_position(name, params)
        self._refresh_ui()

    def _on_position_removed(self, name: str) -> None:
        self._model.remove_position(name)
        self._refresh_ui()

    def _on_distance_added(self, name: str, params: dict, score_set: str) -> None:
        self._model.add_distance(name, params, score_set)
        self._refresh_ui()

    def _on_distance_removed(self, name: str) -> None:
        self._model.remove_distance(name)
        self._refresh_ui()

    def _on_distance_modified(self, name: str, field_key: str, value: float) -> None:
        if name:
            self._model.distances[name][field_key] = value
            self._refresh_json_tab()
        else:
            self._refresh_ui()

    def _on_score_set_added(self, name: str) -> None:
        self._model.add_score_set(name)
        self._refresh_ui()

    def _on_score_set_removed(self, name: str) -> None:
        self._model.remove_score_set(name)
        self._refresh_ui()

    def _on_flexfit_changed(self) -> None:
        self._model.extra_sections["FlexFit"] = self.flexfit_panel._extra_sections.get(
            "FlexFit",
            {},
        )
        self._refresh_json_tab()

    def _refresh_ui(self) -> None:
        self.position_panel.update_positions(self._model.positions)

        label_names = list(self._model.positions.keys())
        self.distance_panel.update_labels(label_names)
        self.distance_panel.update_score_sets(list(self._model.score_sets.keys()))
        self.distance_panel.update_distances_table(
            self._model.distances,
            self._model.score_sets,
        )

        self.flexfit_panel.update_flexfit(self._model.extra_sections)
        self._refresh_json_tab()

    def _refresh_json_tab(self) -> None:
        payload = self._model.fps_json_payload
        text = json.dumps(payload, sort_keys=True, indent=4, separators=(",", ": "))
        self.json_editor.setText(text)

    def onReadTextEdit(self) -> None:
        """Parse raw JSON from the text editor and rebuild the model."""
        try:
            self.fps_json_payload = json.loads(self.json_editor.text())
        except json.JSONDecodeError:
            gui_general.MyMessageBox(
                info="JSON Parse Error.\n",
                details=traceback.format_exc(),
            )

    def onLoadJSON(self, filename: str | bool | None = None) -> None:
        """Load a JSON configuration file."""
        if not filename or isinstance(filename, bool):
            filename = gui_widgets.get_filename(
                "Open JSON Labeling-File",
                "JSON-Files (*.fps.json)",
            )
        if filename:
            try:
                self._model.load_file(filename)
                self._refresh_ui()
            except Exception:
                gui_general.MyMessageBox(
                    info="Failed to load JSON file.\n",
                    details=traceback.format_exc(),
                )

    def onSaveJSON(self, filename: str | bool | None = None) -> None:
        """Save JSON configuration to a file."""
        if not filename or isinstance(filename, bool):
            filename = gui_widgets.save_file(
                "Save JSON Labeling-File",
                "JSON-Files (*.fps.json)",
            )
        if filename:
            try:
                self._model.save_file(filename)
            except Exception:
                gui_general.MyMessageBox(
                    info="Failed to save JSON file.\n",
                    details=traceback.format_exc(),
                )

    def onClearAll(self) -> None:
        """Prompt to clear the entire data model."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Configuration",
            "Are you sure you want to clear all parameters?",
            QtWidgets.QMessageBox.Yes,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.position_panel.clear_all()
            self._model = FpsJsonModel()
            self._refresh_ui()


__all__ = ["FpsJsonEditor"]
