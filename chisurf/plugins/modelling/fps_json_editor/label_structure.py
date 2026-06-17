"""A slim coordinator widget for the fps.json editor."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from qtpy import QtWidgets

import chisurf.gui.decorators
import chisurf.gui.widgets
import chisurf.gui.widgets.general
from chisurf.plugins.core.code_editor import SimpleCodeEditor

from .distance_panel import DistancePanel
from .flexfit_panel import FlexFitPanel
from .model import FpsJsonModel
from .position_panel import PositionPanel

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:

    def persist_plugin_state(n):  # noqa: D103
        return lambda c: c



@persist_plugin_state("fps_json_editor")
class LabelStructure(QtWidgets.QWidget):
    """The main coordinator widget for editing and inspecting fps.json configurations.

    Composes separate panels for position selection, distance restraints, FlexFit,
    and raw JSON code view.
    """

    name = "LabelStructure"

    @chisurf.gui.decorators.init_with_ui(ui_filename="fps_json_edit.ui")
    def __init__(self, *args, client: Any | None = None, **kwargs) -> None:
        """Initialize the editor by nesting sub-panels inside the tab widget."""
        self._client = client or self._make_default_client()
        self._model = FpsJsonModel()

        # Instantiate sub-panels
        self.position_panel = PositionPanel(client=self._client)
        self.distance_panel = DistancePanel()
        self.flexfit_panel = FlexFitPanel()

        # JSON view panel
        self.json_tab_widget = QtWidgets.QWidget()
        json_layout = QtWidgets.QVBoxLayout(self.json_tab_widget)
        json_layout.setContentsMargins(4, 4, 4, 4)

        self.json_editor = SimpleCodeEditor(language='JSON')
        json_layout.addWidget(self.json_editor)

        self.json_update_btn = QtWidgets.QPushButton("Update from Editor Text")
        self.json_update_btn.clicked.connect(self.onReadTextEdit)
        json_layout.addWidget(self.json_update_btn)

        # Clear standard UI tabs and add custom panels
        self.tabWidget.clear()
        self.tabWidget.addTab(self.position_panel, "Positions")
        self.tabWidget.addTab(self.distance_panel, "Distances")
        self.tabWidget.addTab(self.flexfit_panel, "FlexFit")
        self.tabWidget.addTab(self.json_tab_widget, "JSON")

        # Wire up actions from the .ui file menu/toolbar
        self.actionLoad.triggered.connect(self.onLoadJSON)
        self.actionSave.triggered.connect(self.onSaveJSON)
        self.actionClear.triggered.connect(self.onClearAll)

        # Connect sub-panel signals
        self.position_panel.position_added.connect(self._on_position_added)
        self.position_panel.position_removed.connect(self._on_position_removed)

        self.distance_panel.distance_added.connect(self._on_distance_added)
        self.distance_panel.distance_removed.connect(self._on_distance_removed)
        self.distance_panel.distance_modified.connect(self._on_distance_modified)
        self.distance_panel.score_set_added.connect(self._on_score_set_added)
        self.distance_panel.score_set_removed.connect(self._on_score_set_removed)

        self.flexfit_panel.flexfit_changed.connect(self._on_flexfit_changed)

        self._refresh_ui()

    @staticmethod
    def _make_default_client():
        """Create a default FpsJsonEditorClient with local in-process services."""
        from .gui.communication import FpsJsonEditorClient
        return FpsJsonEditorClient()

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
        """Backwards compatible getter for positions dict."""
        return self._model.positions

    @property
    def distances(self) -> dict[str, dict[str, Any]]:
        """Backwards compatible getter for distances dict."""
        return self._model.distances

    @property
    def score_sets(self) -> dict[str, dict[str, Any]]:
        """Backwards compatible getter for score sets."""
        return self._model.score_sets

    @property
    def extra_sections(self) -> dict[str, Any]:
        """Backwards compatible getter for extra sections."""
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
        self._model.extra_sections["FlexFit"] = self.flexfit_panel._extra_sections.get("FlexFit", {})
        self._refresh_json_tab()

    def _refresh_ui(self) -> None:
        self.position_panel.update_positions(self._model.positions)

        label_names = list(self._model.positions.keys())
        self.distance_panel.update_labels(label_names)
        self.distance_panel.update_score_sets(list(self._model.score_sets.keys()))
        self.distance_panel.update_distances_table(self._model.distances, self._model.score_sets)

        self.flexfit_panel.update_flexfit(self._model.extra_sections)
        self._refresh_json_tab()

    def _refresh_json_tab(self) -> None:
        payload = self._model.fps_json_payload
        s = json.dumps(payload, sort_keys=True, indent=4, separators=(',', ': '))
        self.json_editor.setText(s)

    def onReadTextEdit(self) -> None:
        """Parse raw JSON from text edit and rebuild the model."""
        s = self.json_editor.text()
        try:
            p = json.loads(s)
            self.fps_json_payload = p
        except json.JSONDecodeError:
            chisurf.gui.widgets.general.MyMessageBox(
                info="JSON Parse Error.\n",
                details=traceback.format_exc()
            )

    def onLoadJSON(self, filename: str | None = None) -> None:
        """Load JSON configuration file."""
        if filename is None:
            filename = chisurf.gui.widgets.get_filename(
                'Open JSON Labeling-File',
                'JSON-Files (*.fps.json)'
            )
        if filename:
            try:
                self._model.load_file(filename)
                self._refresh_ui()
            except Exception:
                chisurf.gui.widgets.general.MyMessageBox(
                    info="Failed to load JSON file.\n",
                    details=traceback.format_exc()
                )

    def onSaveJSON(self) -> None:
        """Save JSON configuration to a file."""
        filename = chisurf.gui.widgets.save_file(
            'Save JSON Labeling-File',
            'JSON-Files (*.fps.json)'
        )
        if filename:
            try:
                self._model.save_file(filename)
            except Exception:
                chisurf.gui.widgets.general.MyMessageBox(
                    info="Failed to save JSON file.\n",
                    details=traceback.format_exc()
                )

    def onClearAll(self) -> None:
        """Prompt to clear the entire data model."""
        reply = QtWidgets.QMessageBox.question(
            self, 'Clear Configuration',
            "Are you sure you want to clear all parameters?",
            QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._model = FpsJsonModel()
            self._refresh_ui()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = LabelStructure()
    win.show()
    sys.exit(app.exec_())
