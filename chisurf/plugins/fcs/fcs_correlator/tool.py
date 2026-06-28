"""FCS Correlator — two-pane navigation tool."""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

from qtpy import QtWidgets

from chisurf.gui import QtCore
from chisurf.gui.widgets.navigation import NavigationPanelTool
from chisurf.gui.widgets.wizard.tttr_correlator.tttr_correlator import (
    WizardTTTRCorrelator,
)
from chisurf.plugins.fcs.fcs_correlator.wizard import FileAndStepsPage


# ---------------------------------------------------------------------------
# Workflow context
# ---------------------------------------------------------------------------

@dataclass
class FcsWorkflowContext:
    detector_settings: dict[str, Any] = field(default_factory=dict)
    channel_defs: dict[str, Any] = field(default_factory=dict)
    file_paths: list[pathlib.Path] = field(default_factory=list)
    expanded_files: list[str] = field(default_factory=list)
    use_photon_filter: bool = False
    use_fcs_merger: bool = True


# ---------------------------------------------------------------------------
# Panel factory helpers
# ---------------------------------------------------------------------------

def _bind(tool: NavigationPanelTool, role: str, widget: QtWidgets.QWidget) -> None:
    binder = getattr(tool, "bind_workflow_panel", None)
    if callable(binder):
        binder(role, widget)


def _panel_widget(tool: NavigationPanelTool, index: int) -> QtWidgets.QWidget | None:
    if index < 0 or index >= len(tool.panels):
        return None
    wrapper = tool.panels[index].get("instance")
    if wrapper is None:
        return None
    layout = wrapper.layout()
    if layout is None or layout.count() == 0:
        return None
    item = layout.itemAt(0)
    return item.widget() if item is not None else None


# ---------------------------------------------------------------------------
# Panel factories
# ---------------------------------------------------------------------------

def _detector_setup(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.gui.widgets.wizard.tttr_channeldefinition import (
        DetectorWizardPage,
    )
    w = DetectorWizardPage(parent=parent)
    w.setParent(parent)
    _bind(parent, "detector", w)
    return w


def _files_selection(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    w = FileAndStepsPage(parent=parent)
    w.setParent(parent)
    _bind(parent, "files", w)
    return w


def _photon_filter(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.gui.widgets.wizard.tttr_photonfilter import (
        WizardTTTRPhotonFilter,
    )
    w = WizardTTTRPhotonFilter(
        windows={},
        detectors={},
        show_dT=True,
        show_burst=False,
        show_mcs=True,
        show_decay=False,
        show_filter=False,
        parent=parent,
    )
    w.setParent(parent)
    _bind(parent, "filter", w)
    return w


def _correlator_panel(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.fcs.fcs_correlator.correlator_panel import (
        CorrelatorSettingsModel,
    )
    model = CorrelatorSettingsModel()
    form = AutoForm(model, parent=parent)
    model._form = form
    parent._correlator_model = model
    parent._correlator_form = form
    _bind(parent, "correlator", form)
    return form


def _fcs_merger(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    from chisurf.gui.autoform import AutoForm
    from chisurf.plugins.fcs.fcs_correlator.merger_panel import (
        MergerSettingsModel,
    )
    model = MergerSettingsModel()
    form = AutoForm(model, parent=parent)
    model._form = form
    parent._merger_model = model
    parent._merger_form = form
    _bind(parent, "merger", form)
    return form


# ---------------------------------------------------------------------------
# Panel definitions
# ---------------------------------------------------------------------------

CORRELATOR_PANELS = [
    {
        "name": "1. Detector Setup",
        "icon": "\U0001f39b\ufe0f",
        "description": "Define detector configurations and PIE windows.",
        "factory": _detector_setup,
        "role": "detector",
    },
    {
        "name": "2. Files & Steps",
        "icon": "\U0001f4c2",
        "description": "Select TTTR files and choose processing steps.",
        "factory": _files_selection,
        "role": "files",
    },
    {
        "name": "3. Photon / Burst Filter",
        "icon": "\U0001f50d",
        "description": "Filter photons by count rate or burst selection.",
        "factory": _photon_filter,
        "role": "filter",
    },
    {
        "name": "4. Correlator",
        "icon": "\U0001f4ca",
        "description": "Set correlation parameters, compute and view FCS curves.",
        "factory": _correlator_panel,
        "role": "correlator",
    },
    {
        "name": "5. FCS Merger",
        "icon": "\U0001f517",
        "description": "Merge and save FCS correlation curves.",
        "factory": _fcs_merger,
        "role": "merger",
    },
]


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class FcsCorrelatorTool(NavigationPanelTool):
    """FCS Correlator — two-pane navigation tool replacing the QWizard."""

    def __init__(self, parent=None):
        self.workflow_context = FcsWorkflowContext()
        self._workflow_panels: dict[str, QtWidgets.QWidget] = {}
        super().__init__(
            title="FCS Correlator",
            panels=CORRELATOR_PANELS,
            parent=parent,
            minimum_size=(950, 620),
            initial_size=(1180, 760),
            navigation_width=270,
            navigation_min_width=250,
        )

    def bind_workflow_panel(self, role: str, widget: QtWidgets.QWidget) -> None:
        self._workflow_panels[role] = widget
        if role == "detector":
            self._bind_detector_panel(widget)
        elif role == "files":
            self._bind_files_panel(widget)
        self._apply_context_to_panel(role, widget)

    def _bind_detector_panel(self, widget: QtWidgets.QWidget) -> None:
        try:
            widget.setup_combo.currentIndexChanged.connect(
                self._on_detector_setup_changed
            )
        except Exception:
            pass

    def _bind_files_panel(self, widget: QtWidgets.QWidget) -> None:
        try:
            widget.cb_photon_filter.toggled.connect(self._on_files_steps_changed)
            widget.cb_fcs_merger.toggled.connect(self._on_files_steps_changed)
        except Exception:
            pass

    def _on_nav_changed(self, index: int) -> None:
        self._refresh_context()
        super()._on_nav_changed(index)
        if 0 <= index < len(self.panels):
            role = str(self.panels[index].get("role") or "")
            widget = self._panel_widget(index)
            if widget is not None:
                self._apply_context_to_panel(role, widget)

    def _panel_widget(self, index: int) -> QtWidgets.QWidget | None:
        return _panel_widget(self, index)

    def _refresh_context(self) -> None:
        detector = self._workflow_panels.get("detector")
        if detector is not None:
            try:
                self.workflow_context.detector_settings = detector.get_settings()
                self.workflow_context.channel_defs = detector.channels()
            except Exception:
                pass
        files = self._workflow_panels.get("files")
        if files is not None:
            try:
                self.workflow_context.file_paths = [
                    pathlib.Path(p) for p in files.checked_files
                ]
                self.workflow_context.use_photon_filter = (
                    files.cb_photon_filter.isChecked()
                )
                self.workflow_context.use_fcs_merger = (
                    files.cb_fcs_merger.isChecked()
                )
            except Exception:
                pass

    def _apply_context_to_panel(self, role: str, widget: QtWidgets.QWidget) -> None:
        if role == "filter":
            self._apply_context_to_filter(widget)
        elif role == "correlator":
            self._apply_context_to_correlator(widget)
        elif role == "merger":
            self._apply_context_to_merger(widget)

    def _apply_context_to_filter(
        self, widget: QtWidgets.QWidget
    ) -> None:
        settings = self.workflow_context.detector_settings
        dets = settings.get("detectors", {}) or {}
        wins = settings.get("windows", {}) or {}
        dets = {k: v for k, v in dets.items() if isinstance(k, str) and k.strip()}
        wins = {k: v for k, v in wins.items() if isinstance(k, str) and k.strip()}
        if not dets and not wins:
            return
        cb2 = getattr(widget, "comboBox_2", None)
        cb3 = getattr(widget, "comboBox_3", None)
        if cb2 is not None:
            prev = cb2.blockSignals(True)
            cb2.clear()
            cb2.blockSignals(prev)
        if cb3 is not None:
            prev = cb3.blockSignals(True)
            cb3.clear()
            cb3.blockSignals(prev)
        try:
            widget.fill_detectors(dets)
            widget.fill_pie_windows(wins)
        except Exception:
            pass
        if cb2 is not None and cb2.count() > 0:
            cb2.setCurrentIndex(0)
        if cb3 is not None and cb3.count() > 0:
            cb3.setCurrentIndex(0)
        try:
            widget.update_detectors()
            widget.update_pie_windows()
        except Exception:
            pass
        try:
            setup_name = self.workflow_context.detector_settings.get("setup_name")
            if setup_name:
                idx = widget.comboBox.findText(setup_name)
                if idx >= 0:
                    widget.comboBox.setCurrentIndex(idx)
        except Exception:
            pass

    def _apply_context_to_correlator(
        self, widget: QtWidgets.QWidget
    ) -> None:
        import tttrlib

        model = getattr(self, "_correlator_model", None)
        if model is None:
            return
        channel_defs = self.workflow_context.channel_defs
        if channel_defs:
            model._channel_defs = channel_defs
        settings = self.workflow_context.detector_settings
        dets = settings.get("detectors", {}) or {}
        dets = {k: v for k, v in dets.items() if isinstance(k, str) and k.strip()}
        model.load_fcs_presets(
            settings.get("setup_name", ""), dets
        )
        expanded = self._collect_expanded_files()
        if expanded:
            parent = pathlib.Path(expanded[0]).resolve().parent
            model._analysis_folder = parent
        direct_mode = not self.workflow_context.use_photon_filter
        if direct_mode and expanded:
            filetype = settings.get("filetype", "")
            try:
                tttr_obj = None
                for fn in expanded:
                    p = pathlib.Path(fn)
                    if not p.exists():
                        continue
                    try:
                        tt = tttrlib.TTTR(p.as_posix(), filetype)
                    except Exception:
                        try:
                            tt = tttrlib.TTTR(p.as_posix())
                        except Exception:
                            continue
                    if tttr_obj is None:
                        tttr_obj = tt
                    else:
                        tttr_obj.append(tt)
                model._tttr = tttr_obj
            except Exception:
                pass
        try:
            model._form.refresh_plots()
        except Exception:
            pass
        for w in widget.findChildren(QtWidgets.QWidget):
            if getattr(w, "AUTOFORM_REFRESH", False):
                try:
                    w.refresh()
                except Exception:
                    pass

    def _collect_expanded_files(self) -> list[str]:
        import tttrlib
        files = self.workflow_context.file_paths
        allowed = {
            f".{ext.lower()}" if not ext.startswith(".") else ext.lower()
            for ext in tttrlib.TTTR.get_supported_container_names()
        }
        expanded: list[str] = []
        for p_str in files:
            p = pathlib.Path(p_str).resolve()
            if p.is_dir():
                for child in p.iterdir():
                    if child.is_file() and child.suffix.lower() in allowed:
                        expanded.append(str(child.resolve()))
            else:
                expanded.append(str(p))
        return expanded

    def _apply_context_to_merger(self, widget: QtWidgets.QWidget) -> None:
        correlator_model = getattr(self, "_correlator_model", None)
        merger_model = getattr(self, "_merger_model", None)
        if merger_model is None or correlator_model is None:
            return
        correlations = getattr(correlator_model, "_correlations", None)
        analysis_folder = getattr(correlator_model, "_analysis_folder", None)
        output_subdir = getattr(correlator_model, "_output_subdir", None)
        if correlations:
            folder = analysis_folder / output_subdir if analysis_folder and output_subdir else None
            merger_model.set_correlations(correlations, folder)
        elif analysis_folder:
            folder = (
                analysis_folder / output_subdir
                if output_subdir
                else analysis_folder
            )
            merger_model.load_correlations(folder)

    def _on_detector_setup_changed(self) -> None:
        files = self._workflow_panels.get("files")
        if files is not None:
            try:
                files.file_list.clear()
                files._files_or_checks_changed()
            except Exception:
                pass
        filt = self._workflow_panels.get("filter")
        if filt is not None:
            try:
                filt.onClearFiles()
            except Exception:
                pass

    def _on_files_steps_changed(self) -> None:
        import pathlib
        files = self._workflow_panels.get("files")
        if files is not None:
            try:
                self.workflow_context.use_photon_filter = (
                    files.cb_photon_filter.isChecked()
                )
                self.workflow_context.use_fcs_merger = (
                    files.cb_fcs_merger.isChecked()
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Plugin entrypoint
# ---------------------------------------------------------------------------

if __name__ == "plugin":
    tool = FcsCorrelatorTool()
    tool.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    tool = FcsCorrelatorTool()
    tool.show()
    sys.exit(app.exec_())
