"""Integrated burst workflow GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qtpy import QtWidgets

from chisurf.gui.widgets.navigation import NavigationPanelTool
from chisurf.gui.widgets.wizard.tttr_channeldefinition.setup_client import (
    DetectorSetupClient,
)


@dataclass
class BurstWorkflowContext:
    """Shared state handed from earlier burst workflow steps to later steps."""

    channel_settings: dict[str, Any] = field(default_factory=dict)
    raw_files: list[Path] = field(default_factory=list)
    burst_folder: Path | None = None
    bur_files: list[Path] = field(default_factory=list)
    mfdb_artifacts: dict[str, Any] = field(default_factory=dict)
    raw_mfdb_artifacts: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible workflow context payload."""
        return {
            "channel_settings": self.channel_settings,
            "raw_files": [str(path) for path in self.raw_files],
            "burst_folder": str(self.burst_folder) if self.burst_folder else None,
            "bur_files": [str(path) for path in self.bur_files],
            "mfdb_artifacts": self.mfdb_artifacts,
            "raw_mfdb_artifacts": self.raw_mfdb_artifacts,
        }


class BurstDataSelectionWidget(QtWidgets.QWidget):
    """Workflow-local raw TTTR file/folder selector."""

    TTTR_EXTENSIONS = {".spc", ".ht3", ".ptu", ".hdf", ".h5", ".hdf5", ".pt3", ".t3r"}

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Create the data-selection panel."""
        super().__init__(parent)
        self._paths: list[Path] = []
        self._mfdb_imports: dict[str, dict[str, Any]] = {}
        self._mfdb_selections: dict[str, dict[str, Any]] = {}
        self._mfdb_client: Any | None = None

        layout = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.add_files_button = QtWidgets.QPushButton("Import files...", self)
        self.add_folder_button = QtWidgets.QPushButton("Import folder...", self)
        self.mfdb_button = QtWidgets.QPushButton("Select from MFDB...", self)
        self.clear_button = QtWidgets.QPushButton("Clear", self)
        controls.addWidget(self.add_files_button)
        controls.addWidget(self.add_folder_button)
        controls.addWidget(self.mfdb_button)
        controls.addWidget(self.clear_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.file_list = QtWidgets.QListWidget(self)
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list, 1)

        self.status_label = QtWidgets.QLabel("No TTTR files selected.", self)
        layout.addWidget(self.status_label)

        self.add_files_button.clicked.connect(self._select_files)
        self.add_folder_button.clicked.connect(self._select_folder)
        self.mfdb_button.clicked.connect(self._select_mfdb_dataset)
        self.clear_button.clicked.connect(self.clear)

    def paths(self) -> list[Path]:
        """Return selected raw TTTR paths."""
        return list(self._paths)

    def mfdb_payload(self) -> dict[str, Any]:
        """Return MFDB import and selection metadata."""
        return {
            "imports": self._mfdb_imports,
            "selections": self._mfdb_selections,
        }

    def add_paths(self, paths: list[Path]) -> None:
        """Add files or recursively add supported files from folders."""
        new_paths: list[Path] = []
        for path in paths:
            path = Path(path).expanduser()
            if path.is_dir():
                new_paths.extend(
                    sorted(
                        child.resolve()
                        for child in path.rglob("*")
                        if child.is_file() and child.suffix.lower() in self.TTTR_EXTENSIONS
                    )
                )
            elif path.is_file() and path.suffix.lower() in self.TTTR_EXTENSIONS:
                new_paths.append(path.resolve())

        existing = {path.resolve() for path in self._paths}
        for path in new_paths:
            if path.resolve() not in existing:
                self._paths.append(path.resolve())
                existing.add(path.resolve())
                self.file_list.addItem(str(path.resolve()))
                self._import_path_to_mfdb(path.resolve())
        self._update_status()

        callback = getattr(self.parent(), "_on_data_selection_changed", None)
        if callable(callback):
            callback()

    def clear(self) -> None:
        """Clear selected data files."""
        self._paths.clear()
        self._mfdb_imports.clear()
        self._mfdb_selections.clear()
        self.file_list.clear()
        self._update_status()
        callback = getattr(self.parent(), "_on_data_selection_changed", None)
        if callable(callback):
            callback()

    def _select_files(self) -> None:
        """Open a file dialog for TTTR files."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select TTTR files",
            "",
            "TTTR files (*.spc *.ht3 *.ptu *.hdf *.h5 *.hdf5 *.pt3 *.t3r);;All files (*)",
        )
        self.add_paths([Path(path) for path in paths])

    def _select_folder(self) -> None:
        """Open a folder dialog for TTTR files."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select TTTR folder")
        if folder:
            self.add_paths([Path(folder)])

    def _select_mfdb_dataset(self) -> None:
        """Select raw data from MFDB and resolve it to a local path."""
        try:
            from chisurf.gui.widgets.mfdb.dataset_browser import MfdbDatasetPickerDialog

            selection = MfdbDatasetPickerDialog.pick_dataset(
                parent=self,
                kinds=["raw_data", "raw_measurement", "external_reference"],
                scope="all",
                client=self._client(),
            )
            if selection is None:
                return
            local_path = selection.local_path or self._open_mfdb_dataset(selection.artifact_id)
            if not local_path:
                QtWidgets.QMessageBox.warning(
                    self,
                    "MFDB Dataset",
                    f"Could not resolve local path for artifact {selection.artifact_id}.",
                )
                return
            self._mfdb_selections[str(Path(local_path).resolve())] = {
                "artifact_id": selection.artifact_id,
                "artifact_kind": selection.artifact_kind,
                "data_format": selection.data_format,
                "label": selection.label,
                "metadata": selection.metadata,
            }
            self.add_paths([Path(local_path)])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "MFDB Dataset", f"MFDB selection failed:\n{exc}")

    def _open_mfdb_dataset(self, artifact_id: str) -> str | None:
        """Resolve an MFDB dataset artifact to a local path."""
        result = self._client().call("mfdb.datasets.open", {"artifact_id": artifact_id}) or {}
        return result.get("local_path") or result.get("path")

    def _import_path_to_mfdb(self, path: Path) -> None:
        """Import a local file into MFDB object store and raw-data registry."""
        try:
            client = self._client()
            object_result = client.call(
                "mfdb.objects.put",
                {
                    "path": str(path),
                    "filename": path.name,
                    "metadata": {"source": "burst_analysis.data_selection"},
                },
            ) or {}
            payload: dict[str, Any] = {"object_result": object_result}
            try:
                raw_result = client.call(
                    "raw_data.register",
                    {
                        "raw_data": {
                            "file_path": str(path),
                            "data_type": "TTTR",
                            "storage_mode": "file",
                            "header_metadata": {
                                "mfdb_object": object_result.get("object", {}),
                                "source": "burst_analysis.data_selection",
                            },
                        }
                    },
                ) or {}
                payload["raw_data_result"] = raw_result
            except Exception as raw_exc:
                payload["raw_data_error"] = str(raw_exc)
            self._mfdb_imports[str(path)] = payload
        except Exception as exc:
            self._mfdb_imports[str(path)] = {"error": str(exc)}

    def _client(self) -> Any:
        """Return the MFDB RPC client used for import and selection."""
        if self._mfdb_client is None:
            from chisurf.plugins.core.mfdb_admin.gui.client import MFDBClient

            self._mfdb_client = MFDBClient(inprocess=True)
        return self._mfdb_client

    def _update_status(self) -> None:
        """Update selection status label."""
        imported = len([entry for entry in self._mfdb_imports.values() if "error" not in entry])
        self.status_label.setText(
            f"{len(self._paths)} TTTR file(s) selected; {imported} imported to MFDB."
        )


def _data_selection(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the raw data selection panel."""
    widget = BurstDataSelectionWidget(parent=parent)
    _bind(parent, "data", widget)
    return widget


def _channel_selection(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the shared channel selection panel."""
    from chisurf.plugins.core.setup_channel_definition.gui.tool import (
        SetupChannelDefinitionWidget,
    )

    widget = SetupChannelDefinitionWidget(parent=parent)
    _bind(parent, "channels", widget)
    return widget


def _burst_selection(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the burst selection panel."""
    from chisurf.plugins.burst.burst_selection import BurstSelectionTool

    widget = BurstSelectionTool(parent=parent, show_channel_selection=False)
    _bind(parent, "selection", widget)
    return widget


def _burst_bva(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the BVA panel."""
    from chisurf.plugins.burst.burst_bva.gui.tool import BVATool

    widget = BVATool(parent=parent, embedded=True)
    _hide_dock_tab_by_name(widget, "Channel Definitions")
    _bind(parent, "bva", widget)
    return widget


def _burst_mle(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the burst MLE panel."""
    from chisurf.plugins.burst.burst_mle_analysis.wizard import MLELifetimeAnalysisWizard

    widget = MLELifetimeAnalysisWizard(parent=parent)
    _remove_tab_by_name(widget, "Detector Definition")
    _bind(parent, "mle", widget)
    return widget


def _burst_h2mm(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the H2MM panel."""
    from chisurf.plugins.burst.burst_h2mm.gui.tool import H2mmTool

    widget = H2mmTool(parent=parent, embedded=True)
    _hide_dock_tab_by_name(widget, "Channel Definitions")
    _bind(parent, "h2mm", widget)
    return widget


def _burst_browser(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create the burst browser panel."""
    from chisurf.plugins.burst.burst_browser import BurstBrowserWidget

    widget = BurstBrowserWidget(parent=parent)
    _bind(parent, "browser", widget)
    return widget


def _burst_background(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Create burst background estimation panel."""
    from chisurf.plugins.burst.burst_background import BurstBackgroundEstimator

    try:
        widget = BurstBackgroundEstimator(show_channel_definition=False)
    except TypeError as exc:
        if "show_channel_definition" not in str(exc):
            raise
        widget = BurstBackgroundEstimator()
        _remove_tab_by_name(widget, "Channel Definition")
    widget.setParent(parent)
    _bind(parent, "background", widget)
    return widget


def _bind(parent: QtWidgets.QWidget, role: str, widget: QtWidgets.QWidget) -> None:
    """Bind a loaded panel to the workflow coordinator when available."""
    binder = getattr(parent, "bind_workflow_panel", None)
    if callable(binder):
        binder(role, widget)


def _remove_tab_by_name(widget: QtWidgets.QWidget, tab_name: str) -> None:
    """Remove a duplicate tab from an embedded legacy widget."""
    for tab_widget in widget.findChildren(QtWidgets.QTabWidget):
        for index in range(tab_widget.count()):
            if tab_widget.tabText(index) == tab_name:
                tab_widget.removeTab(index)
                return


def _hide_dock_tab_by_name(widget: QtWidgets.QWidget, tab_name: str) -> None:
    """Hide a DockArea tab by its registered name."""
    dock_area = getattr(widget, "dock_area", None)
    if dock_area is None:
        return
    for index in range(len(getattr(dock_area, "_all_widgets", []))):
        try:
            if dock_area.tabText(index) == tab_name:
                dock_area.hideTab(index)
                return
        except Exception:
            return


BURST_PANELS = [
    {
        "name": "1. Data Selection",
        "icon": "📂",
        "description": "Select raw TTTR files used by all later steps.",
        "factory": _data_selection,
        "role": "data",
    },
    {
        "name": "2. Channels",
        "icon": "🔢",
        "description": "Define detector channels and PIE time windows once.",
        "factory": _channel_selection,
        "role": "channels",
    },
    {
        "name": "3. Burst Selection",
        "icon": "🔎",
        "description": "Find and filter bursts from TTTR data.",
        "factory": _burst_selection,
        "role": "selection",
    },
    {
        "name": "4. BVA",
        "icon": "📊",
        "description": "Run burst variance analysis using selected bursts.",
        "factory": _burst_bva,
        "role": "bva",
    },
    {
        "name": "5. MLE-Lifetime",
        "icon": "🎯",
        "description": "Fit burst lifetimes using selected bursts.",
        "factory": _burst_mle,
        "role": "mle",
    },
    {
        "name": "6. H2MM",
        "icon": "🔀",
        "description": "Resolve sub-burst FRET dynamics with photon-by-photon HMM.",
        "factory": _burst_h2mm,
        "role": "h2mm",
    },
    {
        "name": "7. Browser",
        "icon": "📋",
        "description": "Inspect the current burst workflow result.",
        "factory": _burst_browser,
        "role": "browser",
    },
    {
        "name": "────────",
        "icon": "",
        "separator": True,
        "role": "separator",
    },
    {
        "name": "Background",
        "icon": "🌙",
        "description": "Estimate background using the selected data and channel setup.",
        "factory": _burst_background,
        "role": "background",
    },
]


class BurstAnalysisTool(NavigationPanelTool):
    """Integrated five-step burst workflow tool."""

    def __init__(self, parent=None):
        """Create the integrated burst workflow tool."""
        self.workflow_context = BurstWorkflowContext()
        self._workflow_panels: dict[str, QtWidgets.QWidget] = {}
        # Shared detector definition flows through the central
        # ``detector_setups.*`` RPC store (same as the Imaging Tools window).
        self._setup_client = DetectorSetupClient()
        super().__init__(
            title="Burst Analysis",
            panels=BURST_PANELS,
            parent=parent,
            minimum_size=(950, 620),
            initial_size=(1180, 760),
            navigation_width=270,
            navigation_min_width=250,
        )

    def bind_workflow_panel(self, role: str, widget: QtWidgets.QWidget) -> None:
        """Register a loaded panel and apply current workflow context."""
        self._workflow_panels[role] = widget
        if role == "data":
            self._sync_data_context()
        elif role == "channels":
            self._bind_channel_panel(widget)
        elif role == "selection":
            self._bind_selection_panel(widget)
        self._apply_context_to_panel(role, widget)

    def _on_nav_changed(self, index: int) -> None:
        """Refresh and apply workflow context when the user changes steps."""
        self._refresh_workflow_context()
        super()._on_nav_changed(index)
        if 0 <= index < len(self.panels):
            role = str(self.panels[index].get("role") or "")
            widget = self._panel_widget(index)
            if widget is not None:
                self._apply_context_to_panel(role, widget)

    def _panel_widget(self, index: int) -> QtWidgets.QWidget | None:
        """Return the inner widget for a loaded panel wrapper."""
        if index < 0 or index >= len(self.panels):
            return None
        wrapper = self.panels[index].get("instance")
        if wrapper is None:
            return None
        layout = wrapper.layout()
        if layout is None or layout.count() == 0:
            return None
        item = layout.itemAt(0)
        return item.widget() if item is not None else None

    def _bind_channel_panel(self, widget: QtWidgets.QWidget) -> None:
        """Connect step-1 channel changes to downstream panels."""
        page = getattr(widget, "page", None)
        signal = getattr(page, "detectorsChanged", None)
        if signal is not None:
            try:
                signal.connect(self._on_channel_definitions_changed)
            except TypeError:
                pass
        self._sync_channel_context()

    def _bind_selection_panel(self, widget: QtWidgets.QWidget) -> None:
        """Wrap Burst Selection execution so downstream steps see outputs."""
        if getattr(widget, "_burst_analysis_wrapped", False):
            return
        original = getattr(widget, "analyze_files", None)
        if not callable(original):
            return

        def wrapped_analyze_files(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self._sync_selection_context(widget)
            self._apply_context_to_downstream()
            return result

        widget.analyze_files = wrapped_analyze_files
        widget._burst_analysis_wrapped = True

    def _on_channel_definitions_changed(self) -> None:
        """Propagate changed channel definitions to loaded downstream panels."""
        self._sync_channel_context()
        self._apply_context_to_downstream()

    def _refresh_workflow_context(self) -> None:
        """Refresh shared context from loaded upstream widgets."""
        self._sync_data_context()
        self._sync_channel_context()
        selection = self._workflow_panels.get("selection")
        if selection is not None:
            self._sync_selection_context(selection)

    def _sync_data_context(self) -> None:
        """Capture raw data selection from step 1."""
        panel = self._workflow_panels.get("data")
        paths = getattr(panel, "paths", None)
        if callable(paths):
            selected = paths()
            if selected:
                self.workflow_context.raw_files = selected
        mfdb_payload = getattr(panel, "mfdb_payload", None)
        if callable(mfdb_payload):
            self.workflow_context.raw_mfdb_artifacts = mfdb_payload()

    def _on_data_selection_changed(self) -> None:
        """Propagate changed raw data selection to loaded panels."""
        self._sync_data_context()
        self._apply_context_to_downstream()

    def _sync_channel_context(self) -> None:
        """Capture channel definitions from step 1 via the shared RPC store.

        The setup panel's definition is published to the central
        ``detector_setups.*`` RPC store, then read back so the workflow context
        always reflects the canonical (RPC-held) channel definition.
        """
        panel = self._workflow_panels.get("channels")
        page = getattr(panel, "page", None)
        get_settings = getattr(page, "get_settings", None)
        if callable(get_settings):
            try:
                settings = get_settings()
            except Exception:
                settings = None
            if settings:
                self._setup_client.set_current(settings)
        # Pull the canonical definition back from the RPC store.
        current = self._setup_client.get_current()
        if current:
            self.workflow_context.channel_settings = current

    def _sync_selection_context(self, widget: QtWidgets.QWidget) -> None:
        """Capture burst-selection outputs from step 2."""
        raw_files = [Path(path) for path in getattr(widget, "_file_paths", [])]
        if raw_files:
            self.workflow_context.raw_files = raw_files

        result = getattr(widget, "_last_service_result", None) or getattr(widget, "_last_result", None) or {}
        if isinstance(result, dict):
            artifacts = result.get("mfdb_artifacts") or {}
            if artifacts:
                self.workflow_context.mfdb_artifacts = dict(artifacts)

        folder = self._folder_from_selection_result(result)
        if folder is None:
            folder = self._materialize_burst_handoff(widget)
        if folder is not None:
            self.workflow_context.burst_folder = folder
            self.workflow_context.bur_files = sorted(folder.glob("**/*.bur"))

    def _folder_from_selection_result(self, result: object) -> Path | None:
        """Return an output folder from a burst-selection result payload."""
        if not isinstance(result, dict):
            return None
        candidates: list[str] = []
        for key in ("output_folder", "analysis_folder"):
            value = result.get(key)
            if isinstance(value, str):
                candidates.append(value)
        for nested_key in ("metadata", "output_paths"):
            nested = result.get(nested_key) or {}
            if isinstance(nested, dict):
                value = nested.get("output_folder")
                if isinstance(value, str):
                    candidates.append(value)
        for candidate in candidates:
            path = Path(candidate)
            if path.exists() and path.is_dir():
                return path
        return None

    def _materialize_burst_handoff(self, widget: QtWidgets.QWidget) -> Path | None:
        """Write cached burst frames to legacy ``bi4_bur`` handoff layout."""
        frames_by_file = getattr(widget, "_last_frames_by_file", None)
        if not frames_by_file:
            return None
        raw_files = self.workflow_context.raw_files or [Path(path) for path in frames_by_file.keys()]
        if not raw_files:
            return None

        output_folder = raw_files[0].parent / "burst_analysis_handoff"
        bur_folder = output_folder / "bi4_bur"
        bur_folder.mkdir(parents=True, exist_ok=True)

        from chisurf.plugins.burst.burst_selection.api.io import write_bur

        for raw_path in raw_files:
            frame = frames_by_file.get(raw_path.resolve())
            if frame is None:
                frame = frames_by_file.get(raw_path)
            if frame is None:
                continue
            write_bur(frame, bur_folder / f"{raw_path.stem}.bur")

        payload = self.workflow_context.to_payload()
        payload["raw_files"] = [str(path) for path in raw_files]
        (output_folder / "burst_analysis_handoff.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
        return output_folder

    def _apply_context_to_downstream(self) -> None:
        """Apply current workflow context to loaded downstream panels."""
        for role in ("selection", "bva", "mle", "browser", "background"):
            widget = self._workflow_panels.get(role)
            if widget is not None:
                self._apply_context_to_panel(role, widget)

    def _apply_context_to_panel(self, role: str, widget: QtWidgets.QWidget) -> None:
        """Apply current workflow context to one panel."""
        if role == "selection":
            self._apply_channels_to_burst_selection(widget)
        elif role == "bva":
            self._apply_context_to_bva(widget)
        elif role == "mle":
            self._apply_context_to_mle(widget)
        elif role == "h2mm":
            self._apply_context_to_h2mm(widget)
        elif role == "browser":
            self._apply_context_to_browser(widget)
        elif role == "background":
            self._apply_context_to_background(widget)

    def _apply_channels_to_burst_selection(self, widget: QtWidgets.QWidget) -> None:
        """Use step-1 definitions in Burst Selection."""
        raw_files = self.workflow_context.raw_files
        if raw_files and not getattr(widget, "_file_paths", []):
            try:
                widget._add_paths(raw_files)
            except Exception:
                pass
        settings = self.workflow_context.channel_settings
        if not settings:
            return
        try:
            widget._apply_custom_detector_settings(
                windows=settings.get("windows", {}),
                detectors=settings.get("detectors", {}),
                tttr_reading=settings.get("tttr_reading", {}),
            )
        except Exception:
            wizard = getattr(widget, "wizard", None)
            if wizard is not None:
                wizard.windows = settings.get("windows", {})
                wizard.detectors = settings.get("detectors", {})

    def _apply_context_to_bva(self, widget: QtWidgets.QWidget) -> None:
        """Use upstream burst folder and channels in BVA."""
        settings = self.workflow_context.channel_settings
        detector_page = getattr(widget, "detector_page", None)
        if settings and detector_page is not None:
            try:
                detector_page.load_data_into_tables(settings)
                widget._refresh_detector_combos()
            except Exception:
                pass
        if self.workflow_context.burst_folder is not None:
            try:
                widget._set_folder(str(self.workflow_context.burst_folder))
            except Exception:
                pass

    def _apply_context_to_h2mm(self, widget: QtWidgets.QWidget) -> None:
        """Use upstream burst folder and channels in H2MM."""
        settings = self.workflow_context.channel_settings
        detector_page = getattr(widget, "detector_page", None)
        if settings and detector_page is not None:
            try:
                detector_page.load_data_into_tables(settings)
                widget._refresh_detector_combos()
            except Exception:
                pass
        if self.workflow_context.burst_folder is not None:
            try:
                widget._set_folder(str(self.workflow_context.burst_folder))
            except Exception:
                pass

    def _apply_context_to_mle(self, widget: QtWidgets.QWidget) -> None:
        """Use upstream burst files and channels in MLE Lifetime."""
        settings = self.workflow_context.channel_settings
        channel_definer = getattr(widget, "channel_definer", None)
        if settings and channel_definer is not None:
            try:
                channel_definer.load_data_into_tables(settings)
                widget._init_channels_from_wizard()
            except Exception:
                pass
        file_list = getattr(widget, "burst_files_list", None)
        if self.workflow_context.bur_files and file_list is not None and file_list.count() == 0:
            for path in self.workflow_context.bur_files:
                file_list.add_file(str(path))
            try:
                widget.update_burst_files()
            except Exception:
                pass

    def _apply_context_to_browser(self, widget: QtWidgets.QWidget) -> None:
        """Load upstream burst results in Burst Browser."""
        if self.workflow_context.burst_folder is None or getattr(widget, "_df", None) is not None:
            return
        try:
            widget.load_folder(self.workflow_context.burst_folder)
        except Exception:
            pass

    def _apply_context_to_background(self, widget: QtWidgets.QWidget) -> None:
        """Use selected raw files and channel setup in Background Estimation."""
        settings = self.workflow_context.channel_settings
        page = getattr(widget, "detector_wizard_page", None)
        if settings and page is not None:
            try:
                page.load_data_into_tables(settings)
            except Exception:
                pass
        if self.workflow_context.raw_files and not getattr(widget, "tttr_files", []):
            try:
                widget._add_tttr_files([str(path) for path in self.workflow_context.raw_files])
            except Exception:
                pass
