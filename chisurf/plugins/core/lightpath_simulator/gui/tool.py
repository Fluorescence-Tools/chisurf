import json
import logging
from typing import Any

import numpy as np
from qtpy import QtCore, QtWidgets

from chisurf.gui.widgets.dock_area.dock_area import DockArea
from chisurf.gui.widgets.general import apply_compact_table_style
from chisurf.gui.widgets.node_editor.scene import NodeScene
from chisurf.gui.widgets.node_editor.state_tracker import SceneStateTracker
from chisurf.gui.widgets.node_editor.view import NodeView
from chisurf.gui.widgets.node_editor.widgets.widget_palette import WidgetPalette
from chisurf.plugins.core.lightpath_simulator.api.client import LightPathClient
from chisurf.plugins.core.lightpath_simulator.core.workflow import resolve_db_path
from chisurf.plugins.core.lightpath_simulator.gui.easy_mode import (
    LightPathEasyDialog,
    LightPathEasyWidget,
    OPTICAL_PRESETS_DIR,
    save_easy_preset,
    load_easy_preset,
    _graph_to_config,
    _normalize_pid,
    normalize_lightpath_graph,
)
from chisurf.plugins.core.lightpath_simulator.gui.node_types import (
    build_optical_registry,
    get_forster_node_factory,
    get_light_source_factory,
    get_optical_node_factory,
    get_sample_node_factory,
    optical_registry,
)

logger = logging.getLogger(__name__)


class _ProbeInfoLoader(QtCore.QObject):
    """Load the MFDB spectra catalogue without blocking the GUI thread."""

    finished = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, timeout_ms: int = 1500):
        super().__init__()
        self.timeout_ms = timeout_ms

    @QtCore.Slot()
    def run(self) -> None:
        """Fetch probe metadata through the light-path RPC API."""
        client = None
        try:
            client = LightPathClient.from_settings(timeout_ms=self.timeout_ms)
            self.finished.emit(client.get_probes_info())
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if client is not None:
                client.close()

def _deserialize_numpy(obj: Any) -> Any:
    """Recursively converts lists of numbers back into NumPy arrays for plotting."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, (int, float)) for x in obj):
            return np.array(obj, dtype=np.float64)
        else:
            return [_deserialize_numpy(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _deserialize_numpy(v) for k, v in obj.items()}
    return obj


def _json_safe(obj: Any) -> Any:
    """Return a JSON-safe copy of graph state for RPC, settings, and files."""
    if callable(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if str(key).startswith("_") or callable(value):
                continue
            safe_value = _json_safe(value)
            if safe_value is not None:
                cleaned[key] = safe_value
        return cleaned
    if isinstance(obj, (list, tuple)):
        return [value for value in (_json_safe(item) for item in obj) if value is not None]
    return obj


class LightPathSimulatorWidget(QtWidgets.QMainWindow):
    """Main window for defining and simulating an optical light path."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Light Path Simulator")
        self.resize(1000, 700)
        self._is_propagating = False
        self._is_syncing_easy = False
        self._last_detector_signals = []
        self._last_crosstalk_matrices = {}
        self._last_instrument_setting = None
        self.probes = []
        self._probe_loader_thread = None
        self._probe_loader = None
        
        # Initialize the RPC client used by interactive commands.  Probe metadata
        # is loaded asynchronously below so opening the plugin never blocks on ZMQ.
        self.client = self.make_mfdb_client()
        
        # Ensure our node registry is populated
        build_optical_registry()
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.dock_area = DockArea(central)
        self.dock_area.setContextMenuEnabled(True)
        self.dock_area.setContextMenuMode("basic")
        self.dock_area.setTabsClosable(False)
        self.dock_area.layoutChanged.connect(self._save_dock_layout)
        layout.addWidget(self.dock_area, 1)

        self.graph_panel = QtWidgets.QWidget(self.dock_area)
        self.graph_panel.setObjectName("lightpathGraphDock")
        graph_layout = QtWidgets.QVBoxLayout(self.graph_panel)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        self.scene = NodeScene(self)
        self.scene.node_adder = self._on_add_node_requested

        self.view = NodeView(self.scene, self.graph_panel)
        graph_layout.addWidget(self.view)

        self.undo_stack = QtWidgets.QUndoStack(self)
        self.state_tracker = SceneStateTracker(self, self.scene, self.undo_stack)
        self.undo_stack.indexChanged.connect(lambda idx, s=self: s.propagate_graph())

        self.components_panel = QtWidgets.QWidget(self.dock_area)
        self.components_panel.setObjectName("lightpathComponentsDock")
        components_layout = QtWidgets.QVBoxLayout(self.components_panel)
        components_layout.setContentsMargins(4, 4, 4, 4)
        components_layout.setSpacing(4)

        self.palette = WidgetPalette(self.components_panel)
        self._load_optical_palette()
        self.palette.nodeTypeActivated.connect(self._on_palette_node_type_activated)
        components_layout.addWidget(self.palette, stretch=1)

        self.btn_reset = QtWidgets.QPushButton("Reset to Default", self.components_panel)
        self.btn_reset.clicked.connect(self._build_default_path)
        components_layout.addWidget(self.btn_reset)

        self.output_panel = QtWidgets.QWidget(self.dock_area)
        self.output_panel.setObjectName("lightpathEmissionDock")
        output_layout = QtWidgets.QVBoxLayout(self.output_panel)
        output_layout.setContentsMargins(4, 4, 4, 4)
        output_layout.setSpacing(4)

        self.btn_calculate = QtWidgets.QPushButton("Calculate Emission Intensity", self.output_panel)
        self.btn_calculate.clicked.connect(self.calculate_crosstalk)
        output_layout.addWidget(self.btn_calculate)

        self.results_tabs = QtWidgets.QTabWidget(self.output_panel)
        self.results_tabs.setDocumentMode(True)
        self.results_table = self._make_results_table()
        self.excitation_table = self._make_results_table()
        self.emission_table = self._make_results_table()
        self.detected_matrix_table = self._make_results_table()
        self.results_tabs.addTab(self.results_table, "Signals")
        self.results_tabs.addTab(self.excitation_table, "Excitation")
        self.results_tabs.addTab(self.emission_table, "Emission")
        self.results_tabs.addTab(self.detected_matrix_table, "Detected")
        output_layout.addWidget(self.results_tabs, stretch=1)

        self.easy_mode_widget = None
        self._build_docks()
        self._setup_menus()
        self._setup_toolbar()
        self._restore_window_geometry()
        self._restore_dock_layout()

        # Load MFDB spectra after the window has been constructed.
        QtCore.QTimer.singleShot(0, self._start_probe_loading)

    def make_mfdb_client(self):
        """Create an MFDB client using the current session token when available."""
        client = LightPathClient.from_settings(timeout_ms=1500)
        logger.info(
            "Light Path Simulator: created plugin RPC client",
        )
        return client

    def _settings(self) -> QtCore.QSettings:
        """Return the persistent settings store for this plugin."""
        from chisurf.gui.misc_helpers import get_plugin_settings_path

        ini_path = get_plugin_settings_path("lightpath_simulator")
        return QtCore.QSettings(str(ini_path), QtCore.QSettings.IniFormat)

    def _make_results_table(self) -> QtWidgets.QTableWidget:
        """Create a compact read-only results table matching the logging dock."""
        table = QtWidgets.QTableWidget(self.output_panel)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        apply_compact_table_style(table)
        return table

    def _build_docks(self) -> None:
        """Create the custom dock pages for graph, components, and output."""
        self.dock_area.addTab(self.graph_panel, "Optical Path")
        self.dock_area.addTab(self.components_panel, "Optical Components")
        self.dock_area.addTab(self.output_panel, "Emission Probability")
        self.dock_area.set_layout_state(self._default_dock_layout_state(), emit_change=False)

    def _default_dock_layout_state(self) -> dict[str, Any]:
        """Return the default three-pane custom dock arrangement."""
        return {
            "version": 1,
            "root": {
                "type": "splitter",
                "orientation": "horizontal",
                "sizes": [900, 340],
                "children": [
                    {
                        "type": "tab",
                        "tabs": [
                            {
                                "widget_key": "Optical Path",
                                "tab_name": "Optical Path",
                                "tab_text": "Optical Path",
                            }
                        ],
                        "current_index": 0,
                    },
                    {
                        "type": "splitter",
                        "orientation": "vertical",
                        "sizes": [330, 370],
                        "children": [
                            {
                                "type": "tab",
                                "tabs": [
                                    {
                                        "widget_key": "Optical Components",
                                        "tab_name": "Optical Components",
                                        "tab_text": "Optical Components",
                                    }
                                ],
                                "current_index": 0,
                            },
                            {
                                "type": "tab",
                                "tabs": [
                                    {
                                        "widget_key": "Emission Probability",
                                        "tab_name": "Emission Probability",
                                        "tab_text": "Emission Probability",
                                    }
                                ],
                                "current_index": 0,
                            },
                        ],
                    },
                ],
            },
            "active_tab_widget": [0],
            "current_index": 0,
        }

    def _save_window_geometry(self) -> None:
        """Save the main window geometry."""
        try:
            settings = self._settings()
            settings.setValue("geometry", self.saveGeometry())
            settings.sync()
        except Exception as exc:
            logger.error("Failed to save lightpath window geometry: %s", exc)

    def _restore_window_geometry(self) -> None:
        """Restore the main window geometry."""
        try:
            geometry = self._settings().value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
        except Exception as exc:
            logger.error("Failed to restore lightpath window geometry: %s", exc)

    def _save_dock_layout(self) -> None:
        """Save the custom dock layout."""
        try:
            settings = self._settings()
            settings.setValue("dock_layout", json.dumps(self.dock_area.get_layout_state(), sort_keys=True))
            settings.sync()
        except Exception as exc:
            logger.error("Failed to save lightpath dock layout: %s", exc)

    def _restore_dock_layout(self) -> None:
        """Restore the custom dock layout."""
        try:
            value = self._settings().value("dock_layout")
            if isinstance(value, str):
                layout_state = json.loads(value)
            elif isinstance(value, dict):
                layout_state = value
            else:
                return
            self.dock_area.set_layout_state(layout_state, emit_change=False)
        except Exception as exc:
            logger.error("Failed to restore lightpath dock layout: %s", exc)

    def _save_graph_state(self) -> None:
        """Save the node graph and widget population."""
        try:
            settings = self._settings()
            graph = _json_safe(self.scene.to_dict())
            if not graph.get("nodes"):
                return
            settings.setValue("graph", json.dumps(graph, sort_keys=True))
            settings.sync()
        except Exception as exc:
            logger.error("Failed to save lightpath graph state: %s", exc)

    def _start_probe_loading(self) -> None:
        """Start the asynchronous MFDB probe catalogue load."""
        if self._probe_loader_thread is not None:
            return

        thread = QtCore.QThread(self)
        loader = _ProbeInfoLoader(timeout_ms=1500)
        loader.moveToThread(thread)

        thread.started.connect(loader.run)
        loader.finished.connect(self._on_probes_loaded)
        loader.failed.connect(self._on_probe_load_failed)
        loader.finished.connect(thread.quit)
        loader.failed.connect(thread.quit)
        loader.finished.connect(loader.deleteLater)
        loader.failed.connect(loader.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_probe_loader)

        self._probe_loader_thread = thread
        self._probe_loader = loader
        thread.start()

    def _clear_probe_loader(self) -> None:
        """Clear finished probe loader references."""
        self._probe_loader_thread = None
        self._probe_loader = None

    def _on_probes_loaded(self, probes: list[dict[str, Any]]) -> None:
        """Install loaded probe metadata and build the startup graph."""
        self.probes = probes or []
        logger.info("Loaded %d light-path MFDB spectra entries", len(self.probes))
        self._build_easy_mode_tab()
        self._setup_default_path()

    def _on_probe_load_failed(self, message: str) -> None:
        """Open the GUI even if the spectra catalogue cannot be loaded."""
        logger.error("Failed to fetch probe info from backend: %s", message)
        self.probes = []
        self._setup_default_path()

    def _open_easy_mode_dialog(self):
        """Open the easy mode as a standalone dialog."""
        dlg = LightPathEasyDialog(
            self.probes, self,
            db_path=resolve_db_path(),
        )
        if dlg.exec_():
            cfg = dlg.get_optical_config()
            path_name = cfg.get("optical_path_name", "")
            if path_name:
                path = OPTICAL_PRESETS_DIR / f"{path_name}.json"
                if path.exists():
                    try:
                        graph = load_easy_preset(str(path))
                        dyes = cfg.get("dyes", {})
                        for n in graph.get("nodes", []):
                            if n["type"] == "sample":
                                n["config"]["dye_properties"] = dyes
                                n["config"]["probe_ids"] = [_normalize_pid(p) for p in dyes]
                                n["config"]["probe_id"] = _normalize_pid(next(iter(dyes))) if dyes else None
                            if n["type"] == "forster_radius":
                                n["config"]["kappa2"] = cfg.get("kappa2", 0.6667)
                                n["config"]["n"] = cfg.get("n", 1.33)
                        self.load_graph_from_dict(graph)
                    except Exception:
                        pass

    def _setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        easy_act = QtWidgets.QAction("Easy Mode (Dialog)…", self)
        easy_act.triggered.connect(self._open_easy_mode_dialog)
        file_menu.addAction(easy_act)
        file_menu.addSeparator()
        
        export_act = QtWidgets.QAction("Export Instrument Setting (JSON)...", self)
        export_act.triggered.connect(self._on_export_json)
        file_menu.addAction(export_act)
        
        file_menu.addSeparator()

        save_mfdb_act = QtWidgets.QAction("Save Simulation to MFDB...", self)
        save_mfdb_act.triggered.connect(self._on_save_to_mfdb)
        file_menu.addAction(save_mfdb_act)

        load_mfdb_act = QtWidgets.QAction("Load Simulation from MFDB...", self)
        load_mfdb_act.triggered.connect(self._on_load_from_mfdb)
        file_menu.addAction(load_mfdb_act)

        file_menu.addSeparator()

        save_act = QtWidgets.QAction("Save Graph...", self)
        save_act.triggered.connect(self._on_save_graph)
        file_menu.addAction(save_act)
        
        load_act = QtWidgets.QAction("Load Graph...", self)
        load_act.triggered.connect(self._on_load_graph)
        file_menu.addAction(load_act)

        file_menu.addSeparator()

        save_preset_act = QtWidgets.QAction("Save as Optical Path Preset...", self)
        save_preset_act.triggered.connect(self._on_save_optical_preset)
        file_menu.addAction(save_preset_act)

    def _on_export_json(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Instrument Setting", "", "JSON (*.json)")
        if path:
            self.propagate_graph()
            if self._last_instrument_setting:
                try:
                    with open(path, "w") as f:
                        json.dump(self._last_instrument_setting, f, indent=2)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "Export Failed", f"Could not save file:\n{e}")
            else:
                QtWidgets.QMessageBox.warning(self, "Export Warning", "No instrument setting has been simulated yet.")

    def _on_save_optical_preset(self):
        """Save the current graph as a named preset for use in easy mode."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Optical Path Preset",
            "Preset name:",
        )
        if not ok or not name:
            return
        OPTICAL_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
        from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
        graph = _json_safe(self.scene.to_dict())
        # Strip cached/transient data from nodes
        for n in graph.get("nodes", []):
            cfg = n.get("config", {})
            for key in list(cfg.keys()):
                if key.startswith("_"):
                    del cfg[key]
        save_easy_preset(graph, str(OPTICAL_PRESETS_DIR / f"{name.strip()}.json"))
        # Refresh the Easy Mode preset combo so the new preset appears immediately
        if self.easy_mode_widget is not None:
            self.easy_mode_widget._refresh_preset_list()

    def _on_save_graph(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Graph", "", "JSON (*.json)")
        if path:
            try:
                state = _json_safe(self.scene.to_dict())
                with open(path, "w") as f:
                    json.dump(state, f, indent=2)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Failed", f"Could not save graph:\n{e}")

    def _on_load_graph(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Graph", "", "JSON (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                self.scene.from_dict(normalize_lightpath_graph(state))
                self.propagate_graph()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Failed", f"Could not load graph:\n{e}")

    def _on_save_to_mfdb(self):
        """Persist the current graph and simulation result as MFDB artifacts."""
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Simulation to MFDB",
            "Name:",
            text="Light path simulation",
        )
        if not ok:
            return
        graph = _json_safe(self.scene.to_dict())
        self.propagate_graph(graph)
        try:
            res = self.client.save(graph, name=name or "Light path simulation")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "MFDB Save Failed", str(exc))
            return
        QtWidgets.QMessageBox.information(
            self,
            "Saved to MFDB",
            f"Saved operation {res.get('operation_id')}",
        )

    def _on_load_from_mfdb(self):
        """Load a previously saved lightpath graph from MFDB."""
        try:
            records = self.client.list_saved()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "MFDB Load Failed", str(exc))
            return
        if not records:
            QtWidgets.QMessageBox.information(self, "MFDB", "No saved light path simulations found.")
            return
        labels = [
            f"{item.get('name') or item.get('operation_id')} ({item.get('operation_id')})"
            for item in records
        ]
        label, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Load Simulation from MFDB",
            "Simulation:",
            labels,
            0,
            False,
        )
        if not ok or not label:
            return
        operation_id = records[labels.index(label)]["operation_id"]
        try:
            load_res = self.client.get(operation_id)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "MFDB Load Failed", str(exc))
            return
        self.scene.clear()
        self.scene.from_dict(normalize_lightpath_graph(load_res["graph"]))
        self.propagate_graph()

    def _setup_toolbar(self) -> None:
        """Create the save/load toolbar on the optical path dock."""
        toolbar = QtWidgets.QToolBar("Graph", self)
        toolbar.setObjectName("lightpathToolbar")
        toolbar.setIconSize(QtCore.QSize(16, 16))
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        toolbar.setStyleSheet("""
            QToolBar {
                background: rgb(45, 49, 58);
                border: none;
                border-bottom: 1px solid rgb(55, 60, 70);
                spacing: 2px;
                padding: 2px 4px;
            }
            QToolButton {
                color: rgb(235, 235, 235);
                background: rgb(60, 65, 75);
                border: 1px solid rgb(50, 55, 65);
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
                min-height: 22px;
            }
            QToolButton:hover {
                background: rgb(72, 80, 95);
                border: 1px solid rgb(60, 120, 160);
            }
            QToolButton:pressed {
                background: rgb(50, 55, 65);
            }
        """)

        save_act = QtWidgets.QAction("Save Graph", self)
        save_act.setToolTip("Save the current graph to a JSON file")
        save_act.triggered.connect(self._on_save_graph)

        load_act = QtWidgets.QAction("Load Graph", self)
        load_act.setToolTip("Load a graph from a JSON file")
        load_act.triggered.connect(self._on_load_graph)

        save_preset_act = QtWidgets.QAction("Save Preset", self)
        save_preset_act.setToolTip("Save as optical path preset for the Easy Mode")
        save_preset_act.triggered.connect(self._on_save_optical_preset)

        save_mfdb_act = QtWidgets.QAction("Save to MFDB", self)
        save_mfdb_act.setToolTip("Persist the current graph and simulation to MFDB")
        save_mfdb_act.triggered.connect(self._on_save_to_mfdb)

        load_mfdb_act = QtWidgets.QAction("Load from MFDB", self)
        load_mfdb_act.setToolTip("Load a previously saved graph from MFDB")
        load_mfdb_act.triggered.connect(self._on_load_from_mfdb)

        toolbar.addAction(save_act)
        toolbar.addAction(load_act)
        toolbar.addAction(save_preset_act)
        toolbar.addSeparator()
        toolbar.addAction(save_mfdb_act)
        toolbar.addAction(load_mfdb_act)

        self.addToolBar(toolbar)

    def _load_optical_palette(self):
        """Build a custom hierarchy for the palette widget using our optical registry."""
        self.palette.clear()
        group_item = QtWidgets.QTreeWidgetItem(["Optical Path"])
        group_item.setFlags(group_item.flags() & ~QtCore.Qt.ItemIsSelectable)
        self.palette.addTopLevelItem(group_item)
        
        for n_id, n_type in optical_registry.all_types().items():
            child = QtWidgets.QTreeWidgetItem([n_type.title])
            child.setData(0, QtCore.Qt.UserRole, n_id)
            group_item.addChild(child)
            
        group_item.setExpanded(True)

    def _on_palette_node_type_activated(self, node_type_id: str):
        """Drops a node into the center of the current view."""
        self.state_tracker.begin_action(f"Add {node_type_id}")
        view_center = self.view.viewport().rect().center()
        scene_pos = self.view.mapToScene(view_center)
        item = self._on_add_node_requested(node_type_id, scene_pos)
        if item is not None:
            self.scene.addItem(item)
        self.state_tracker.commit_action()

    def _on_add_node_requested(self, node_type_id: str, pos: QtCore.QPointF):
        """Hook called by the scene when a node needs to be created, routing to our registry."""
        node_type = optical_registry.get(node_type_id)
        if node_type is None:
            return None
        
        from chisurf.gui.widgets.node_editor.model import NodeModel
        from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
        
        factory_generator = None
        if node_type.id == "light_source":
            factory_generator = get_light_source_factory(self.probes)
        elif node_type.id == "sample":
            factory_generator = get_sample_node_factory(self.probes)
        elif node_type.id == "filter":
            factory_generator = get_optical_node_factory(self.probes, "transmission", "Filter:")
        elif node_type.id == "splitter":
            factory_generator = get_optical_node_factory(self.probes, "transmission", "Splitter:")
        elif node_type.id == "detector":
            factory_generator = get_optical_node_factory(self.probes, "quantum_efficiency", "Detector:")
        elif node_type.id == "combiner":
            factory_generator = lambda config: QtWidgets.QWidget() # Stub
        elif node_type.id == "forster_radius":
            factory_generator = get_forster_node_factory()

        model_config = node_type.default_config.copy()
        
        content_factory = None
        if factory_generator is not None:
             content_factory = lambda config=model_config, f=factory_generator: f(config)
             
        model = NodeModel(
            node_type=node_type.id,
            title=node_type.title,
            inputs=node_type.inputs,
            outputs=node_type.outputs,
            config=model_config,
            content_factory=content_factory
        )
        
        item = NodeGraphicsItem(model, width=node_type.width)
        item.setPos(pos.x(), pos.y())
        return item

    def propagate_graph(self, state=None):
        """Propagate spectral signals from sources through the graph via ZMQ RPC."""
        if self._is_propagating:
            return
            
        self._is_propagating = True
        try:
            if not state:
                try:
                    state = _json_safe(self.scene.to_dict())
                except RuntimeError:
                    return
            else:
                state = _json_safe(state)

            # Call backend RPC simulation
            try:
                res = self.client.simulate(state)
            except Exception as exc:
                logger.error(f"ZMQ RPC simulation call failed: {exc}")
                return

            if not res:
                logger.error("Simulation failed: Empty response")
                return

            sim_states = res.get("states", {})
            self._last_detector_signals = res.get("detector_signals", [])
            self._last_crosstalk_matrices = res.get("crosstalk_matrices", {})
            self._last_instrument_setting = res.get("instrument_setting")
            self._update_output_tables()

            # Sync back to live nodes
            from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
            id_to_item = {it.model.id: it for it in self.scene.items() if isinstance(it, NodeGraphicsItem)}
            
            for n_id, ns_dict in sim_states.items():
                if n_id not in id_to_item: continue
                it = id_to_item[n_id]
                
                # Deserialise array values from backend response
                input_spectra = _deserialize_numpy(ns_dict.get("input_spectra", {}))
                output_spectra = _deserialize_numpy(ns_dict.get("output_spectra", {}))
                
                node_char = ns_dict.get("node_char")
                if isinstance(node_char, list):
                    if len(node_char) == 2 and isinstance(node_char[0], list):
                        node_char = (np.array(node_char[0], dtype=np.float64), np.array(node_char[1], dtype=np.float64))
                    else:
                        node_char = np.array(node_char, dtype=np.float64)
                
                # Update live config for plots
                it.model.config.update({
                    "_input_spectra": input_spectra,
                    "_output_spectra": output_spectra,
                    "_node_char": node_char,
                    "_last_signals": ns_dict.get("config", {}).get("_last_signals", {}),
                    "_last_results": ns_dict.get("config", {}).get("_last_results", [])
                })
                if "_update_plot" in it.model.config:
                    it.model.config["_update_plot"]()

            self._sync_easy_mode_from_graph()
        finally:
            self._is_propagating = False

    def calculate_crosstalk(self):
        """Builds result table based on propagated signals."""
        self.propagate_graph()
        self._update_output_tables()

    def _update_output_tables(self) -> None:
        """Refresh all emission probability and crosstalk tables."""
        self._populate_signal_table(self._last_detector_signals)
        matrices = self._last_crosstalk_matrices or {}
        self._populate_matrix_table(self.excitation_table, matrices.get("excitation", {}))
        self._populate_matrix_table(self.emission_table, matrices.get("emission", {}))
        self._populate_matrix_table(self.detected_matrix_table, matrices.get("detected", {}))

    def _populate_signal_table(self, row_data: list[dict[str, Any]]) -> None:
        """Populate the detailed detector signal rows."""
        headers = ["Laser Source", "Detector Name", "Dye", "Detected Intensity"]
        self.results_table.setRowCount(len(row_data))
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        
        for r_idx, row in enumerate(row_data):
            self.results_table.setItem(r_idx, 0, QtWidgets.QTableWidgetItem(row["laser"]))
            self.results_table.setItem(r_idx, 1, QtWidgets.QTableWidgetItem(row["detector"]))
            self.results_table.setItem(r_idx, 2, QtWidgets.QTableWidgetItem(row["dye"]))
            
            val_item = QtWidgets.QTableWidgetItem(f"{row['intensity']:.4e}")
            val_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.results_table.setItem(r_idx, 3, val_item)

    def _populate_matrix_table(self, table: QtWidgets.QTableWidget, matrix: dict[str, Any]) -> None:
        """Populate a compact matrix table from a backend matrix payload."""
        rows = matrix.get("rows") or []
        columns = matrix.get("columns") or []
        values = matrix.get("values") or []
        table.setRowCount(len(rows))
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([str(column) for column in columns])
        table.setVerticalHeaderLabels([str(row) for row in rows])
        table.verticalHeader().setVisible(True)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

        for row_idx, row_values in enumerate(values):
            for col_idx, value in enumerate(row_values):
                item = QtWidgets.QTableWidgetItem(f"{float(value):.4e}")
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                table.setItem(row_idx, col_idx, item)

    # Required interface for state_tracker
    def begin_undo(self, text: str) -> None:
        self.state_tracker.begin_action(text)

    def commit_undo(self) -> None:
        self.state_tracker.commit_action()
        self.propagate_graph()

    def closeEvent(self, event):
        """Save graph state, dock layout, and geometry on close."""
        self._save_window_geometry()
        self._save_dock_layout()
        self._save_graph_state()

        thread = self._probe_loader_thread
        if thread is not None and thread.isRunning():
            thread.quit()
            if not thread.wait(3000):
                logger.warning("Probe catalogue loader did not stop before close")
        self.client.close()
            
        super().closeEvent(event)

    def _create_content_factory_for_json_node(self, node_type_id: str, config: dict) -> Any:
        """Constructs content factories when restoring from saved dict state."""
        factory_generator = None
        if node_type_id == "light_source":
            factory_generator = get_light_source_factory(self.probes)
        elif node_type_id == "sample":
            factory_generator = get_sample_node_factory(self.probes)
        elif node_type_id == "filter":
            factory_generator = get_optical_node_factory(self.probes, "transmission", "Filter:")
        elif node_type_id == "splitter":
            factory_generator = get_optical_node_factory(self.probes, "transmission", "Splitter:")
        elif node_type_id == "detector":
            factory_generator = get_optical_node_factory(self.probes, "quantum_efficiency", "Detector:")
        elif node_type_id == "combiner":
            factory_generator = lambda config: QtWidgets.QWidget() # Stub
        elif node_type_id == "forster_radius":
            factory_generator = get_forster_node_factory()
            
        if factory_generator is not None:
            return lambda config=config, f=factory_generator: f(config)
        return None

    def _setup_default_path(self):
        """Restores the saved graph layout, or builds the default path if none exists."""
        try:
            saved_graph = self._settings().value("graph")
            if saved_graph:
                try:
                    state = json.loads(saved_graph) if isinstance(saved_graph, str) else saved_graph
                    if not isinstance(state, dict) or not state.get("nodes"):
                        raise ValueError("saved graph has no nodes")
                    self.scene.clear()
                    self.scene.from_dict(normalize_lightpath_graph(state))
                    self.propagate_graph()
                    self.view.centerOn(550, 200)
                    return
                except Exception as e:
                    logger.error(f"Failed to load saved graph: {e}")
        except Exception as e:
            logger.error(f"Error checking for saved graph: {e}")

        self._build_default_path()

    def _build_easy_mode_tab(self):
        """Add the easy-mode tab to the dock area."""
        if self.easy_mode_widget is not None:
            return
        self.easy_mode_widget = LightPathEasyWidget(self.probes, self, db_path=resolve_db_path())
        self.dock_area.addTab(self.easy_mode_widget, "Easy Mode")

    def _sync_easy_mode_from_graph(self):
        """Sync the Easy Mode form to reflect the current Full Simulator graph."""
        if self.easy_mode_widget is None or self._is_syncing_easy:
            return
        try:
            graph = _json_safe(self.scene.to_dict())
        except RuntimeError:
            return
        if not graph.get("nodes"):
            return
        cfg = _graph_to_config(graph)
        self.easy_mode_widget._suppress_recalc = True
        self.easy_mode_widget._suppress_form_sync = True
        try:
            self.easy_mode_widget._populate_form(cfg)
        finally:
            self.easy_mode_widget._suppress_recalc = False
            self.easy_mode_widget._suppress_form_sync = False
        if self.easy_mode_widget.auto_recalc_cb.isChecked():
            self.easy_mode_widget.recalculate()

    def load_graph_from_dict(self, graph: dict) -> None:
        """Load a graph dict into the node editor and propagate."""
        self.scene.from_dict(normalize_lightpath_graph(graph))
        self.propagate_graph()

    def _update_easy_config_in_place(self, cfg: dict) -> None:
        """Update existing graph node configs from Easy Mode without rearranging."""
        from chisurf.gui.widgets.node_editor.node_item import NodeGraphicsItem
        items = [it for it in self.scene.items() if isinstance(it, NodeGraphicsItem)]

        for it in items:
            if it.model.node_type == "light_source":
                it.model.config["source_mode"] = "manual"
                it.model.config["manual_lines"] = cfg.get("lasers", "488:1.0, 640:1.0")

            elif it.model.node_type == "sample":
                dye_ids = [_normalize_pid(p) for p in cfg.get("dyes", {})]
                it.model.config["probe_ids"] = dye_ids
                it.model.config["probe_id"] = dye_ids[0] if dye_ids else None
                it.model.config["dye_properties"] = cfg.get("dyes", {})

            elif it.model.node_type == "forster_radius":
                it.model.config["kappa2"] = cfg.get("kappa2", 0.6667)
                it.model.config["n"] = cfg.get("n", 1.33)

        # Splitters — sort by position for cascade order
        splitters = sorted(
            [it for it in items if it.model.node_type == "splitter"],
            key=lambda it: (it.pos().x(), it.pos().y()),
        )
        if splitters:
            splitters[0].model.config["probe_id"] = _normalize_pid(
                cfg.get("excitation_dichroic_probe_id")
            )
        em_splitters = cfg.get("emission_splitters", [])
        for i, sp in enumerate(splitters[1:], start=1):
            if i - 1 < len(em_splitters):
                sp.model.config["probe_id"] = _normalize_pid(
                    em_splitters[i - 1].get("probe_id")
                )
                sp.model.config["splitter_type"] = em_splitters[i - 1].get("type", "Dichroic")

        # Detectors — sort by position
        detectors = sorted(
            [it for it in items if it.model.node_type == "detector"],
            key=lambda it: (it.pos().x(), it.pos().y()),
        )
        easy_dets = cfg.get("detectors", [])
        for i, det in enumerate(detectors):
            if i < len(easy_dets):
                det.model.config["detector_name"] = easy_dets[i].get(
                    "name", f"Channel {i + 1}"
                )

        # Filters — sort by position; update probe_id from detector config
        filters = sorted(
            [it for it in items if it.model.node_type == "filter"],
            key=lambda it: (it.pos().x(), it.pos().y()),
        )
        for i, f in enumerate(filters):
            if i < len(easy_dets):
                bp_pid = _normalize_pid(easy_dets[i].get("bandpass_probe_id"))
                f.model.config["probe_id"] = bp_pid

        for it in items:
            if "_update_plot" in it.model.config:
                it.model.config["_update_plot"]()

        self.propagate_graph()

    def _build_default_path(self):
        """Build the canonical default light-path graph."""
        # Programmatically builds a standard Laser -> Sample -> Dichroic -> 2 Detectors path.
        self.scene.clear()
        
        # Try to find some reasonable defaults from pre-fetched probes metadata
        green_dye_id = None
        red_dye_id = None
        dichroic_id = None
        green_filter_id = None
        red_filter_id = None
        
        for p in self.probes:
            i_id = p["probe_id"]
            i_name = p["name"]
            name_lower = i_name.lower().replace("-", " ")
            if "atto 488" in name_lower and not green_dye_id and p.get("has_em"):
                green_dye_id = i_id
            if "atto 647n" in name_lower and not red_dye_id and p.get("has_em"):
                red_dye_id = i_id
            if "561lp" in name_lower and not dichroic_id and p.get("has_trans"):
                dichroic_id = i_id
            if "bp" in name_lower and "500" in name_lower and not green_filter_id and p.get("has_trans"):
                green_filter_id = i_id
            if "bp" in name_lower and "650" in name_lower and not red_filter_id and p.get("has_trans"):
                red_filter_id = i_id

        # 1. Light Source
        ls = self._on_add_node_requested("light_source", QtCore.QPointF(50, 200))
        ls_cfg = ls.model.config
        ls_cfg["source_mode"] = "manual"
        ls_cfg["manual_lines"] = "488:1.0, 640:1.0"
        self.scene.addItem(ls)
        
        # 2. Sample
        sample = self._on_add_node_requested("sample", QtCore.QPointF(300, 200))
        sample_cfg = sample.model.config
        dyes = []
        if green_dye_id: dyes.append(green_dye_id)
        if red_dye_id: dyes.append(red_dye_id)
        if dyes: sample_cfg["probe_ids"] = dyes
        self.scene.addItem(sample)
        
        # 3. Splitter
        dichroic = self._on_add_node_requested("splitter", QtCore.QPointF(550, 200))
        if dichroic_id: dichroic.model.config["probe_id"] = dichroic_id
        self.scene.addItem(dichroic)
        
        # 4. Red Path (Transmitted if > 561nm)
        red_f = self._on_add_node_requested("filter", QtCore.QPointF(800, 100))
        if red_filter_id: red_f.model.config["probe_id"] = red_filter_id
        self.scene.addItem(red_f)
        
        red_det = self._on_add_node_requested("detector", QtCore.QPointF(1050, 100))
        red_det.model.config["detector_name"] = "Red Channel"
        self.scene.addItem(red_det)
        
        # 5. Green Path (Reflected if < 561nm)
        green_f = self._on_add_node_requested("filter", QtCore.QPointF(800, 300))
        if green_filter_id: green_f.model.config["probe_id"] = green_filter_id
        self.scene.addItem(green_f)
        
        green_det = self._on_add_node_requested("detector", QtCore.QPointF(1050, 300))
        green_det.model.config["detector_name"] = "Green Channel"
        self.scene.addItem(green_det)
        
        from chisurf.gui.widgets.node_editor.edge_item import EdgeGraphicsItem
        
        def connect(src_item, src_port_idx, dst_item, dst_port_idx):
            edge = EdgeGraphicsItem(src_item.port_items[src_port_idx])
            edge.set_end_port(dst_item.port_items[dst_port_idx])
            self.scene.addItem(edge)
            self.scene.register_edge(edge)

        # Wire it up
        connect(ls, 0, sample, 0)
        connect(sample, 1, dichroic, 0)
        
        # Forster Radius connection
        forster = self._on_add_node_requested("forster_radius", QtCore.QPointF(300, 400))
        self.scene.addItem(forster)
        connect(sample, 2, forster, 0) # Sample Dye Data -> Forster Dye Data
        
        # A 561LP dummy transmits red (long pass) and reflects green (short)
        connect(dichroic, 1, red_f, 0) # Splitter Trans -> Red Filter
        connect(red_f, 1, red_det, 0)
        
        connect(dichroic, 2, green_f, 0) # Splitter Refl -> Green Filter
        connect(green_f, 1, green_det, 0)
        
        self.view.centerOn(550, 200)
        self.propagate_graph()
